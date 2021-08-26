import os
import glob
from reading import *
import subprocess

def cleaner(fn_i,fn_o,overwrite=False):
    """remove everything until the last load (ld) path, and remove the last ld path too

    fn_i  --  filename input
    fn_o  --  filename output
    fn_i+".TEMPORARYFILE.TMP"  --  name for temporary file
    """

    if fn_o == fn_i:
        assert overwrite
    if overwrite:
        assert fn_i == fn_o
        fn_o = fn_i+".TEMPORARYFILE.TMP"   # name for temporary file
        # move back at the end
    if os.path.exists("fn_o"):
        assert overwrite

    with open(fn_i, 'r') as fi:
        lines = fi.readlines()
    
    fo = open(fn_o, 'w')
    
    printer = False
    for line in lines:
        if 'ld' in line:
            printer = False
            # This erases all previous
            fo.close()
            fo = open(fn_o, 'w')
        elif 'ACC' in line and 'sh' in line:
            printer = True
        if printer:
            fo.write(line)

    fi.close()
    fo.close()

    if overwrite:
        os.remove(fn_i)
        os.rename(fn_o, fn_i)
        print("overwrite file...",fn_i,"(was temporarily %s)"%fn_o)
    else:
        print("file written...",fn_o)

def sieve_trajectories(folders, n_clean, max_cycnum):
    rmdirs = []
    shdirs = []
    for fol in folders:
        toremove_list, tokeep_list = sieve_trajectories_from_folder(fol, n_clean = n_clean, max_cycnum=max_cycnum)
        rmdirs+=toremove_list
        shdirs+=tokeep_list
    
    alldirs = rmdirs+shdirs

    return rmdirs, shdirs, alldirs

def sieve_trajectories_from_folder(indir,pe_fn="pathensemble.txt",n_clean=0,max_cycnum=999999999):
    """
        indir is an ensemble folder (like .../000/)
        Returns a list of directories containing the non-shoot trajectories of this ensemble.
        
        pe_fn: filename of the pathensemble.txt file (I don't know, this may change)
        indir: directory where pe_fn is located
    """
    
    ACCFLAGS, REJFLAGS = set_flags_ACC_REJ()

    pe = read_pathensemble(indir+"/"+pe_fn)
    toremove_list = []
    tokeep_list = []
    assert len(pe.cyclenumbers) == len(pe.generation)
    for flag, cycnum, gen in zip(pe.flags,pe.cyclenumbers,pe.generation):
        if flag in ACCFLAGS:
            flagstr = "traj-acc"
        elif flag in REJFLAGS:
            flagstr = "traj-rej"
        else:
            raise NotImplementedError("Paths are either rejected or accepted. Path "+str(cycnum)+" of ensemble "+str(indir)+" does not even belong to the rejected, but to the flag: "+str(flag))
        dirstring = indir+"/traj/"+flagstr+"/"+str(cycnum)+"/"
        if cycnum >= n_clean and cycnum < max_cycnum:
            if gen != 'sh':
                toremove_list.append(dirstring)
            else:
                tokeep_list.append(dirstring)
    return toremove_list,tokeep_list


def remove_nonshoot_trajectories(dirlist,outputfolder,check_type=True,filetypes=[".xtc",".trr",".gro"],donotremove=True,stop_on_error=True):
    """
        Removes the nonshoot trajectories in the folders given by dirlist
    """
    with open (outputfolder+"/removedtrajs.txt", "a+") as ff:
        for trajdir in dirlist:
            if os.path.isdir(trajdir):
                is_shoot, moveline = check_if_ordertxt_is_shoot(trajdir)
                if not is_shoot:
                    filelist = glob.glob(trajdir+"traj/*")
                    if filelist: # AKA if filelist is not emty
                        for f in filelist:
                            if check_type == True:
                                fname,fext = os.path.splitext(f)
                                assert fext in filetypes, "Not a trajectory!! : "+f
                                if donotremove:
                                    ff.write("would have removed: "+f+"\n\tmoveline: "+moveline)
                                    ff.write("\n")
                                else:
                                    ff.write("removed: "+f+"\n\tmoveline: "+moveline)
                                    os.remove(f)
                            else:
                                if donotremove:
                                    ff.write("would have removed: "+f+"\n\tmoveline: "+moveline)
                                    ff.write("\n")
                                else:
                                    ff.write("removed: "+f+"\n\tmoveline: "+moveline)
                                    os.remove(f)
                else:
                    ff.write("This is a shooting move!\n")
                    ff.write(moveline+"\n")
                    if stop_on_error:
                        assert False, "You tried to delete a shooting move trajectory, stopped deleting. Check removedtrajs.txt"
            else: #if trajdir does not exist
                ff.write(trajdir +" does not exist\n")
                if stop_on_error:
                    assert False, "you wanted to delete a non-existing file, stopped deleting files. Check removedtrajs.txt"
    ff.close()

def check_if_ordertxt_is_shoot(trajdir):
    with open(trajdir+"/order.txt", "r") as f:
        moveline = f.readline()
        shootstr = "move: (\'sh\',"
        shootcheck = shootstr in moveline
    f.close()    
    return shootcheck, moveline

def subsample_trajectories(trajectories, dt, ndx_fn, tpr_fn, procs=1):
    if procs > 1:
        chunk_list = split_list(trajectories, procs)
    else:
        chunk_list = [trajectories]

    proc_list = []

    for i, chunk_trr in enumerate(chunk_list):
        fn = write_chunk_trr(chunk_trr, str(i))
        cmd = ["./subsampler_chunk", '-f', fn, '-n', ndx_fn, '-s', tpr_fn, '-t', dt, '-c', str(i)]
        proc_list.append(subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE))

    for proc in proc_list:
        proc.communicate()
    
    return_codes = [None]*procs
    
    all_went_well = True

    for i,proc in enumerate(proc_list):
        return_codes[i] = proc.returncode
        if return_codes[i] is not None and return_codes[i] == 0:
            print("subsampling chunk "+str(i)+" went well.")
        else:
            all_went_well = False
            print("subsampling chunk "+str(i)+ "went BAD.")

    return all_went_well

def write_chunk_trr(trr_files, chunk_id):
    fn = "chunk_"+chunk_id+"_trr_paths.txt"
    with open(fn, "w+") as f:
        for el in trr_files:
            f.write(el+"\n")
    f.close()
    return fn

def read_chunk_trr(fn):
    trr_files = []
    with open(fn, 'r') as f:
        for line in f:
            trr_files.append(line.strip())
    f.close()
    return trr_files

def split_list(l,n):
    #split list l with L elements into chunks of size ~ L/n
    k,m=divmod(len(l),n)
    return [l[i*k+min(i,m):(i+1)*k+min(i+1,m)] for i in range(n)]

def get_max_cycnum_ensemble(indir,pe_fn="pathensemble.txt"):
    pe = read_pathensemble(indir+"/"+pe_fn)
    max_cycnum = 0
    #Maybe just np.max(pe.cyclenumbers), I don't know
    for cycnum in pe.cyclenumbers:
        if cycnum > max_cycnum:
            max_cycnum = cycnum
    return max_cycnum

def get_max_cycnum_simul(folders):
    # We will actually return max_cynum, as we don't want to move a file that is still being written to...
    max_cycnum = 0
    for fol in folders:
        trial_cycnum = get_max_cycnum_ensemble(fol)
        if trial_cycnum > max_cycnum:
            max_cycnum = trial_cycnum
    return max_cycnum

def list_trajectory_folders_lt_maxnum(indir,max_cycnum,pe_fn="pathensemble.txt"):
    """
        indir is an ensemble folder (like .../000/)
        Returns a list of directories containing the non-shoot trajectories of this ensemble.
        
        pe_fn: filename of the pathensemble.txt file (I don't know, this may change)
        indir: directory where pe_fn is located
        max_cycnum: only cycnums less than max_cycnum will be moved
    """
    
    ACCFLAGS, REJFLAGS = set_flags_ACC_REJ()

    pe = read_pathensemble(indir+"/"+pe_fn)
    move_list = []
    assert len(pe.cyclenumbers) == len(pe.generation)
    for flag, cycnum, gen in zip(pe.flags,pe.cyclenumbers,pe.generation):
        if flag in ACCFLAGS:
            flagstr = "traj-acc"
        elif flag in REJFLAGS:
            flagstr = "traj-rej"
        else:
            raise NotImplementedError("Paths are either rejected or accepted. Path "+str(cycnum)+" of ensemble "+str(indir)+" does not even belong to the rejected, but to the flag: "+str(flag))
        if cycnum < max_cycnum:
            dirstring = indir+"/traj/"+flagstr+"/"+str(cycnum)+"/"
            move_list.append(dirstring)
    return move_list

def move_trajectories(traj_dirs, target_dir,verbose=True):
    for movedir in traj_dirs:
        temp_target = target_dir+"/"+movedir
        os.system('mkdir -p '+temp_target)
        os.system('mv '+movedir+'/* '+temp_target)
        print("moved: \nfrom: "+movedir+"\nto: "+temp_target)

def execute_command(cmd, cwd=None, inputs=None, fn=''):
    """
    Command is executed, after which we wait until it is terminated, and return the return code of the command.

    cmd: command given as list of strings
    cwd: current working directory
    inputs: in bytes!!!
    return_code: return code of command, an integer
    """

    cmd_string = ' '.join(cmd)
    print("Executing: %s", cmd_string)
    if inputs is not None:
        print("With input: %s", inputs)

    out_fn = fn+'stdout.txt'
    err_fn = fn+'stdin.txt'

    if cwd: 
        out_fn = os.path.join(cwd, out_fn)
        err_fn = os.path.join(cwd, err_fn)

    #initialize return code as None object
    return_code = None

    with open(out_fn, 'wb') as fout, open(err_fn, 'wb') as ferr:
        exe = subprocess.Popen(
                cmd,
                stdin = subprocess.PIPE,
                stdout = fout,
                stderr = ferr,
                shell = False,
                cwd = cwd
        )

        exe.communicate(input=inputs)
        
        # I think this is the command that forces to wait till execution.
        return_code = exe.returncode
    
    if return_code != 0:
        print('Execution failed')
        print('Attempted command: %s', cmd_string)
        print('Execution directory: %s', cwd)
        if inputs is not None:
            print('Input to the command was: %s', inputs)
        print('Return code from the command was: %i', return_code)
        print('Output from command can be found in: %s', out_fn)
        print('Errors from command can be found in: %s', err_fn)
        error_message = ('Command failed:\n {}. \nReturn code: {}').format(cmd_string,return_code)
        raise RuntimeError(error_message)

    if return_code is not None and return_code == 0:
        os.remove(out_fn)
        os.remove(err_fn)

    return return_code

def collect_trajectories_from_trajdirs(shoot_dirs):
    trajectories = []
    for trajdir in shoot_dirs:
        temptraj = glob.glob(trajdir+"traj/*.trr")
        trajectories += temptraj
    return trajectories

def append_to_txt(source_file, temp_file):
    with open(source_file, "a+") as f:
        f.write("-"*20)
        f.write("Appending from "+temp_file)
        f.write("-"*20)
        with open(temp_file, "r") as g:
            for line in g:
                f.write(g)
        g.close()
    f.close()

def read_from_cleanfn(cleanfn):
    assert os.path.isfile(cleanfn), "We cannot find cleanfn file at: "+cleanfn
    with open(cleanfn, 'r') as f:
        temp = f.readline().strip()
        try: 
            n_clean = int(temp)
        except ValueError:
            print("We did not find an integer in n_clean_fn ("+str(cleanfn)+"), but: "+temp)
    f.close()
    return n_clean

def write_to_cleanfn(cleanfn, n_clean):
    assert os.path.isfile(cleanfn), "We cannot find cleanfn file at: "+cleanfn
    with open(cleanfn, 'w+') as f:
        f.write(str(n_clean))
    f.close()

def write_trajdirs_to_txt(list_trajdirs, fn):
    with open(fn, "w+") as f:
        for path in list_trajdirs:
            f.write(str(path)+"\n")

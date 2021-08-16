import os
import glob
from .reading import *

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

def list_nonshoot_trajectories(indir,pe_fn="pathensemble.txt"):
    """
        indir is an ensemble folder (like .../000/)
        Returns a list of directories containing the non-shoot trajectories of this ensemble.
        
        pe_fn: filename of the pathensemble.txt file (I don't know, this may change)
        indir: directory where pe_fn is located
    """
    
    ACCFLAGS, REJFLAGS = set_flags_ACC_REJ()

    pe = read_pathensemble(indir+"/"+pe_fn)
    toremove_list = []

    assert len(pe.cyclenumbers) == len(pe.generation)
    for flag, cycnum, gen in zip(pe.flags,pe.cyclenumbers,pe.generation):
        if flag in ACCFLAGS:
            flagstr = "traj-acc"
        elif flag in REJFLAGS:
            flagstr = "traj-rej"
        else:
            raise NotImplementedError("Paths are either rejected or accepted. Path "+str(cycnum)+" of ensemble "+str(indir)+" does not even belong to the rejected, but to the flag: "+str(flag))

        if gen != 'sh':
            toremove_list.append(indir+"/traj/"+flagstr+"/"+str(cycnum)+"/")

    return toremove_list


def remove_nonshoot_trajectories(dirlist,check_type=True,filetypes=[".xtc",".trr",".gro"],donotremove=True,stop_on_error=True):
    """
        Removes the nonshoot trajectories in the folders given by dirlist
    """
    with open ("removedtrajs.txt", "w+") as ff:
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
        return shootcheck, moveline

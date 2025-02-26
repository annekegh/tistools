import tempfile
import os
import glob
from .reading import *
import subprocess

def cleaner(fn_i, fn_o, overwrite=False):
    """
    Remove everything until the last load (ld) path, and remove the last ld path too.

    Parameters
    ----------
    fn_i : str
        Filename input.
    fn_o : str
        Filename output.
    overwrite : bool, optional
        If True, allows overwriting the input file.

    Notes
    -----
    If overwrite is True, fn_o will be a temporary file that will replace fn_i.
    """
    if fn_o == fn_i:
        assert overwrite
    if overwrite:
        assert fn_i == fn_o
        fn_o = fn_i + ".TEMPORARYFILE.TMP"  # name for temporary file

    if os.path.exists(fn_o):
        assert overwrite

    with open(fn_i, 'r') as fi:
        lines = fi.readlines()

    with open(fn_o, 'w') as fo:
        printer = False
        for line in lines:
            if 'ld' in line:
                printer = False
                fo.close()
                fo = open(fn_o, 'w')
            elif 'ACC' in line and 'sh' in line:
                printer = True
            if printer:
                fo.write(line)

    if overwrite:
        os.remove(fn_i)
        os.rename(fn_o, fn_i)
        print(f"overwrite file... {fn_i} (was temporarily {fn_o})")
    else:
        print(f"file written... {fn_o}")

def sieve_trajectories(folders, n_clean, max_cycnum):
    """
    Sieve trajectories from multiple folders.

    Parameters
    ----------
    folders : list of str
        List of folder paths.
    n_clean : int
        Minimum cycle number to clean.
    max_cycnum : int
        Maximum cycle number to keep.

    Returns
    -------
    tuple
        Lists of directories to remove, to keep, and all directories.
    """
    rmdirs = []
    shdirs = []
    for fol in folders:
        toremove_list, tokeep_list = sieve_trajectories_from_folder(fol, n_clean=n_clean, max_cycnum=max_cycnum)
        rmdirs += toremove_list
        shdirs += tokeep_list

    alldirs = rmdirs + shdirs
    return rmdirs, shdirs, alldirs

def sieve_trajectories_from_folder(indir, pe_fn="pathensemble.txt", n_clean=0, max_cycnum=999999999):
    """
    Sieve trajectories from a single folder.

    Parameters
    ----------
    indir : str
        Directory where pe_fn is located.
    pe_fn : str, optional
        Filename of the pathensemble.txt file.
    n_clean : int, optional
        Minimum cycle number to clean.
    max_cycnum : int, optional
        Maximum cycle number to keep.

    Returns
    -------
    tuple
        Lists of directories to remove and to keep.
    """
    ACCFLAGS, REJFLAGS = set_flags_ACC_REJ()
    pe = read_pathensemble(os.path.join(indir, pe_fn))
    toremove_list = []
    tokeep_list = []

    assert len(pe.cyclenumbers) == len(pe.generation)
    for flag, cycnum, gen in zip(pe.flags, pe.cyclenumbers, pe.generation):
        if flag in ACCFLAGS:
            flagstr = "traj-acc"
        elif flag in REJFLAGS:
            flagstr = "traj-rej"
        else:
            raise NotImplementedError(f"Path {cycnum} of ensemble {indir} does not belong to the accepted or rejected flags.")
        
        dirstring = os.path.join(indir, "traj", flagstr, str(cycnum))
        if n_clean <= cycnum < max_cycnum:
            if gen != 'sh':
                toremove_list.append(dirstring)
            else:
                tokeep_list.append(dirstring)

    return toremove_list, tokeep_list

def remove_nonshoot_trajectories(dirlist, outputfolder, check_type=True, filetypes=[".xtc", ".trr", ".gro"], donotremove=True, stop_on_error=True):
    """
    Removes the non-shoot trajectories in the folders given by dirlist.

    Parameters
    ----------
    dirlist : list of str
        List of directories to process.
    outputfolder : str
        Folder to save the log of removed trajectories.
    check_type : bool, optional
        If True, checks the file type before removing.
    filetypes : list of str, optional
        List of allowed file types.
    donotremove : bool, optional
        If True, only logs the files that would be removed.
    stop_on_error : bool, optional
        If True, stops on encountering an error.
    """
    with open(os.path.join(outputfolder, "removedtrajs.txt"), "a+") as ff:
        for trajdir in dirlist:
            if os.path.isdir(trajdir):
                is_shoot, moveline = check_if_ordertxt_is_shoot(trajdir)
                if not is_shoot:
                    filelist = glob.glob(os.path.join(trajdir, "traj", "*"))
                    if filelist:
                        for f in filelist:
                            if check_type:
                                _, fext = os.path.splitext(f)
                                assert fext in filetypes, f"Not a trajectory: {f}"
                            if donotremove:
                                ff.write(f"Would have removed: {f}\n\tMove line: {moveline}\n")
                            else:
                                ff.write(f"Removed: {f}\n\tMove line: {moveline}\n")
                                os.remove(f)
                    else:
                        if donotremove:
                            ff.write(f"Would have removed: {f}\n\tMove line: {moveline}\n")
                        else:
                            ff.write(f"Removed: {f}\n\tMove line: {moveline}\n")
                            os.remove(f)
                else:
                    ff.write("This is a shooting move!\n")
                    ff.write(f"{moveline}\n")
                    if stop_on_error:
                        raise AssertionError("You tried to delete a shooting move trajectory, stopped deleting. Check removedtrajs.txt")
            else:
                ff.write(f"{trajdir} does not exist\n")
                if stop_on_error:
                    raise AssertionError("You wanted to delete a non-existing file, stopped deleting files. Check removedtrajs.txt")

def check_if_ordertxt_is_shoot(trajdir):
    """
    Check if the order.txt file in the given directory indicates a shooting move.

    Parameters
    ----------
    trajdir : str
        Directory containing the order.txt file.

    Returns
    -------
    tuple
        Boolean indicating if it's a shooting move and the first line of the order.txt file.
    """
    with open(os.path.join(trajdir, "order.txt"), "r") as f:
        moveline = f.readline()
        shootstr = "move: ('sh',"
        shootcheck = shootstr in moveline
    return shootcheck, moveline

def subsample_trajectories(trajectories, dt, ndx_fn, tpr_fn, procs=1):
    """
    Spawns processes that subsample the listed trajectories.

    Parameters
    ----------
    trajectories : list of str
        List of paths to the trajectory files.
    dt : float
        Sampling period.
    ndx_fn : str
        Path to the index.ndx file.
    tpr_fn : str
        Path to the topol.tpr file.
    procs : int, optional
        Number of processes to run in parallel.

    Returns
    -------
    bool
        True if all return codes are zero, False otherwise.
    """
    traj_sizes = get_traj_sizes(trajectories)
    print(f"Number of processors for subsampling: {procs}")
    chunk_list = weighted_split_list(trajectories, procs, traj_sizes) if procs > 1 else [trajectories]

    log_file = "cleanfolder/log_subsampling.txt"
    proc_list = []
    out_list = []
    err_list = []
    cid_list = []
    fn_list = []

    for i, chunk_trr in enumerate(chunk_list):
        fn = write_chunk_trr(chunk_trr, str(i))
        f_out = tempfile.TemporaryFile()
        f_err = tempfile.TemporaryFile()
        cmd = ["./subsampler_chunk", '-f', fn, '-n', ndx_fn, '-s', tpr_fn, '-t', dt, '-c', str(i)]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=f_out, stderr=f_err, shell=False)
        proc_list.append(proc)
        out_list.append(f_out)
        err_list.append(f_err)
        cid_list.append(str(i))
        fn_list.append(fn)

    with open(log_file, "wb+") as f_log:
        for proc, f_out, f_err, cid in zip(proc_list, out_list, err_list, cid_list):
            subline = f"--SUBPROCESS {cid} --\n".encode('ascii')
            outline = f"--OUTPUT {cid} --\n".encode('ascii')
            errline = f"--ERROR {cid} --\n".encode('ascii')
            intline = ("-" * 15 + "\n").encode('ascii')
            proc.wait()
            f_log.write(intline)
            f_log.write(subline)
            f_log.write(intline)
            f_log.write(outline)
            f_log.write(intline)
            f_out.seek(0)
            f_log.write(f_out.read())
            f_out.close()
            f_log.write(intline)
            f_log.write(errline)
            f_log.write(intline)
            f_err.seek(0)
            f_log.write(f_err.read())
            f_err.close()
            f_log.write(intline)
            f_log.close()

    all_went_well = True
    for i, proc in enumerate(proc_list):
        return_code = proc.returncode
        if return_code is not None and return_code == 0:
            print(f"Subsampling chunk {i} completed successfully.")
        else:
            all_went_well = False
            raise RuntimeError("One or more subsampling processes failed. Check the log files in 'cleanfolder':\nstdout: {i}_stdout.txt\nstderr: {i}_stderr.txt for details.")

    if all_went_well:
        for fn in fn_list:
            os.remove(fn)
    return all_went_well

def write_chunk_trr(trr_files, chunk_id):
    """
    Write the list of trajectory files to a text file.

    Parameters
    ----------
    trr_files : list of str
        List of trajectory file paths.
    chunk_id : str
        Identifier for the chunk.

    Returns
    -------
    str
        Filename of the written text file.
    """
    fn = f"chunk_{chunk_id}_trr_paths.txt"
    with open(fn, "w+") as f:
        for el in trr_files:
            f.write(el + "\n")
    return fn

def read_chunk_trr(fn):
    """
    Read the list of trajectory files from a text file.

    Parameters
    ----------
    fn : str
        Filename of the text file.

    Returns
    -------
    list of str
        List of trajectory file paths.
    """
    trr_files = []
    with open(fn, 'r') as f:
        for line in f:
            trr_files.append(line.strip())
    return trr_files

def split_list(l, n):
    """
    Split list l into n chunks of approximately equal size.

    Parameters
    ----------
    l : list
        List to be split.
    n : int
        Number of chunks.

    Returns
    -------
    list of lists
        List containing n chunks.
    """
    assert n > 1, "Please don't call the split function to split into 1"
    k, m = divmod(len(l), n)
    return [l[i*k + min(i, m):(i+1)*k + min(i+1, m)] for i in range(n)]

def weighted_split_list(l, n, s, verbose=False):
    """
    Split list l into n chunks with similar total size.

    Parameters
    ----------
    l : list
        List to be split.
    n : int
        Number of chunks.
    s : list of int
        Sizes of each list element.
    verbose : bool, optional
        If True, prints the chunk load and imbalance.

    Returns
    -------
    list of lists
        List containing n chunks.
    """
    assert n > 1, "Please don't call the split function to split into 1"
    assert len(l) == len(s), f"List and weight-list have different lengths: {len(l)}; {len(s)}"

    chunks = [[] for _ in range(n)]
    chunks_w = [0 for _ in range(n)]
    for t, tw in zip(l, s):
        idx = chunks_w.index(min(chunks_w))
        chunks[idx].append(t)
        chunks_w[idx] += tw

    max_weight = max(chunks_w)
    min_weight = min(chunks_w)

    if verbose:
        print(f"Chunk load, in units min_chunk_weight [{min_weight} bytes]")
        for w in chunks_w:
            print(f"{round(w/min_weight, 2)}\t")
        print(f"\nImbalance max_chunk_weight/min_chunk_weight: {round(max_weight/min_weight, 2)}")

    return chunks

def get_max_cycnum_ensemble(indir, pe_fn="pathensemble.txt"):
    """
    Get the maximum cycle number in the ensemble.

    Parameters
    ----------
    indir : str
        Directory where pe_fn is located.
    pe_fn : str, optional
        Filename of the pathensemble.txt file.

    Returns
    -------
    int
        Maximum cycle number.
    """
    pe = read_pathensemble(os.path.join(indir, pe_fn))
    return max(pe.cyclenumbers)

def get_max_cycnum_simul(folders):
    """
    Get the maximum cycle number across multiple folders.

    Parameters
    ----------
    folders : list of str
        List of folder paths.

    Returns
    -------
    int
        Maximum cycle number.
    """
    max_cycnum = 0
    for fol in folders:
        trial_cycnum = get_max_cycnum_ensemble(fol)
        if trial_cycnum > max_cycnum:
            max_cycnum = trial_cycnum
    return max_cycnum

def list_trajectory_folders_lt_maxnum(indir, max_cycnum, pe_fn="pathensemble.txt"):
    """
    List trajectory folders with cycle numbers less than max_cycnum.

    Parameters
    ----------
    indir : str
        Directory where pe_fn is located.
    max_cycnum : int
        Maximum cycle number.
    pe_fn : str, optional
        Filename of the pathensemble.txt file.

    Returns
    -------
    list of str
        List of directories containing the non-shoot trajectories.
    """
    ACCFLAGS, REJFLAGS = set_flags_ACC_REJ()
    pe = read_pathensemble(os.path.join(indir, pe_fn))
    move_list = []

    assert len(pe.cyclenumbers) == len(pe.generation)
    for flag, cycnum, gen in zip(pe.flags, pe.cyclenumbers, pe.generation):
        if flag in ACCFLAGS:
            flagstr = "traj-acc"
        elif flag in REJFLAGS:
            flagstr = "traj-rej"
        else:
            raise NotImplementedError(f"Path {cycnum} of ensemble {indir} does not belong to the accepted or rejected flags.")
        
        if cycnum < max_cycnum:
            dirstring = os.path.join(indir, "traj", flagstr, str(cycnum))
            move_list.append(dirstring)

    return move_list

def move_trajectories(traj_dirs, target_dir, verbose=False):
    """
    Move trajectories to the target directory.

    Parameters
    ----------
    traj_dirs : list of str
        List of directories to move.
    target_dir : str
        Target directory.
    verbose : bool, optional
        If True, prints the source and destination of moved directories.
    """
    for movedir in traj_dirs:
        temp_target = os.path.join(target_dir, movedir)
        os.makedirs(temp_target, exist_ok=True)
        os.system(f'mv {movedir}/* {temp_target}')
        if verbose:
            print(f"Moved from: {movedir}\nTo: {temp_target}")

def execute_command(cmd, cwd=None, inputs=None, fn=''):
    """
    Execute a command and wait until it is terminated.

    Parameters
    ----------
    cmd : list of str
        Command given as list of strings.
    cwd : str, optional
        Current working directory.
    inputs : bytes, optional
        Input to the command.
    fn : str, optional
        Filename for logging.

    Returns
    -------
    int
        Return code of the command.
    """
    cmd_string = ' '.join(cmd)
    print(f"Executing command: {cmd_string}")
    if inputs is not None:
        print(f"With input: {inputs.decode('ascii')}")
    
    out_fn = f'{fn}stdout.txt'
    err_fn = f'{fn}stderr.txt'
    if cwd:
        out_fn = os.path.join(cwd, out_fn)
        err_fn = os.path.join(cwd, err_fn)

    return_code = None
    with open(out_fn, 'wb') as fout, open(err_fn, 'wb') as ferr:
        exe = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=fout, stderr=ferr, shell=False, cwd=cwd)
        exe.communicate(input=inputs)
        return_code = exe.returncode

    if return_code != 0:
        print(f"Execution failed. Command: {cmd_string}, Directory: {cwd}, Input: {inputs}, Return code: {return_code}")
        print(f"Output: {out_fn}, Errors: {err_fn}")
        raise RuntimeError(f"Command failed: {cmd_string}. Return code: {return_code}")
    
    if return_code == 0:
        print(f"Command '{cmd_string}' executed successfully.")
        os.remove(out_fn)
        os.remove(err_fn)
    
    return return_code

def collect_trajectories_from_trajdirs(shoot_dirs):
    """
    Collect trajectory files from the given directories.

    Parameters
    ----------
    shoot_dirs : list of str
        List of directories containing trajectories.

    Returns
    -------
    list of str
        List of trajectory file paths.
    """
    trajectories = []
    for trajdir in shoot_dirs:
        temptraj = glob.glob(os.path.join(trajdir, "traj", "*.trr"))
        trajectories += temptraj
    return trajectories

def append_to_txt(temp_file, source_file):
    """
    Append the contents of temp_file to source_file.

    Parameters
    ----------
    temp_file : str
        Path to the temporary file.
    source_file : str
        Path to the source file.
    """
    with open(source_file, "a+") as f:
        f.write("-" * 20)
        f.write(f"Appending from {temp_file}")
        f.write("-" * 20)
        f.write("\n")
        with open(temp_file, "r") as g:
            for line in g:
                f.write(line)

def read_from_cleanfn(cleanfn):
    """
    Read the clean file number from the given file.

    Parameters
    ----------
    cleanfn : str
        Path to the clean file.

    Returns
    -------
    int
        Clean file number.

    Raises
    ------
    FileNotFoundError
        If the clean file does not exist.
    ValueError
        If the content of the clean file is not an integer.
    """
    if not os.path.isfile(cleanfn):
        raise FileNotFoundError(f"We cannot find cleanfn file at: {cleanfn}")
    
    with open(cleanfn, 'r') as f:
        temp = f.readline().strip()
        try:
            n_clean = int(temp)
        except ValueError:
            raise ValueError(f"We did not find an integer in cleanfn file ({cleanfn}), but found: {temp}")
    
    return n_clean

def write_to_cleanfn(cleanfn, n_clean):
    """
    Write the clean file number to the given file.

    Parameters
    ----------
    cleanfn : str
        Path to the clean file.
    n_clean : int
        Clean file number to write.

    Raises
    ------
    FileNotFoundError
        If the clean file does not exist.
    """
    if not os.path.isfile(cleanfn):
        raise FileNotFoundError(f"We cannot find cleanfn file at: {cleanfn}")
    
    with open(cleanfn, 'w+') as f:
        f.write(str(n_clean))
    print(f"Clean file number {n_clean} written to {cleanfn}.")

def write_trajdirs_to_txt(list_trajdirs, fn):
    """
    Write the list of trajectory directories to a text file.

    Parameters
    ----------
    list_trajdirs : list of str
        List of trajectory directories.
    fn : str
        Filename of the text file.
    """
    with open(fn, "w+") as f:
        for path in list_trajdirs:
            f.write(str(path) + "\n")
    print(f"Trajectory directories written to {fn}.")

def get_traj_sizes(trajectories):
    """
    Get the sizes of the trajectory files.

    Parameters
    ----------
    trajectories : list of str
        List of trajectory file paths.

    Returns
    -------
    list of int
        List of sizes (in bytes) of the trajectory files.
    """
    size_list = [os.path.getsize(traj) for traj in trajectories]
    return size_list

def filter_gmx_trajectory(traj, group_idx, index_file, topol_file, out_ftype="xtc", delete=True):
    """
    Filter the trajectory to only save the atoms in the specified group.

    Parameters
    ----------
    traj : str
        Path to the trajectory file.
    group_idx : int
        Index of the group to save.
    index_file : str
        Path to the index file.
    topol_file : str
        Path to the topology file.
    out_ftype : str, optional
        Output file type (default is "xtc").
    delete : bool, optional
        If True, deletes the original trajectory file after filtering.

    Raises
    ------
    RuntimeError
        If the filtering process fails.
    """
    outfn = traj[:-4] + "." + out_ftype
    cmd = ["gmx", "trjconv", "-f", traj, "-s", topol_file, "-o", outfn]
    if index_file is not None:
        cmd += ["-n", index_file]
    
    print(f"Executing command: {' '.join(cmd)}")
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, shell=False)
    input_str = f"{group_idx}\n"
    input_bytes = input_str.encode()
    p.communicate(input_bytes)
    p.wait()

    if p.returncode != 0:
        raise RuntimeError(f"Filtering trajectory {traj} failed with return code {p.returncode}.")
    
    if delete:
        os.remove(traj)
        print(f"Original trajectory {traj} deleted after filtering.")
    else:
        print(f"Filtered trajectory saved as {outfn}.")
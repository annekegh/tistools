# MOVE TO DEV
#!/usr/bin/env python3

print("uh-oh")
import tempfile
#print("imported tempfile")
import os
#print("importoed os")
import glob
#print("imported glob")
import subprocess
#print("imported subprocess")


def execute_command(cmd, cwd=None, inputs=None, fn=''):
    """
    Command is executed, after which we wait until it is terminated, and return the return code of the command.

    cmd: command given as list of strings
    cwd: current working directory
    inputs: in bytes!!!
    return_code: return code of command, an integer
    """

    cmd_string = ' '.join(cmd)
    print("Executing:\n"+cmd_string)
    if inputs is not None:
        print(("With input: ").encode('ascii')+inputs)

    out_fn = fn+'stdout.txt'
    err_fn = fn+'stderr.txt'

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
        print('Command '+str(cmd_string)+' went perfectly fine.')
        os.remove(out_fn)
        os.remove(err_fn)

    return return_code

def list_all_shoot_dirs(logfile):
    # Logfile: the global logger of tistools-clean-trajectories
    # Returns: shdirs, list with all shoottraj dirs (end with /)
    shdirs = []
    with open(logfile,'r') as f:
        for line in f:
            if line[0] != '-':
                shdirs.append(line.strip())
    return shdirs

def collect_trajectories_from_dirs(dir_list, ext="trr"):
    # dir_list must be a list!
    assert type(dir_list) == list, "dir_list must be a list"
    trajectories = []
    for trajdir in dir_list:
        temptraj = glob.glob(trajdir+"traj/*."+ext)
        # += will add the elements of temptraj to trajectories,
        # while append will add temptraj as a whole. As edge case,
        # empty lists will thus not add anything with +=, while
        # append will add an empty list.
        trajectories += temptraj
    return trajectories

def get_traj_sizes(trajectories):
    """ Returns list of sizes (in bytes) of the trajectories """
    size_list = []
    for traj in trajectories:
        size_list.append(os.path.getsize(traj))
    return size_list

def weighted_split_list(l,n,s,verbose=False):
    """
    split list l with L elements into n chunks, where each chunk
    has similar total size (bytes). The size of each list element
    is given by s
    """
    assert n > 1, "Please don't cal lthe split function to split into 1"
    assert len(l) == len(s), "list and weight-list have differnt len: "+str(len(l))+";"+str(len(s))

    #chunks = [[]]*n        # In this notation, all list elements are linked!!!!!!!
    #chunks_w = [0]*n

    chunks = [[] for i in range(n)]
    chunks_w = [0 for i in range(n)]

    for t, tw in zip(l,s):
        idx = chunks_w.index(min(chunks_w)) # This will give the 'first' min_idx when multiple min exist
        chunks[idx].append(t)
        chunks_w[idx] += tw

    max_weight = max(chunks_w)
    min_weight = min(chunks_w)

    if verbose:
        print("Chunk load, in units min_chunk_weight ["+str(min_weight)+" bytes]\n")
        for w in chunks_w:
            print(str(round(w/min_weight,2))+"\t")
        print("\n")

    print("Imbalance max_chunk_weight/min_chunk_weight: "+str(round(max_weight/min_weight,2)))

    return chunks

def write_chunk_paths(trr_files, chunk_id, folder=None):
    fn = "chunk_"+chunk_id+"_paths.txt"
    if folder is not None:
        fn = folder+"/"+fn
    with open(fn, "w+") as f:
        for el in trr_files:
            f.write(el+"\n")
    f.close()
    return fn

def read_chunk_paths(fn):
    trr_files = []
    with open(fn, 'r') as f:
        for line in f:
            trr_files.append(line.strip())
    f.close()
    return trr_files



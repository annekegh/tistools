"""function for dealing with pyretis output"""

import numpy as np

def set_flags_ACC_REJ():
    # Hard-coded rejection flags found in output files
    REJFLAGS = ['FTL', 'NCR', 'BWI', 'BTL', 'BTX', 'FTX',]
    REJFLAGS += ['BTS','KOB','FTS','EWI','SWI']
    REJFLAGS += ['MCR']
    REJFLAGS += ['TSS','TSA']
    """
         MCR': 'Momenta change rejection',
        'BWI': 'Backward trajectory end at wrong interface',
        'BTL': 'Backward trajectory too long (detailed balance condition)',
        'BTX': 'Backward trajectory too long (max-path exceeded)',
        'BTS': 'Backward trajectory too short',
        'KOB': 'Kicked outside of boundaries',
        'FTL': 'Forward trajectory too long (detailed balance condition)',
        'FTX': 'Forward trajectory too long (max-path exceeded)',
        'FTS': 'Forward trajectory too short',
        'NCR': 'No crossing with middle interface',
        'EWI': 'Initial path ends at wrong interface',
        'SWI': 'Initial path starts at wrong interface'
        'TSS' = No valid indices to select for swapping
        'TSA'= Rejection due to the target swap acceptance criterium.
    """

    ACCFLAGS = ['ACC',]
    # option 1: do not change, as if omitted
    # option 2: accept
    #ACCFLAGS += ['0-L']
    # option 3: reject
    REJFLAGS += ['0-L']

    # check not in both
    tmpflags = ACCFLAGS+REJFLAGS
    for flag in tmpflags:
        if flag in ACCFLAGS:
            assert flag not in REJFLAGS
        if flag in REJFLAGS:
            assert flag not in ACCFLAGS
    # check in at least one
    assert '0-L' in tmpflags

    return ACCFLAGS, REJFLAGS

class PathEnsemble(object):
    def __init__(self,data):
        # data is a list of lines.split() in the pathensemble.txt file
        self.cyclenumbers   = np.array([int(dat[0]) for dat in data])
        self.pathnumbers    = np.array([int(dat[1]) for dat in data])
        self.newpathnumbers = np.array([int(dat[2]) for dat in data])
        self.lmrs       = np.array(["".join(dat[3:6]) for dat in data])
        self.lengths    = np.array([int(dat[6]) for dat in data])
        self.flags      = np.array([dat[7] for dat in data])
        self.generation = np.array([dat[8] for dat in data])

        self.ncycle     = len(self.lengths)
        self.totaltime  = np.sum(self.lengths)
        # consistency is automatic

class OrderParameter(object):

    def __init__(self,cyclenumbers,lengths,flags,generation,longtraj,data):
        # data is an np.array of order parameters

        assert_consistent(cyclenumbers,lengths,flags,generation,longtraj,data)

        self.cyclenumbers = np.array(cyclenumbers)
        self.lengths    = np.array(lengths)
        self.flags      = np.array(flags)
        self.generation = np.array(generation)
        self.longtraj   = np.array(longtraj)
        self.ops        = data
        self.ncycle     = len(self.lengths)
        self.totaltime  = np.sum(self.lengths)

    # This could work on OrderParameter object
    #def select_traj(self,i):
    #    if i==0: previous_length=0
    #    else:
    #        previous_length = np.sum(self.lengths[:i])
    #    thislength = self.lengths[i]
    #    traj = self.ops[previous_length:previous_length+thislength,:]
    #    return traj

def assert_consistent(cyclenumbers,lengths,flags,generation,longtraj,data):
        n = len(cyclenumbers)
        assert n == len(lengths)
        assert n == len(flags)
        assert n == len(generation)
        assert np.sum(lengths) == len(data)
        assert np.sum(lengths) == len(longtraj)

def read_pathensemble(fn,ostart=0):
    """read pathensemble.txt file

    Format pathensemble.txt:

         0          1          0 L M L     383 ACC ld  9.925911427e-01  1.490895033e+00       0     191  0.000000000e+00       0       0  1.000000000e+00
         1          2          1 L M L     250 ACC sh  9.794998169e-01  1.581833124e+00       0     137  1.284500122e+00     243     174  1.000000000e+00

    """
    data = []
    with open(fn,"r+") as f:
        lines = f.readlines()
        for line in lines:
            words = line.split()
            cycle = int(words[0])
            if cycle >= ostart:
                data.append(words)
            # get start-middle-end (combine chars to a string): "L M R"
            #     "".join(data[3:6])
            # get length:
            #      int(data[6])
            # get status/flag/acceptance:  "ACC,"
            #      data[7]
            # get generation:
            #      data[8]

    pe = PathEnsemble(data)
    return pe


def read_order(fn,ostart=0):
    """read file order.txt

    Format order.txt:

# Cycle: 0, status: ACC, move: ('ld', 0, 0, 0)
#     Time       Orderp
         0     0.992591     0.751421     0.122551     1.440613     0.235256     0.625671
         1     1.002283     0.737442     0.095843     1.420257     0.252691     0.565861
    ...

    """

    #data_op = np.loadtxt(fn)

    # now read line per line
    cyclenumbers = []
    lengths = []
    flags = []
    generation = []

    longtraj = []
    data = []

    ntraj = 0
    ntraj_started = 0
    last_length = 0

    # check first line
    with open(fn,"r+") as f:
        line = f.readline()
        if not line.startswith("# Cycle:"):
            print("First line of orderparameter file %s  didn't start with cycle"%fn)
            raise ValueError("something wrong")

    with open(fn,'r') as f:        # read this file
        for i, line in enumerate(f):
            #if i > ostart:   # NOOOOOO  # TODO

            # header time
            if line.startswith("#     Time"):
                pass

            # header Cycle
            elif line.startswith("# Cycle:"):
                if ntraj_started > 0:  # not the very first traj
                    if last_length > 0:
                        # successfully update the previous one
                        ntraj += 1
                        lengths[-1] = last_length

                    elif last_length == 0:
                        print("WARNING"*30)
                        print("encountered traj with length 0")
                        print("at cyclenumber",line)
                        # undo a few things
                        cyclenumbers = cyclenumbers[:-1]
                        flags = flags[:-1]
                        generation = generation[:-1]
                        ntraj_started -= 1  # previous was fake alarm

                # and reset for new one
                ntraj_started += 1
                last_length = 0

                # extract the time, cyclenumber, flag, generation
                words = line.split()
                cyclenumbers.append(int(words[2][:-1]))  # remove the last character, which is a comma
                flags.append(words[4][:-1])         # remove the last character, which is a comma: ACC,
                generation.append(words[6][2:4])    # remove characters: ('sh',
                # length to be updated!!!
                lengths.append(0)

            # Collect order parameter of traj
            else:
                words = line.split()

                # words[0] = the time step of the traj, counting starts at 0
                assert int(words[0])==last_length
                last_length += 1    # update traj length

                # the order parameters
                # assume just 1 op: words[1]
                longtraj.append(float(words[1]))
                # collect all order parameters: words[1:]
                if len(words[1:])>1:
                    data.append([float(word) for word in words[1:]])  # skip the time
                else:
                    data.append(float(words[1]))  # this is a stupied copy of longtraj

    # finish the last trajectory when done with reading

    # if order parameter lines were the last lines of order.txt
    # then I need to update
    if last_length > 0:
        ntraj += 1
        lengths[-1] = last_length

    # if "# Cycle:" was the last line of order.txt
    elif ntraj_started == ntraj+1:
        # undo a few things
        cyclenumbers = cyclenumbers[:-1]
        flags = flags[:-1]
        generation = generation[:-1]
        lengths = lengths[:-1]

    data = np.array(data)
    if len(data.shape)==1:
        data = data.reshape((len(data),1))

    # verify lengths:
    assert_consistent(cyclenumbers,lengths,flags,generation,longtraj,data)
    op = OrderParameter(cyclenumbers,lengths,flags,generation,longtraj,data)
    return op


def get_weights(flags,ACCFLAGS,REJFLAGS):
    """
    Returns:
      weights -- array with weight of each trajectory, 0 if not accepted
      ncycle_true -- sum of weights
    """

    ntraj = len(flags)
    weights = np.zeros(ntraj,int)

    accepted = 0
    rejected = 0
    omitted = 0

    acc_w = 0
    acc_index = 0
    tot_w = 0
    assert flags[0] == 'ACC'
    for i,flag in enumerate(flags):
        if flag in ACCFLAGS:
            # store previous traj with accumulated weight
            weights[acc_index] = acc_w
            tot_w += acc_w
            # info for new traj
            acc_index = i
            acc_w = 1
            accepted += 1
        elif flag in REJFLAGS:
            acc_w += 1    # weight of previous accepted traj increased
            rejected += 1
        else:
            omitted += 1
    #if flag[-1] in REJFLAGS:
        # I did not store yet the weight of the previous accepted path
        # because I do not have the next accepted path yet
        # so neglect this path, I guess.
    # at the end: store the last accepted path with its weight
    weights[acc_index] = acc_w
    tot_w += acc_w

    print("weights:")
    print("accepted     ",accepted)
    print("rejected     ",rejected)
    print("omitted      ",omitted)
    print("total trajs  ",ntraj)
    print("total weights",np.sum(weights))

    assert omitted == 0
    ncycle_true = np.sum(weights)
    miss = len(flags)-1 - ncycle_true
    for i in range(miss):
        assert flags[-(i+1)] in REJFLAGS
        # the reason why this could happen

    return weights, ncycle_true

#########################################

def get_data_ensemble_consistent(folder,):
    """
    folder -- can be somedir/000, somedir/001, somedir/002, etc
    """

    fn_op = "%s/order.txt"%folder
    fn_path = "%s/pathensemble.txt"%folder

    # READ
    op = read_order(fn_op)
    pe = read_pathensemble(fn_path)

    print("Reading...")
    print("cycle_op  ",op.ncycle)
    print("cycle_path",pe.ncycle)
    print("total time op  ",op.totaltime)
    print("total time path",pe.totaltime)

    if op.totaltime < pe.totaltime:      # TODO
        print("fix lengths...")
        data_path = data_path[:-1]
        lengths = lengths[:-1]
        flags = flags[:-1]
        generation = generation[:-1]
        print("data_op  ",data_op.shape)
        print("data_path",len(data_path))
        print("total time op  ",len(data_op))
        print("total time path",np.sum(lengths))
    elif op.totaltime > pe.totaltime:
        print("oei")
    assert op.totaltime == pe.totaltime

    # matching
    assert op.ncycle == pe.ncycle
    match = [l1 == l2 for l1,l2 in zip(op.lengths,pe.lengths)]
    #print(match)
    try:
        match.index(False)
    except:
        ValueError()

    match = [l1 == l2 for l1,l2 in zip(op.flags,pe.flags)]
    try:
        match.index(False)
    except:
        ValueError()

    match = [l1 == l2 for l1,l2 in zip(op.generation,pe.generation)]
    try:
        match.index(False)
    except:
        ValueError()

    return op, pe


#########################################

# reading the RESTART file
# the pyretis.restart file is a pickle object

def read_restart_file(filename):
    """Read restart info for a simulation.

    Parameters
    ----------
    filename : string
        The file we are going to read from.
        filename = "pyretis.restart"

    """
    import pickle
    with open(filename, 'rb') as infile:
        info = pickle.load(infile)
    return info


#########################################
# Other functions


def select_traj(data_op,lengths,i):
    # data_op can be e.g. OrderParameter the ops attribute (.ops)
    if i==0: previous_length=0
    else:
        previous_length = np.sum(lengths[:i])
    thislength = lengths[i]
    #I made sure that len(data_op.shape)>1
    #I made sure that len(data_sel.shape)>1
    data_sel = np.zeros((thislength,data_op.shape[1]))
    data_sel[:,:] = data_op[previous_length:previous_length+thislength,:]
    return data_sel


def get_ntraj(accepted,acc=True):    # TODO needed?

    if not acc:
        # OPTION: do all
        ntraj = len(length)
        indices = np.arange(ntraj)
    else:
        # OTHER OPTION
        ntraj = np.sum(accepted)
        indices = [i for i in range(len(accepted)) if accepted[i]]
        #print(indices)
    return ntraj,indices


###########################################
def read_inputfile(filename):
    """read interfaces and timestep from inputfile

    Add the zero_left interface to the end of the list, if this
    is present."""

    # interfaces
    with open(filename,"r") as f:
        for line in f:
            if "interfaces" in line and "=" in line and not line.strip().startswith("#"):
                parts = line.split("[")
                parts2 = parts[1].split("]")
                words = parts2[0].split(",")
                interfaces = [float(word) for word in words]
                break
    #st = line.find("[",)  # beg=0, end=len(string))
    #end = line.find("]")
    #line = line[st+1,end]
    #words = line.split(",")

    zero_left = None
    with open(filename,"r") as f:
        for line in f:
            if "zero_left" in line and "=" in line and not line.strip().startswith("#"):
                parts = line.split("=")
                zero_left = float(parts[1])
                break

    # interfaces
    with open(filename,"r") as f:
        for line in f:
            if "timestep" in line and "=" in line and not line.strip().startswith("#"):
                parts = line.split("=")
                timestep = float(parts[1])
                break

    return interfaces,zero_left,timestep


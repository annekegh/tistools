"""function for dealing with pyretis output"""

import numpy as np

def set_flags_ACC_REJ():
    # Hard-coded rejection flags found in output files
    REJFLAGS = ['FTL', 'NCR', 'BWI', 'BTL', 'BTX', 'FTX',]
    REJFLAGS += ['BTS','KOB','FTS','EWI','SWI']
    REJFLAGS += ['MCR']
    REJFLAGS += ['TSS','TSA']
    REJFLAGS += ['HAS','CSA','NSG']
    REJFLAGS += ['SWD']
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
        'SWI': 'Initial path starts at wrong interface',
        'TSS': 'No valid indices to select for swapping',
        'TSA': 'Rejection due to the target swap acceptance criterium',
        'HAS': 'High acceptance swap rejection for SS/CS detailed balance',
        'CSA': 'Common Sense super detailed balance rejection',
        'NSG': 'Path has no suitable segments',
        'SWD': 'PPTIS swap with incompatible propagation direction'.
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
    def __init__(self,data=None):
        if data is not None:
            # data is a list of lines.split() in the pathensemble.txt file
            self.cyclenumbers   = np.array([int(dat[0]) for dat in data])
            self.pathnumbers    = np.array([int(dat[1]) for dat in data])
            self.newpathnumbers = np.array([int(dat[2]) for dat in data])
            self.lmrs       = np.array(["".join(dat[3:6]) for dat in data])
            self.lengths    = np.array([int(dat[6]) for dat in data])
            self.flags      = np.array([dat[7] for dat in data])
            self.generation = np.array([dat[8] for dat in data])
            self.lambmins   = np.array([float(dat[9]) for dat in data])
            self.lambmaxs   = np.array([float(dat[10]) for dat in data])

            self.ncycle     = len(self.lengths)
            self.totaltime  = np.sum(self.lengths)
            # consistency is automatic

            self.weights = []
            self.shootlinks = np.full_like(self.cyclenumbers,None,dtype=object)
            self.name = ""
            self.interfaces = [] # [ [L, M, R], string([L,M,R]) ] 2 lists in a list

            self.has_zero_minus_one = False 
            self.in_zero_minus = False 
            self.in_zero_plus = False

    def set_name(self, name):
        self.name = name

    def set_weights(self, weights):
        self.weights=weights

    def set_interfaces(self, interfaces):
        self.interfaces = interfaces

    def set_zero_minus_one(self, has_zero_minus_one):
        self.has_zero_minus_one = has_zero_minus_one

    def set_in_zero_minus(self, in_zero_minus):
        self.in_zero_minus = in_zero_minus

    def set_in_zero_plus(self, in_zero_plus):
        self.in_zero_plus = in_zero_plus

    def update_shootlink(self,cycnum,link):
        cycnumlist = (self.cyclenumbers).tolist()
        cyc_idx = cycnumlist.index(cycnum)
        self.shootlinks[cyc_idx] = link
    
    def get_shootlink(self, cycnum):
        cycnumlist = (self.cyclenumbers).tolist()
        cyc_idx = cycnumlist.index(cycnum)
        return self.shootlinks[cyc_idx]

    def save_pe(self, fn):
        import pickle 
        with open("pe_"+fn+".pkl", 'wb') as g:
            pickle.dump(self, g, pickle.HIGHEST_PROTOCOL)

    def unify_pe(self):
        """ Unify the pathensemble by replacing the zero weight
        paths with the previous non-zero weight path. This only
        works if high_acceptance = False (?). 
        
        """
        new_pe = PathEnsemble()
        new_pe.cyclenumbers = self.cyclenumbers
        new_pe.pathnumbers = self.pathnumbers
        new_pe.newpathnumbers = self.newpathnumbers
        new_pe.weights = np.ones_like(self.weights)
        new_pe.lmrs = np.repeat(self.lmrs, self.weights)
        new_pe.lengths = np.repeat(self.lengths, self.weights)
        new_pe.flags = np.repeat(self.flags, self.weights)
        new_pe.generation = np.repeat(self.generation, self.weights)
        new_pe.lambmins = np.repeat(self.lambmins, self.weights)
        new_pe.lambmaxs = np.repeat(self.lambmaxs, self.weights)
        new_pe.interfaces = self.interfaces
        new_pe.name = self.name
        new_pe.has_zero_minus_one = self.has_zero_minus_one
        new_pe.in_zero_minus = self.in_zero_minus
        new_pe.in_zero_plus = self.in_zero_plus

        print("Are all weights 1? ", np.all(new_pe.weights == 1))
        print("Are all paths accepted? ", np.all(new_pe.flags == "ACC"))

        return new_pe

    def sample_pe(self, cycle_ids):
        """ Only keep the cyclenumbers in cycle_ids. """
        new_pe = PathEnsemble()
        new_pe.cyclenumbers = self.cyclenumbers[cycle_ids]
        new_pe.pathnumbers = self.pathnumbers[cycle_ids]
        new_pe.newpathnumbers = self.newpathnumbers[cycle_ids]
        new_pe.lmrs = self.lmrs[cycle_ids]
        new_pe.lengths = self.lengths[cycle_ids]
        new_pe.flags = self.flags[cycle_ids]
        new_pe.generation = self.generation[cycle_ids]
        new_pe.weights = self.weights[cycle_ids]
        new_pe.lambmins = self.lambmins[cycle_ids]
        new_pe.lambmaxs = self.lambmaxs[cycle_ids]
        #new_pe.shootlinks = self.shootlinks[cycle_ids]
        new_pe.name = self.name
        new_pe.interfaces = self.interfaces
        new_pe.has_zero_minus_one = self.has_zero_minus_one
        new_pe.in_zero_minus = self.in_zero_minus
        new_pe.in_zero_plus = self.in_zero_plus
        return new_pe


    def bootstrap_pe(self, N, Bcycle, Acycle=0):
        """Bootstrap the path ensemble by sampling N elements within the cycle 
        range [Acycle, Bcycle].

        Note1: Weights should already be set for the original path ensemble
        Note2: Load paths should be removed from the path ensemble before 
               bootstrapping
        Note3: We bootstrap indices of the cycles, and then use the indices to
               bootstrap the path ensemble

        Parameters
        ----------
        N : int
            Number of paths to sample
        Bcycle : int
            Upper bound of the cycle range
        Acycle : int, optional
            Lower bound of the cycle range. The default is 0.
        """

        import random

        # 1. Get the indices of cycles within the range [Acycle, Bcycle]. 
        #    Because of restarts, pe.cyclenumbers may not be a continuous range.
        #    We only want to select accepted paths, with a probability given by
        #    the weights. Discard rejected paths, AND load paths. 
        idx, idx_w = [], []
        for i in range(len(self.cyclenumbers)):
            if self.cyclenumbers[i] >= Acycle and \
               self.cyclenumbers[i] <= Bcycle and \
               self.flags[i] == 'ACC' and \
               self.generation[i] != 'ld':
                idx.append(i)
                idx_w.append(self.weights[i])

        # 2. Sample N indices from the list of indices, respecting the weights
        idx_sample = random.choices(idx, weights=idx_w, k=N)

        # 3. Create a new path ensemble object
        pe_new = PathEnsemble()

        # 4. Update the path ensemble object with the sampled indices
        pe_new.cyclenumbers = self.cyclenumbers[idx_sample]
        pe_new.pathnumbers = self.pathnumbers[idx_sample]
        pe_new.newpathnumbers = self.newpathnumbers[idx_sample]
        pe_new.lmrs = self.lmrs[idx_sample]
        pe_new.lengths = self.lengths[idx_sample]
        pe_new.flags = self.flags[idx_sample]
        pe_new.generation = self.generation[idx_sample]
        pe_new.weights = self.weights[idx_sample]
        pe_new.lambmins = self.lambmins[idx_sample]
        pe_new.lambmaxs = self.lambmaxs[idx_sample]
        pe_new.name = self.name
        pe_new.interfaces = self.interfaces
        #pe_new.shootlinks = self.shootlinks[idx_sample]
        pe_new.in_zero_minus = self.in_zero_minus
        pe_new.in_zero_plus = self.in_zero_plus
        pe_new.has_zero_minus_one = self.has_zero_minus_one

        return pe_new    


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

    try: 
        pe_name = fn.split("/")[0] 
    except: 
        pe_name = fn
    finally: 
        pe.set_name(pe_name)

    return pe


def read_orderparameter(fn, ostart=0):
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

    subdata_list = []
    subdata = []

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
                subdata_list.append(subdata)
                subdata = []
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
                if len(words[1:])>1: data.append([float(word) for word in words[1:]])  # skip the time
                else:
                    data.append(float(words[1]))  # this is a stupied copy of longtraj
                    subdata.append(float(words[1]))

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

    subdata_list.append(subdata)
    subdata_list = subdata_list[1:]

    # verify lengths:
    assert_consistent(cyclenumbers,lengths,flags,generation,longtraj,data)
    return subdata_list

def strip_endpoints(order_list):
    stripped_order_list = []
    for orders in order_list:
        stripped_order_list.append(orders[1:-1])
    return stripped_order_list

def get_flat_list_and_weights(orders,weights):
    all_orders = np.array([item for sublist in orders for item in sublist])
    all_ws = []
    for w,o in zip(weights,orders):
        all_ws += [w]*len(o)
    return all_orders, all_ws

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
                if len(words[1:])>1: data.append([float(word) for word in words[1:]])  # skip the time
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


def get_weights(flags,ACCFLAGS,REJFLAGS,verbose=True):
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
    if verbose:
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

def get_LMR_interfaces(interfaces, zero_left):
    """Get the left, middle, right interfaces for each PyRETIS folder-ensemble"""
    LMR_interfaces = []
    LMR_strings = []
    if zero_left:
        LMR_interfaces.append([zero_left, (zero_left + interfaces[0])/2., interfaces[0]])
        LMR_strings.append(["l_[-1]", "( l_[-1] + l_[0] ) / 2", "l_[0]"])
    else:
        LMR_interfaces.append([interfaces[0], interfaces[0], interfaces[0]])
        LMR_strings.append(["l_[0]", "l_[0]", "l_[0]"])
    LMR_interfaces.append([interfaces[0], interfaces[0], interfaces[1]])
    LMR_strings.append(["l_[0]", "l_[0]", "l_[1]"])
    for i in range(1, len(interfaces)-1):
        LMR_interfaces.append([interfaces[i-1], interfaces[i], interfaces[i+1]])
        LMR_strings.append(["l_[{}]".format(i-1), "l_[{}]".format(i), "l_[{}]".format(i+1)])

    return LMR_interfaces, LMR_strings

########################################
def get_shoot_weights():
    """
    Get the weights of the paths as a function of the shoottrajectories only.
    """
    # We will do this in two steps:
    # 1) List the trajectories per ensemble, with correct weights.
    # 2) Swap moves require us to allocate the correct trajectory file.

    ### 1) Get the pathensemble weights, put them in a list. 
    pes = get_pes()

    ### 2) Non-shoot moves need their original shoot-path link
    linked_pes = link_pes(pes)
########################################


def get_shoot_weights_phase_1(pe):
    """
    Notes: we are not interested in load traj. So search for first accepted sh traj
    and start from there.
    You will have a problem in 000 and last ensemble, who don't have paths for all
    of the cycle numbers...
    """
    ACCFLAGS, REJFLAGS = set_flags_ACC_REJ()

    weights = np.zeros(len(pe.flags),int)
    weights[0] = 1 # Load path is accepted
    last_acc_cycnum = 0
    for i,flag in enumerate(pe.flags):
        if flag in REJFLAGS:
            weights[last_acc_cycnum] += 1
        else:
            weights[i] = 1
            last_acc_cycnum = i
    return weights

def get_shoot_weights_phase_2(pes,folders):
    ACCFLAGS, REJFLAGS = set_flags_ACC_REJ()
    
    # Here, for each cycle, we will note the link to the defining shoot-trajectory.
    # For rejected trajectories, NoneType will be found
    sh_src_folders = []
    
    for ens_idx, pe in enumerate(pes):
        print("working in pe number", ens_idx)
        cycindices = np.arange(len(pe.cyclenumbers))
        for cycidx, cycnum, flag, generation in zip(cycindices, pe.cyclenumbers, pe.flags,
                pe.generation):
            print("working on cycidx "+str(cycidx)+", which is cycnum "+str(cycnum))
            if flag in ACCFLAGS:
                if generation == "ld":
                    sh_src = "loadtraj"
                elif generation == "sh":
                    sh_src = folders[ens_idx]+"/traj/traj-acc/"+str(cycnum)+"/"
                else:
                    sh_src = find_sh_src(ens_idx, cycnum, generation, pes, folders)
                sh_src_folders.append(sh_src)
            else:
                assert flag in REJFLAGS
                sh_src = None
                sh_src_folders.append(sh_src)
            pe.update_shootlink(cycnum, sh_src)

            if flag in ACCFLAGS:
                assert sh_src is not None, "ACC, but no sh_src found?? cycnum: "+str(cycnum)+"gen: "+str(generation)


def find_sh_src(ens_idx, cycnum, generation, pes, folders):
    """ Recursive function to find the accepted shoot trajectory from which 
    the accepted non-shoot trajectory stems """
    target_ens_idx = None
    if generation == "tr":
        target_ens_idx = ens_idx
    elif generation == "s-":
        target_ens_idx = ens_idx - 1
        #print("I actually find this thing")
    elif generation == "s+": 
        target_ens_idx = ens_idx + 1
        #print("Also this one I actually find")
    elif generation == "ld":
        print("We should not see a load trajectory in this function")
    else:
        print("Euhm, weird generation: ", generation)
    assert target_ens_idx is not None

    prev_ACC_idx = find_prev_ACC_cyc_idx(cycnum, pes[target_ens_idx])
    # First we check whether this trajectory is already been linked to a shoot traj,
    # in this case we need not look further. Otherwise, we do look further.
    if pes[target_ens_idx].get_shootlink(pes[target_ens_idx].cyclenumbers[prev_ACC_idx]) is not None:
        return pes[target_ens_idx].get_shootlink(pes[target_ens_idx].cyclenumbers[prev_ACC_idx])
    else:
        if (pes[target_ens_idx]).generation[prev_ACC_idx] == "sh":
            testme = folders[target_ens_idx]+"/traj/traj-acc/"+str((pes[target_ens_idx]).cyclenumbers[prev_ACC_idx])+"/"
            return testme
        else: # RECURSIVE FUNCTION REQUIRES A RETURN STATEMENT AAAAAH
            return find_sh_src(target_ens_idx, (pes[target_ens_idx]).cyclenumbers[prev_ACC_idx],
                (pes[target_ens_idx]).generation[prev_ACC_idx], pes, folders)

def find_prev_ACC_cyc_idx(target_cycnum, pe):
    ACCFLAGS, REJFLAGS = set_flags_ACC_REJ()
    cycnums = pe.cyclenumbers
    # I used to have x > target_cycnum, but gives err for last idx.
    # No now i take >=, and then give text_idx + 1 to start with ...
    first_larger_cyc_idx = list(x >= target_cycnum for x in cycnums).index(True)
    test_idx = first_larger_cyc_idx+1
    found = False
    while not found:
        test_idx -= 1
        if pe.flags[test_idx] in ACCFLAGS and cycnums[test_idx] < target_cycnum:
            found = True
    # We assume we find an accepted path ...
    #print("test_idx_found "+str(test_idx)+ " with cycnum "+str(cycnums[test_idx]))
    return test_idx


from json import load
import numpy as np
from .reading import *
import logging
import bisect

# Hard-coded rejection flags found in output files
ACCFLAGS,REJFLAGS = set_flags_ACC_REJ() 

# logger
logger = logging.getLogger(__name__)


# Masks for the paths #
# ------------------- #
def get_lmr_masks(pe, masktype="all"):
    """
    Return one (or all) boolean array(s) of the paths, based on whether or not
    path i is of type masktype. 
    """
    # Define the types of paths 
    types = ["RML","RMR","LMR","LML","***","**L","**R","*M*","*ML",
             "*MR","L**","L*L","LM*","R**","R*R","RM*"]
    
    # Obtain the boolean masks for each path type
    masks = [pe.lmrs == t for t in types]

    # Create a dictionary of types with their corresponding masks
    masks = dict(zip(types, masks))

    if masktype == "all":
        return masks
    else:
        return masks[masktype], masktype


def get_flag_mask(pe, status):
    """
    Returns boolean array 
    """
    if status == "REJ":
        return ~get_flag_mask(pe, "ACC")
    else:
        return pe.flags == status 


def get_generation_mask(pe, generation):
    """
    Returns boolean array
    """
    return pe.generation == generation


def get_hard_load_mask(loadmask):
    """Returns boolean array. All paths corresponding to cycles before the
    last load cycle (included), are set to True. The others are unchanged.
    """
    # Get the last load cycle
    last_load_cycle = loadmask.argmax()
    # first make a copy of the loadmask
    hard_load_mask = loadmask.copy()
    # Set all paths before and during the last load cycle to True
    hard_load_mask[:last_load_cycle+1] = True
    return hard_load_mask


def select_with_masks(A, masks):
    """
    Returns the elements of the array A that are True in all of 
    the masks, where each mask is a similarly sized boolean array.
    
    A: np.array
    masks: list of masks, where each mask is a boolean array 
            with the same shape as A
    """
    # first check whether masks have the same shape as A
    for mask in masks:
        assert mask.shape == A.shape
    # now we can use the masks to select the elements of A
    union_mask = np.all(masks,axis=0).astype(bool)
    return A[union_mask]

def select_with_OR_masks(A, masks):
    """
    Returns the elements of the array A that are True in at least one of 
    the masks, where each mask is a similarly sized boolean array.
    
    A: np.array
    masks: list of masks, where each mask is a boolean array 
            with the same shape as A
    """
    # first check whether masks have the same shape as A
    for mask in masks:
        assert mask.shape == A.shape
    # now we can use the masks to select the elements of A
    union_mask = np.any(masks,axis=0).astype(bool)
    return A[union_mask]


# Statistical analysis #
# -------------------- #
def bootstrap(data, N=1000, Ns=None):
    """
    Bootstrap the data.

    Parameters
    ----------
    data : numpy.array
        The data can be multidimensional, but the first dimension must be the
        time/cycle dimension.
    N : int, optional
        The number of times to bootstrap the data
    Ns : int, optional
        The number of samples to take from the data. If not specified, then
        Ns = len(data).
    """

    # Check if Nsamples is specified
    Nsamples = Nsamples if Nsamples is not None else len(data)

    # Bootstrap the data
    boot_data = [np.random.choice(data, Nsamples) for _ in range(N)]

    return boot_data 



def bootstrap_analysis_repptis(pathensembles, nN=10, nB=1000):
    """
    Perform a boostrap analysis on REPPTIS simulation data. 

    Parameters
    ----------
    pathensembles : list of PathEnsemble objects
        The PathEnsemble objects must be from REPPTIS simulations.
    zero_left : bool, optional
        If True, then the zero_left interface was used in the [0^-'] ensemble.
        In this case, the [0^-'] ensemble has interfaces
        [lambda_(zero_left), (lambda_(zero_left)+lambda_0)/2, lambda_0], or in
        pathensembe terms: [L, M, R] = [L, (L+R)/2, R].
        If False, then the zero_left interface was not used in the [0^-'] 
        ensemble, and [L,M,R] = [-inf, R, R] = [-inf, lambda_zero, lambda_zero].
    nN : int, optional
        Bootstrap will be performed nN times on data ranging from cycle 0 to 
        cycle N//nN, where N is the total number of cycles in the simulation.
        The arange[N//nN, N, N//nN] is called the timeslice for which 
        bootstrap is performed.
    nB : int, optional
        The number of bootstrap samples to take for each bootstrap analysis.
    """

    # First, we collect the data from all the pathensembles, which includes
    # the number of paths in each ensemble, their weights, their path types, 
    # and their generation.
    data = {}
    for i, pe in enumerate(pathensembles):
        # We produce a dictionary of the data for each pathensemble
        pe_data = {}
        # Not interested in the zero minus ensemble
        if pe.in_zero_minus:
            logger.info(f"Passing pathensemble {i} because this is the zero "+\
                        f"minus ensemble: {pe.name}")
            pass
        logger.info(f"Doing pathensemble {i}")
        for Bcycle in np.arange((pe.cyclenumbers)[-1]//nN,
                                (pe.cyclenumbers)[-1],
                                ((pe.cyclenumbers)[-1]//nN)):
            logger.info(f"Doing bootstrap analysis for cycle {Bcycle}")
            # We produce a list of the data for each timeslice
            pe_ts_data = []
            for j in range(nB):
                # A. bootstrap the pathensemble
                pe_boot = pe.bootstrap_pe(nB, Bcycle)
                # B. get the local crossing probabilities
                ploc = get_local_probs(pe_boot)
                pe_ts_data.append(ploc)
            # C. Calculate the mean and std of the local crossing probabilities
            #    for each path type. Use numpy mean and std functions.
            p = {}
            for pathtyp in ("RMR", "RML", "LMR", "LML"):
                p[pathtyp] = {'mean': np.mean([d[pathtyp] for d in pe_ts_data]),
                              'std': np.std([d[pathtyp] for d in pe_ts_data])}
            pe_data[Bcycle] = p
        data[i] = pe_data
    
    # Calculate the global crossing probabilities, for each timeslice Bcycle. 
    # We do this by putting the local crossing probabilities (and stddevs) of 
    # all the pathensembles together in a list, for one give Bcycle. Then we 
    # calculate the mean and std of the global crossing probabilities for this
    # Bcycle.


    return data 
        


def bootstrap_repptis(pathensembles, nN=10, nB=1000):
    """
    Perform a boostrap analysis on REPPTIS simulation data. The strategy of 
    this bootstrap analysis will be different. The first loop will be over the 
    timeslices. Then bootstrapping takes place for each timeslice, where cycle
    numbers are chosen randomly within the timeslice. The paths corresponding 
    to those cycle numbers are then used to calculate the local crossing 
    probabilities in all the pathensembles. The global crossing probabilities 
    are then calculated from those local crossing probabilities. 

    Parameters
    ----------
    pathensembles : list of PathEnsemble objects
        The PathEnsemble objects must be from REPPTIS simulations.
    zero_left : bool, optional
        If True, then the zero_left interface was used in the [0^-'] ensemble.
        In this case, the [0^-'] ensemble has interfaces
        [lambda_(zero_left), (lambda_(zero_left)+lambda_0)/2, lambda_0], or in
        pathensembe terms: [L, M, R] = [L, (L+R)/2, R].
        If False, then the zero_left interface was not used in the [0^-'] 
        ensemble, and [L,M,R] = [-inf, R, R] = [-inf, lambda_zero, lambda_zero].
    nN : int, optional
        Bootstrap will be performed nN times on data ranging from cycle 0 to 
        cycle N//nN, where N is the total number of cycles in the simulation.
        The arange[N//nN, N, N//nN] is called the timeslice for which 
        bootstrap is performed.
    nB : int, optional
        The number of bootstrap samples to take for each bootstrap analysis.
    """

    data = {}
    # for each pathensemble, we save the indices of accepted cycle numbers in a
    # dictionary, because we will use this a lot. We do not accept load cycles, 
    # so if a load cycle is sampled, we will just not use it. 
    pathcycle_ids = {}
    for i, pe in enumerate(pathensembles):
        loadmask = get_generation_mask(pe, "load")
        accmask = get_flag_mask(pe, "ACC")
        pathcycle_ids[i] = select_with_masks(pe.cyclenumbers,
                                             [accmask, ~loadmask])

    for Bcycle in np.arange((pathensembles[0].cyclenumbers)[-1]//nN,
                            (pathensembles[0].cyclenumbers)[-1],
                            ((pathensembles[0].cyclenumbers)[-1]//nN)):
        logger.info(f"Doing bootstrap analysis for cycle {Bcycle}")
        # We produce a list of the data for each timeslice
        ts_data = {}
        for j in range(nB):
            if j % 100 == 0:
                logger.info(f"Doing bootstrap sample {j}")
            # A. Select cycle numbers randomly within the timeslice [1, Bcycle],
            #    using replacement. We start from one to discard the initial
            #    load cycle.
            cycle_ids = np.random.choice(np.arange(start=1,stop=Bcycle), 
                                         Bcycle, replace=True)
            # Store the data for each pathensemble in a dictionary
            boot_data = {}
            boot_data['ens'] = {}
            for i, pe in enumerate(pathensembles):
                if pe.in_zero_minus:
                    logger.info(f"Passing pathensemble {i} because this is "+\
                                f"the zero minus ensemble: {pe.name}")
                    pass
                boot_data['ens'][i] = {}  # init dict, after discard zero_minus
                # map the cycle numbers to the indices of accepted cycles
                boot_cycle_ids = find_closest_number_lte(cycle_ids,
                                                         pathcycle_ids[i])
                # sample the pathensemble at the given cycle indices
                boot_pe = pe.sample_pe(boot_cycle_ids)
                #boot_data[i]['pe'] = boot_pe
                # B. get the local crossing probabilities
                ploc = get_local_probs(boot_pe)
                boot_data['ens'][i]['ploc'] = ploc
            # C. Calculate the global crossing probabilities 
            Pmin, Pplus, Pcross = get_global_probs([boot_data['ens'][i]['ploc'] 
                                                    for i in 
                                                    boot_data['ens'].keys()])
            boot_data['Pmin'] = Pmin
            boot_data['Pplus'] = Pplus
            boot_data['Pcross'] = Pcross
            ts_data[j] = boot_data
        # save the boot_data for this timeslice 
        data[Bcycle] = {}
        data[Bcycle]['data'] = ts_data
        # D. Calculate the mean and std of the local crossing probabilities for
        #    each pathensemble, and the mean and std of the global crossing for 
        #    each bootstrap sample.
        ts_stats = {}
        # first the local crossing probabilities
        for i in boot_data['ens'].keys():
            ts_stats[i] = {}
            ts_stats[i]['ploc'] = {}
            # know that ts_data[j]['ens'][i]['ploc'] is a dict with keys 
            for ptype in ts_data[j]['ens'][i]['ploc'].keys():
                ts_stats[i]['ploc'][ptype] = {}
                ts_stats[i]['ploc'][ptype]['mean'] = \
                    np.mean([ts_data[j]['ens'][i]['ploc'][ptype] \
                    for j in ts_data.keys()])
                ts_stats[i]['ploc'][ptype]['std'] = \
                    np.std([ts_data[j]['ens'][i]['ploc'][ptype] \
                    for j in ts_data.keys()])
        # then the global crossing probabilities
        # know that ts_data[j]['Pmin'] is a list, and we want the mean and std
        # for each element of the list. (averaging is done over the bootstrap
        # samples)
        for crosstype in ['Pmin', 'Pplus', 'Pcross']:
            # print(ts_data.keys())
            # print(ts_data[j][crosstype])
            ts_stats[crosstype] = {}
            ts_stats[crosstype]['mean'] = \
                np.mean(np.array([ts_data[j][crosstype]
                                  for j in ts_data.keys()]),axis=0)
            ts_stats[crosstype]['std'] = \
                np.std(np.array([ts_data[j][crosstype]
                                 for j in ts_data.keys()]),axis=0)
                                               
        data[Bcycle]['stats'] = ts_stats

    return data

def analyse_bootstrap_data(data):
    """
    Analyse the bootstrap data and return the mean and std of the local and 
    global crossing probabilities.
    
    Parameters
    ----------
    data : dict
        Dictionary of bootstrap data.
        
    Returns
    -------
    stats : dict
        Dictionary of mean and std of the local and global crossing 
        probabilities.
    """
    stats = {}
    for Bcycle in data.keys():
        stats[Bcycle] = {}
        stats[Bcycle]['ploc'] = {}
        stats[Bcycle]['ploc']['mean'] = {}
        stats[Bcycle]['ploc']['std'] = {}
        for i in data[Bcycle]['stats'].keys():
            if i == 'Pmin' or i == 'Pplus' or i == 'Pcross':
                stats[Bcycle][i] = {}
                stats[Bcycle][i]['mean'] = data[Bcycle]['stats'][i]
                stats[Bcycle][i]['std'] = 0
            else:
                for ptype in data[Bcycle]['stats'][i]['ploc'].keys():
                    stats[Bcycle]['ploc']['mean'][i, ptype] = \
                        data[Bcycle]['stats'][i]['ploc'][ptype]['mean']
                    stats[Bcycle]['ploc']['std'][i, ptype] = \
                        data[Bcycle]['stats'][i]['ploc'][ptype]['std']
    return stats



def find_closest_number_lte(A, B):
    """
    For each element in A, search the closest number in B that is smaller or 
    equal to the element in A. If there is no number in B that is smaller or 
    equal to the element in A, then it is discarded
    
    Parameters
    ----------
    A : ndarray
        Array of numbers to find closest numbers for.
    B : ndarray
        Array of numbers to find closest numbers from.
        
    Returns
    -------
    C : ndarray
        Array of closest numbers in B that are smaller or equal to the numbers 
        in A. If 
    """
    B = np.sort(B)  # Sort B in ascending order
    C = []
    for i, a in enumerate(A):
        j = bisect.bisect_right(B, a)  # Find idx where a is to be inserted in B
        if j != 0:  # if zero, then a is smaller than all elements in B: discard
            # closest number <= a is the one just before the index j
            C.append(B[j-1])
    return C

def inefficiency(corr):
    """Calculate the integrated autocorrelation time of a time series.
    
    Parameters
    ----------
    corr : array-like
        The autocorrelation function of the time series.
        
    Returns
    -------
    tau_int : float
        The integrated autocorrelation time.
        
    """
    tau_int = 0
    for i in range(1, len(corr)):
        if corr[i] < 0:  # negative correlation, so time series is decorrelated
            break
        tau_int += corr[i]
    return 1 + 2 * tau_int

def block(x, block_size):
    """Calculate the mean and standard error of the mean of a time series
    using blocking.

    Parameters
    ----------
    x : array-like
        The time series.
    block_size : int
        The number of samples in each block.

    Returns
    -------
    block_size : int
        The number of samples in each block.
    tau_int : float
        The integrated autocorrelation time.
    num_ind : float
        The number of independent samples.
    mean : float
        The mean of the time series.
    se_mean : float
        The standard error of the mean of the time series.

    """
    # determine the number of blocks that can be formed
    num_blocks = np.floor(len(x) / block_size).astype(int)
    # trim x to have a multiple of block_size
    x_block = x[:num_blocks * block_size]
    # reshape x to a matrix of blocks
    x_block = x_block.reshape(num_blocks, block_size)
    # calculate the mean of each block
    mean_block = np.mean(x_block, axis=1)
    # calculate the mean and standard error of the means of the blocks
    mean = np.mean(mean_block)
    se_mean = np.std(mean_block, ddof=1) / np.sqrt(num_blocks - 1)
    # calculate the autocorrelation function of the block means
    corr = np.correlate(mean_block - mean, mean_block - mean, mode='full')
    # trim the autocorrelation function to remove the negative lags
    corr = corr[corr.size // 2:]
    # Normalize the autocorrelation function (corr = 1 at lag 0)
    corr = np.nan_to_num(corr/corr[0])
    # calculate the integrated autocorrelation time of the block means
    tau_int = inefficiency(corr)
    # calculate the effective sample size of the blocked data
    num_ind = np.nan_to_num(np.floor( \
        num_blocks * block_size / tau_int).astype(int))
    return (block_size, tau_int, num_ind, mean, se_mean)


def full_block_analysis(x):
    """Perform a full block analysis of a time series.

    Parameters
    ----------
    x : array-like
        The time series.

    Returns
    -------
    opt_block_size : int
        The optimal block size.
    opt_tau_int : float
        The integrated autocorrelation time.
    opt_num_ind : float
        The number of independent samples.
    opt_mean : float
        The mean of the time series.
    opt_se_mean : float
        The standard error of the mean of the time series.
    block_sizes : array-like
        The block sizes.
    inefficiencies : array-like
        The inefficiencies.
    se_means : array-like
        The standard errors of the means.

    """

    # set the range of block sizes to consider
    min_block_size = 1
    max_block_size = min(1000, len(x) // 2)
    results = []
    # loop over block sizes and perform blocking for each size
    for block_size in range(min_block_size, max_block_size):
        result = block(x, block_size)
        results.append(result)
    results = np.array(results)
    # extract relevant statistics from the results
    block_sizes = results[:, 0]
    inefficiencies = results[:, 1]
    se_means = results[:, 4]
    # determine the optimal block size based on integrated autocorrelation time
    min_idx = np.argmin(inefficiencies)
    opt_block_size, opt_tau_int, opt_num_ind, opt_mean, opt_se_mean = \
        results[min_idx]
    # return relevant statistics and results for all block sizes
    return (opt_block_size, opt_tau_int, opt_num_ind, opt_mean, opt_se_mean, 
            block_sizes, inefficiencies, se_means)

def running_average(x):
    """Calculate the running average of a time series.

    Parameters
    ----------
    x : array-like
        The time series.

    Returns
    -------
    x_avg : array-like
        The running average of the time series.

    """
    cumsum = np.cumsum(x)
    x_avg = cumsum / np.arange(1, len(x)+1)
    return x_avg


def unwrap_by_weight(v, w=None):
    """ v is a vector (one dimensional) containing real numbers, while 
    w contains integer numbers representing the weights of the samples. 
    The function returns the unwrapped vector u, where the zero-weight
    samples are replaced by the previous non-zero-weight sample.
    
    """

    return np.repeat(v, w)

def running_avg_local_probs(pathtype_cycles, w, tr = False):
    """
    
    Parameters
    ----------
    pathtype_cycles : dictionary of Ncycle arrays 
        The keys are the pathtypes (LMR, LML, etc.), which contain 0/1 arrays
        indicating the presence of each pathtype in each cycle. 
    w : array-like
        The weights of the paths
    """

    cumsums = {}
    for key in pathtype_cycles.keys():
        cumsums[key] = np.cumsum(pathtype_cycles[key])
    if tr:
        cumsums["LML"] = 2*cumsums["LML"]
        cumsums["RMR"] = 2*cumsums["RMR"]
        temp = cumsums["LMR"] + cumsums["RML"]
        cumsums["LMR"] = temp
        cumsums["RML"] = temp
    p_NP = cumsums['LMR']/(cumsums['LMR']+cumsums['LML'])
    p_NN = cumsums['LML']/(cumsums['LMR']+cumsums['LML'])
    p_PP = cumsums['RMR']/(cumsums['RML']+cumsums['RMR'])
    p_PN = cumsums['RML']/(cumsums['RML']+cumsums['RMR'])
    return p_NN, p_NP, p_PN, p_PP

def cross_dist_distr(pe):
    """Return the distribution of lambda values for the LMR and RML paths.

    It calculates the distribution of lambda_max values for LM* paths, and
    the distribution of lambda_min values for RM* paths.

    Parameters
    ----------
    pe : object like :py:class:`.PathEnsemble`
        A PPTIS path ensemble object (tistools-processed).

    Returns
    -------
    L : float
        The left interface of the pathensemble.
    M : float
        The middle interface of the pathensemble.
    R : float
        The right interface of the pathensemble.
    percents : array of floats
        The distribution of lambda_max values for LM* paths, given at lambda
        values 'lambs' between middle and right interfaces.
    lambs : array of floats
        The lambda values at which the lambda_max distribution is given.
    percents2 : array of floats
        The distribution of lambda_min values for RM* paths, given at lambda
        values 'lambs2' between left and middle interfaces.
    lambs2 : array of floats
        The lambda values at which the lambda_min distribution is given.

    """
    # LM*:
    paths = select_with_OR_masks(pe.lambmaxs, [pe.lmrs == "LML",
                                               pe.lmrs == "LMR"])
    w = select_with_OR_masks(pe.weights, [pe.lmrs == "LML",
                                          pe.lmrs == "LMR"])
    repeat = np.repeat(paths, w)
    L,M,R = pe.interfaces[0][0], pe.interfaces[0][1], \
        pe.interfaces[0][2]
    percents = []

    lambs = np.linspace(M, np.max(paths), 100) if paths.size != 0 else \
        np.linspace(M, R, 100)
    for i in lambs:
        percents.append(np.sum(repeat >= i))
    percents = percents/percents[0] if percents else percents
        
    # RM*:
    paths2 = select_with_OR_masks(pe.lambmins, [pe.lmrs == "RMR",
                                                pe.lmrs == "RML"])
    w2 = select_with_OR_masks(pe.weights, [pe.lmrs == "RMR",
                                           pe.lmrs == "RML"])
    repeat2 = np.repeat(paths2, w2)
    percents2 = []
    lambs2 = np.linspace(np.min(paths2), M, 100) if paths2.size != 0 else \
        np.linspace(L, M, 100)
    for i in lambs2:
        percents2.append(np.sum(repeat2 <= i))
    percents2 = percents2/percents2[-1] if percents2 else percents2
    return L, M, R, percents, lambs, percents2, lambs2

def pathlength_distr(upe):
    """
    Create the pathlength distributions for the path types (LML, ...) of a path
    ensemble. The path ensemble is assumed to be unified already.

    Parameters
    ----------
    upe : PathEnsemble object (unified)

    Returns
    -------
    data : dict containing the pathlength distributions for each path type
    """

    pathtypes = ("LML", "LMR", "RML", "RMR")
    data = {}
    for ptype in pathtypes:
        data[ptype] = {"mean": np.nan, "std": np.nan}
        pathlengths = select_with_masks(upe.lengths, 
                                        [upe.lmrs == ptype])
        data[ptype]["mean"] = np.mean(pathlengths)
        data[ptype]["std"] = np.std(pathlengths)
        hist, bin_edges = np.histogram(pathlengths, bins=100)
        bin_centers = (bin_edges[:-1] + bin_edges[1:])/2
        data[ptype]["hist"] = hist
        data[ptype]["bin_centers"] = bin_centers
    return data


def get_local_probs(pe, w = None, tr=False):
    """
    Returns the local crossing probabilities for the PPTIS ensemble pe.
    This is only for the [i^+-] or [0^+-'] ensembles, and NOT for [0^-'].

    Parameters
    ----------
    pe : PathEnsemble object
        The PathEnsemble object must be from a PPTIS simulation.
    w : array-like, optional
        The weights of the paths. If None, the weights are calculated from
        the flags. The default is None.
    tr : bool, optional
        If True, infinite time-reversal reweighting is performed.
    Returns
    -------
    p : dict
        A dictionary with the local crossing probabilities for the PPTIS
        ensemble pe. The keys are the path types (RMR, RML, LMR, LML), and 
        the values are the local crossing probabilities.
    """
    # Get the lmr masks, weights, ACCmask, and loadmask of the paths
    masks = get_lmr_masks(pe)
    if w is None:
        w, _ = get_weights(pe.flags, ACCFLAGS, REJFLAGS, verbose = False)
    accmask = get_flag_mask(pe, "ACC")
    loadmask = get_generation_mask(pe, "ld")
    msg = f"Ensemble {pe.name} has {len(w)} paths.\n The total "+\
            f"weight of the ensemble is {np.sum(w)}\nThe total amount of "+\
            f"accepted paths is {np.sum(accmask)}\nThe total amount of "+\
            f"load paths is {np.sum(loadmask)}"
    logger.debug(msg)
    # Get the weights of the RMR, RML, LMR and LML paths
    w_path = {}
    for pathtype in ["RMR", "RML", "LMR", "LML"]:
        w_path[pathtype] = np.sum(select_with_masks(w, [masks[pathtype],
                                                        accmask, ~loadmask]))
    msg = "Weights of the different paths:\n"+f"wRMR = {w_path['RMR']}\n"+\
            f"wRML = {w_path['RML']}\nwLMR = {w_path['LMR']}\n"+\
            f"wLML = {w_path['LML']}"
    print(msg)
    if tr:  # TR reweighting. Note this is not block-friendly TODO
        w_path['RMR'] = 2*w_path['RMR']
        w_path['LML'] = 2*w_path['LML']
        temp = w_path['RML'] + w_path['LMR']
        w_path['RML'] = temp
        w_path['LMR'] = temp
    # Get the total weight of paths starting from left, or from right
    wR = w_path['RMR'] + w_path['RML']
    wL = w_path['LMR'] + w_path['LML']
    # And calculate local crossing probabilities
    p = {}
    for pathtype, direc_w in zip(("RMR", "RML", "LMR", "LML"),
                                (wR, wR, wL, wL)):
        p[pathtype] = w_path[pathtype]/direc_w if direc_w != 0 else np.nan
    msg = "Local crossing probabilities:\n"+f"pRMR = {p['RMR']}\n"+\
            f"pRML = {p['RML']}\npLMR = {p['LMR']}\npLML = {p['LML']}"
    print(msg)

    # Extra for milestoning
    #----------------------
    # Get the total weight of paths arriving to left, or to right
    w2R = w_path['LMR'] + w_path['RMR']
    w2L = w_path['LML'] + w_path['RML']
    w_tot = w2R + w2L
    # And calculate local crossing probabilities
    p["2R"] = w2R/w_tot if w_tot != 0 else np.nan
    p["2L"] = w2L/w_tot if w_tot != 0 else np.nan
    msg = "Local crossing probabilities:\n"+f"p2R = {p['2R']}\n"+\
            f"p2L = {p['2L']}"
    print(msg)

    return p

        
def get_globall_probs(ps):
    """Returns the global crossing probabilities for the PPTIS simulation, 
    together with the standard deviation of these probabilities.

    This follows the recursive relation 
    P_plus[j] = (pLMR[j-1]*P_plus[j-1]) / (pLMR[j-1]+pLML[j-1]*P_min[j-1])
    P_min[j] = (pRML[j-1]P_min[j-1])/(pLMR[j-1]+pLML[j-1]*P_min[j-1])
    where P_plus[0] = 1, P_min[0] = 1, and j is the ensemble number.

    Parameters
    ----------
    ps : list of dicts
        A list of dictionaries with the local crossing probabilities for each
        PPTIS ensemble. The keys are the path types (RMR, RML, LMR, LML).
        ps[i]['LMR']['mean'] and ps[i]['LMR']['std'] are the mean and stdev of
        p_{i}^{mp}, respectively.

    Returns
    -------
    Pmin : list of floats
        Float per ensemble i. Represents the probability of crossing A earlier
        than i+1, given that the path crossed i.
    Pplus : list of floats
        Float per ensemble i. Represents the probability of crossing i+1 earlier
        than A, given that the path crossed i. 
    Pcross : list of floats
        Float per ensemble i. Represents the TIS probability of crossing i+1
    """

    Pplus, Pmin, Pcross = [1.], [1.], [1., ps[1]['LMR']]
    for i, p in enumerate(ps):
        if i <= 1: continue  # This is [0^{\pm}'], so skip
        # Calculate the global crossing probabilities
        Pplus.append((p['LMR']*Pplus[-1])/(p['LMR']+p['LML']*Pmin[-1]))
        Pmin.append((p['RML']*Pmin[-1])/(p['LMR']+p['LML']*Pmin[-1]))
        # Calculate the TIS probabilities
        Pcross.append(ps[1]['LMR']*Pplus[-1])
    return Pmin, Pplus, Pcross

def get_global_probz(pmps, pmms, ppps, ppms):
    """Return the global crossing probabilities for the PPTIS simulation.

    This follows the recursive relation
    P_plus[j] = (pLMR[j-1]*P_plus[j-1]) / (pLMR[j-1]+pLML[j-1]*P_min[j-1])
    P_min[j] = (pRML[j-1]P_min[j-1])/(pLMR[j-1]+pLML[j-1]*P_min[j-1])
    where P_plus[0] = 1, P_min[0] = 1, and j is the ensemble number.

    Parameters
    ----------
    pmps : list of floats
        The local probability (P) of having type LMR (MinusPlus) for each
        ensemble (S).
    pmms : list of floats
        Local LML
    ppps : list of floats
        Local RMR
    ppms : list of floats
        Local RML

    Returns
    -------
    pmin : list of floats
        Float per ensemble i. Represents the probability of crossing A earlier
        than i+1, given that the path crossed i.
    pplus : list of floats
        Float per ensemble i. Represents the probability of crossing i+1
        earlier than A, given that the path crossed i.
    pcross : list of floats
        Float per ensemble i. Represents the TIS probability of crossing i+1.
    """
    # if there is any NaN in pmps, pmms, ppps, ppms, return NaN
    if np.isnan(pmps).any() or np.isnan(pmms).any() or \
            np.isnan(ppps).any() or np.isnan(ppms).any():
        return [np.nan, np.nan, np.nan]
    pplus, pmin, pcross = [1.], [1.], [1., pmps[0]]
    for i, pmp, pmm, _, ppm in zip(range(len(pmps)), pmps, pmms, ppps, ppms):
        if i == 0:  # This is [0^{\pm}'], so skip
            continue
        # Calculate the global crossing probabilities
        # if divide by zero, or divide by NaN, return NaN
        if pmp + pmm * pmin[-1] == 0 or\
                (pmp is np.nan) or\
                (pmm * pmin[-1] is np.nan):
            return [np.nan, np.nan, np.nan]
        pplus.append((pmp*pplus[-1])/(pmp+pmm*pmin[-1]))
        pmin.append((ppm*pmin[-1])/(pmp+pmm*pmin[-1]))
        # Calculate the TIS probabilities
        pcross.append(pmps[0]*pplus[-1])
    return pmin, pplus, pcross


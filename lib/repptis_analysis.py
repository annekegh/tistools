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
    Retrieve boolean mask(s) indicating which paths in the ensemble match a given type.

    Parameters
    ----------
    pe : PathEnsemble
        The path ensemble from which to extract masks.
    masktype : str, optional
        The path type(s) to include in the masks (e.g., "RMR"). Defaults to "all",
        meaning masks for all available path types will be returned.

    Returns
    -------
    dict
        A dictionary where keys are path types and values are boolean arrays 
        indicating which paths in `pe` belong to the corresponding type.
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
    Generate a boolean mask indicating which paths in the ensemble match a given status.

    Parameters
    ----------
    pe : PathEnsemble
        The path ensemble from which to extract the mask.
    status : str
        The status flag to filter paths by (e.g., "ACC" for accepted paths, "REJ" for rejected paths).
        If "REJ" is specified, the mask will be the complement of "ACC".

    Returns
    -------
    np.ndarray
        A boolean array where `True` indicates that a path in `pe` has the specified status.
    """
    if status == "REJ":
        return ~get_flag_mask(pe, "ACC")
    else:
        return pe.flags == status 


def get_generation_mask(pe, generation):
    """
    Generate a boolean mask indicating which paths in the ensemble belong to a given generation.

    Parameters
    ----------
    pe : PathEnsemble
        The path ensemble from which to extract the mask.
    generation : int
        The generation number to filter paths by.

    Returns
    -------
    np.ndarray
        A boolean array where `True` indicates that a path in `pe` belongs to the specified generation.
    """
    return pe.generation == generation


def get_hard_load_mask(loadmask):
    """
    Generate a boolean mask where all paths corresponding to cycles up to and including 
    the last load cycle are set to True.

    Parameters
    ----------
    loadmask : np.ndarray
        A boolean array indicating load cycles.

    Returns
    -------
    np.ndarray
        A boolean array where all elements up to and including the last `True` value 
        in `loadmask` are set to `True`, while others remain unchanged.
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
    Select elements from an array based on multiple boolean masks.

    Parameters
    ----------
    A : np.ndarray
        The input array from which elements are selected.
    masks : list of np.ndarray
        A list of boolean arrays, each having the same shape as `A`. 
        Elements in `A` are selected only if they are `True` in all masks.

    Raises
    ------
    ValueError
        If any mask does not have the same shape as `A`.

    Returns
    -------
    np.ndarray
        A 1D array containing elements of `A` where all masks are `True`.
    """
    # Check whether all masks have the same shape as A
    for i, mask in enumerate(masks):
        if mask.shape != A.shape:
            raise ValueError(f"Mask at index {i} has shape {mask.shape}, expected {A.shape}")

    # Compute the intersection of all masks and select elements from A
    union_mask = np.all(masks, axis=0)
    return A[union_mask]

def select_with_OR_masks(A, masks):
    """
    Select elements from an array based on multiple boolean masks using a logical OR operation.

    Parameters
    ----------
    A : np.ndarray
        The input array from which elements are selected.
    masks : list of np.ndarray
        A list of boolean arrays, each having the same shape as `A`. 
        Elements in `A` are selected if they are `True` in at least one mask.

    Raises
    ------
    ValueError
        If any mask does not have the same shape as `A`.

    Returns
    -------
    np.ndarray
        A 1D array containing elements of `A` where at least one mask is `True`.
    """
    # Check whether all masks have the same shape as A
    for i, mask in enumerate(masks):
        if mask.shape != A.shape:
            raise ValueError(f"Mask at index {i} has shape {mask.shape}, expected {A.shape}")

    # Compute the union of all masks and select elements from A
    union_mask = np.any(masks, axis=0)
    return A[union_mask]


# Statistical analysis #
# -------------------- #
def bootstrap(data, N=1000, Ns=None):
    """
    Perform bootstrapping on the given data.

    Parameters
    ----------
    data : numpy.ndarray
        The input data, which can be multidimensional. The first dimension
        must represent the time or cycle dimension.
    N : int, optional
        The number of bootstrap samples to generate. Default is 1000.
    Ns : int, optional
        The number of samples to draw from the data for each bootstrap sample. 
        If not specified, `Ns` defaults to the length of the first dimension of `data`.

    Returns
    -------
    list of numpy.ndarray
        A list containing `N` bootstrap samples, each with `Ns` samples drawn from `data`.

    Notes
    -----
    Each bootstrap sample is created by randomly selecting `Ns` samples from `data` with replacement.
    """
    # Check if Nsamples is specified
    Nsamples = Ns if Ns is not None else len(data)

    # Bootstrap the data
    boot_data = [np.random.choice(data, Nsamples) for _ in range(N)]

    return boot_data 



def bootstrap_local_repptis(pathensembles, nN=10, nB=1000):
    """
    Perform bootstrap analysis on each :py:class:`.PathEnsemble` individually to estimate local crossing probabilities.

    This function analyzes each :py:class:`.PathEnsemble` separately, splitting the simulation cycles into timeslices
    and performing bootstrap resampling within each timeslice. It calculates the mean and standard deviation
    of local crossing probabilities for each path type within each timeslice.

    Parameters
    ----------
    pathensembles : list of PathEnsemble
        A list of :py:class:`.PathEnsemble` objects from REPPTIS simulations. Each object contains simulation data
        for analysis.
    nN : int, optional
        The number of timeslices to divide the simulation cycles into. Each timeslice corresponds to
        a range of cycles from 0 to N//nN, where N is the total number of cycles. Default is 10.
    nB : int, optional
        The number of bootstrap samples to generate for each timeslice. Default is 1000.

    Returns
    -------
    dict
        A nested dictionary where the keys are the indices of the :py:class:`.PathEnsemble` objects. For each ensemble,
        the value is another dictionary containing the mean and standard deviation of local crossing
        probabilities for each path type (e.g., "RMR", "RML", "LMR", "LML") within each timeslice.

    Notes
    -----
    - The zero minus ensemble is skipped during analysis.
    - Local crossing probabilities are calculated for each bootstrap sample, and their statistics
      (mean and standard deviation) are computed for each path type.

    The analysis involves the following steps:
    1. Collect data from each :py:class:`.PathEnsemble`, skipping zero minus ensembles.
    2. Perform bootstrap analysis over cycles, creating timeslices for each cycle range.
    3. For each bootstrap sample, calculate local crossing probabilities for each path type.
    4. Compute the mean and standard deviation of the local probabilities for each path type.
    5. Calculate global crossing probabilities by aggregating the results from all :py:class:`.PathEnsemble` objects.
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
        

def global_bootstrap_repptis(pathensembles, nN=10, nB=1000):
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
    Analyze bootstrap data to compute the mean and standard deviation of local and global crossing probabilities.

    This function processes the results of a bootstrap analysis, extracting and organizing the mean and standard
    deviation of local crossing probabilities for each ensemble and path type, as well as global crossing
    probabilities (Pmin, Pplus, Pcross) for each timeslice.

    Parameters
    ----------
    data : dict
        A dictionary containing bootstrap analysis results. The keys are timeslice cycle numbers (Bcycle), and the
        values are dictionaries with the following structure:
        - 'stats': A dictionary containing statistics for local and global crossing probabilities.
            - For local probabilities: Statistics are stored under ensemble indices, with mean and standard
              deviation for each path type (e.g., "RMR", "RML", "LMR", "LML").
            - For global probabilities: Statistics are stored under keys 'Pmin', 'Pplus', and 'Pcross'.

    Returns
    -------
    stats : dict
        A dictionary organized by timeslice cycle numbers (Bcycle). For each timeslice, the value is a dictionary
        containing:
        - 'ploc': A dictionary with the mean and standard deviation of local crossing probabilities for each
                  ensemble and path type. The keys are tuples of (ensemble index, path type).
        - 'Pmin', 'Pplus', 'Pcross': Dictionaries with the mean and standard deviation of global crossing
                                      probabilities. The standard deviation is set to 0 for global probabilities
                                      if not explicitly calculated.

    Notes
    -----
    - Local crossing probabilities are specific to each ensemble and path type.
    - Global crossing probabilities (Pmin, Pplus, Pcross) are aggregated across all ensembles.
    - The standard deviation for global probabilities is set to 0 by default, as it is not explicitly calculated
      in the input data.
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
    For each element in array `A`, find the closest number in array `B` that is less than or equal to it.

    This function searches for the nearest value in `B` that is less than or equal to each element in `A`.
    If no such number exists in `B` (i.e., all elements in `B` are greater than the element in `A`),
    the element is discarded.

    Parameters
    ----------
    A : ndarray
        An array of numbers for which the closest numbers in `B` are to be found.
    B : ndarray
        An array of numbers to search for the closest values. This array is sorted in ascending order
        internally before processing.

    Returns
    -------
    C : ndarray
        An array of the closest numbers in `B` that are less than or equal to the corresponding elements
        in `A`. If no valid number is found for an element in `A`, it is excluded from the result.

    Notes
    -----
    - The array `B` is sorted in ascending order before processing.
    - If an element in `A` is smaller than all elements in `B`, it is discarded.
    - The function uses binary search (`bisect_right`) for efficient lookup.

    Examples
    --------
    >>> A = np.array([5, 10, 15, 20])
    >>> B = np.array([3, 7, 12, 18])
    >>> find_closest_number_lte(A, B)
    [3, 7, 12, 18]  # Closest numbers <= [5, 10, 15, 20] in B
    """

    B = np.sort(B)  # Sort B in ascending order
    C = []
    for _, a in enumerate(A):
        j = bisect.bisect_right(B, a)  # Find idx where a is to be inserted in B
        if j != 0:  # if zero, then a is smaller than all elements in B: discard
            # closest number <= a is the one just before the index j
            C.append(B[j-1])
    return C

def inefficiency(corr):
    """
    Calculates the integrated autocorrelation time (τ_int) of a time series.

    The integrated autocorrelation time quantifies the correlation between 
    successive samples in a time series.  A higher τ_int indicates stronger 
    autocorrelation and a longer time required for the time series to become 
    effectively decorrelated.

    Parameters
    ----------
    corr : array-like
        The autocorrelation function of the time series.  This is typically 
        calculated using a method like the  `autocorrelation` function from 
        the `scipy.signal` module.

    Returns
    -------
    tau_int : float
        The integrated autocorrelation time.  This value represents the 
        effective number of independent samples in the time series.

    Notes
    -----
    The calculation terminates when the autocorrelation function becomes 
    negative, as this indicates the time series has effectively decorrelated.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.signal import autocorrelation
    >>> time_series = np.random.randn(100)
    >>> corr = autocorrelation(time_series)
    >>> inefficiency(corr) 
    # Output:  A float representing the τ_int value
    """
    tau_int = 0
    for i in range(1, len(corr)):
        if corr[i] < 0:  # negative correlation, so time series is decorrelated
            break
        tau_int += corr[i]
    return 1 + 2 * tau_int

def block(x, block_size):
    """
    Calculates the mean and standard error of the mean of a time series using blocking.

    This method divides the time series into blocks of equal size and then 
    calculates the mean and standard error of the mean of the blocks. This 
    approach helps account for autocorrelation in the time series, providing 
    more accurate estimates of the mean and its uncertainty.

    Parameters
    ----------
    x : array-like
        The time series data.
    block_size : int
        The number of samples in each block.

    Returns
    -------
    block_size : int
        The number of samples in each block.
    tau_int : float
        The integrated autocorrelation time of the block means.
    num_ind : int
        The effective number of independent samples in the time series.
    mean : float
        The mean of the time series.
    se_mean : float
        The standard error of the mean of the time series.

    Notes
    -----
    - The time series is trimmed to ensure it contains an integer number of blocks.
    - The autocorrelation function of the block means is calculated and used to 
      determine the integrated autocorrelation time (τ_int).
    - The effective number of independent samples (num_ind) is calculated 
      based on the block size and τ_int.

    Examples
    --------
    >>> import numpy as np
    >>> time_series = np.random.randn(100)
    >>> block_size = 10
    >>> block_size, tau_int, num_ind, mean, se_mean = block(time_series, block_size)
    >>> print(f"τ_int: {tau_int:.2f}, num_ind: {num_ind:.2f}")
    # Output:  τ_int: ..., num_ind: ...
    """
    # Determine the number of blocks that can be formed
    num_blocks = np.floor(len(x) / block_size).astype(int)
    # Trim x to have a multiple of block_size
    x_block = x[:num_blocks * block_size]
    # Reshape x to a matrix of blocks
    x_block = x_block.reshape(num_blocks, block_size)
    # Calculate the mean of each block
    mean_block = np.mean(x_block, axis=1)
    # Calculate the mean and standard error of the means of the blocks
    mean = np.mean(mean_block)
    se_mean = np.std(mean_block, ddof=1) / np.sqrt(num_blocks - 1)
    # Calculate the autocorrelation function of the block means
    corr = np.correlate(mean_block - mean, mean_block - mean, mode='full')
    # Trim the autocorrelation function to remove the negative lags
    corr = corr[corr.size // 2:]
    # Normalize the autocorrelation function (corr = 1 at lag 0)
    corr = np.nan_to_num(corr/corr[0])
    # Calculate the integrated autocorrelation time of the block means
    tau_int = inefficiency(corr)
    # Calculate the effective sample size of the blocked data
    num_ind = np.nan_to_num(np.floor( \
        num_blocks * block_size / tau_int).astype(int))
    return (block_size, tau_int, num_ind, mean, se_mean)


def full_block_analysis(x):
    """
    Performs a comprehensive block analysis of a time series to determine 
    the optimal block size for estimating the mean and standard error of the 
    mean.

    This function systematically explores a range of block sizes, applying the 
    `block` function to each size. It then identifies the block size that 
    minimizes the integrated autocorrelation time (τ_int), indicating the 
    most effective balance between reducing autocorrelation and maintaining 
    a sufficient number of blocks for reliable estimates.

    Parameters
    ----------
    x : array-like
        The time series data.

    Returns
    -------
    opt_block_size : int
        The optimal block size that minimizes the integrated autocorrelation 
        time.
    opt_tau_int : float
        The integrated autocorrelation time corresponding to the optimal 
        block size.
    opt_num_ind : int
        The effective number of independent samples corresponding to the 
        optimal block size.
    opt_mean : float
        The mean of the time series calculated using the optimal block size.
    opt_se_mean : float
        The standard error of the mean calculated using the optimal block size.
    block_sizes : array-like
        An array containing all the block sizes considered during the analysis.
    inefficiencies : array-like
        An array containing the integrated autocorrelation times (τ_int) 
        for each block size.
    se_means : array-like
        An array containing the standard errors of the mean for each block size.

    Notes
    -----
    - The range of block sizes considered is from 1 to the minimum of 1000 
      or half the length of the time series.
    - The optimal block size is determined by finding the block size that 
      minimizes the integrated autocorrelation time.

    Examples
    --------
    >>> import numpy as np
    >>> time_series = np.random.randn(100)
    >>> (opt_block_size, opt_tau_int, opt_num_ind, opt_mean, opt_se_mean, 
    ...  block_sizes, inefficiencies, se_means) = full_block_analysis(time_series)
    >>> print(f"Optimal block size: {opt_block_size}")
    # Output:  Optimal block size: ...
    """

    # Set the range of block sizes to consider
    min_block_size = 1
    max_block_size = min(1000, len(x) // 2)
    results = []
    # Loop over block sizes and perform blocking for each size
    for block_size in range(min_block_size, max_block_size):
        result = block(x, block_size)
        results.append(result)
    results = np.array(results)
    # Extract relevant statistics from the results
    block_sizes = results[:, 0]
    inefficiencies = results[:, 1]
    se_means = results[:, 4]
    # Determine the optimal block size based on integrated autocorrelation time
    min_idx = np.argmin(inefficiencies)
    opt_block_size, opt_tau_int, opt_num_ind, opt_mean, opt_se_mean = \
        results[min_idx]
    # Return relevant statistics and results for all block sizes
    return (opt_block_size, opt_tau_int, opt_num_ind, opt_mean, opt_se_mean, 
            block_sizes, inefficiencies, se_means)

def running_average(x):
    """
    Calculates the running average of a time series.

    This function computes the cumulative average of the time series up to each 
    point in time. This can be helpful for identifying trends and smoothing out 
    fluctuations in the data.

    Parameters
    ----------
    x : array-like
        The time series data.

    Returns
    -------
    x_avg : array-like
        The running average of the time series, where each element represents 
        the average of all values up to and including that point in time.

    Examples
    --------
    >>> import numpy as np
    >>> time_series = np.array([1, 2, 3, 4, 5])
    >>> running_average(time_series)
    array([1. , 1.5, 2. , 2.5, 3. ])
    """
    cumsum = np.cumsum(x)
    x_avg = cumsum / np.arange(1, len(x)+1)
    return x_avg


def unwrap_by_weight(v, w=None):
    """Unwraps a vector by repeating elements based on weights.

    This function takes a vector `v` containing real numbers and a weight vector 
    `w` containing integer weights. It returns an unwrapped vector `u` where 
    elements in `v` are repeated according to their corresponding weights in `w`. 
    If a weight in `w` is zero, the corresponding element in `v` is replaced by 
    the previous non-zero-weighted element.

    Parameters
    ----------
    v : array-like
        The input vector containing real numbers.
    w : array-like, optional
        The weight vector containing integer weights. If not provided, all 
        elements in `v` are assumed to have a weight of 1.

    Returns
    -------
    u : array-like
        The unwrapped vector where elements in `v` are repeated according to 
        their weights in `w`. Zero-weight elements are replaced by the previous 
        non-zero-weighted element.

    Examples
    --------
    >>> import numpy as np
    >>> v = np.array([1, 2, 3])
    >>> w = np.array([2, 0, 1])
    >>> unwrap_by_weight(v, w)
    array([1, 1, 2, 3])
    """

    return np.repeat(v, w)

def running_avg_local_probs(pathtype_cycles, w=None, tr=False):
    """Calculates running averages of local probabilities for different path types.

    This function takes a dictionary of path type cycles, optional weights, and a 
    boolean flag `tr` to calculate running averages of local probabilities for 
    different path types. 

    Parameters
    ----------
    pathtype_cycles : dict of arrays
        A dictionary where keys are path types (e.g., 'LMR', 'LML') and values 
        are arrays of 0/1 indicating the presence of each path type in each 
        cycle.
    w : array-like, optional
        The weights of the paths. If not provided, all paths are assumed to have 
        equal weight (1).
    tr : bool, optional
        If True, applies a specific transformation to the 'LML', 'RMR', 'LMR', 
        and 'RML' path types. Defaults to False.

    Returns
    -------
    p_NN, p_NP, p_PN, p_PP : arrays
        Arrays representing the running averages of local probabilities for 
        different path types:
            - p_NN: Probability of 'LML' path type.
            - p_NP: Probability of 'LMR' path type.
            - p_PN: Probability of 'RML' path type.
            - p_PP: Probability of 'RMR' path type.

    Examples
    --------
    >>> import numpy as np
    >>> pathtype_cycles = {'LMR': np.array([1, 0, 1]), 
    ...                    'LML': np.array([0, 1, 0]),
    ...                    'RMR': np.array([0, 1, 1]),
    ...                    'RML': np.array([1, 0, 0])}
    >>> # Example with weights
    >>> w = np.array([1, 1, 1])
    >>> p_NN, p_NP, p_PN, p_PP = running_avg_local_probs(pathtype_cycles, w)
    >>> print(p_NN)
    [0.  1.  0.5]
    >>> # Example without weights (default)
    >>> p_NN, p_NP, p_PN, p_PP = running_avg_local_probs(pathtype_cycles)
    >>> print(p_NN)
    [0.  1.  0.5]
    """
    cumsums = {}
    for key in pathtype_cycles.keys():
        if w is None:
            cumsums[key] = np.cumsum(pathtype_cycles[key])  # Default weight 1
        else:
            cumsums[key] = np.cumsum(pathtype_cycles[key] * w)
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
    """
    Calculate the distribution of lambda values for LMR and RML paths.

    This function analyzes the distribution of lambda_max values for LM* paths 
    (LML and LMR) and the distribution of lambda_min values for RM* paths 
    (RMR and RML) within a PPTIS path ensemble.

    Parameters
    ----------
    pe : :py:class:`.PathEnsemble`
        A PPTIS path ensemble object (tistools-processed).

    Returns
    -------
    L : float
        The left interface of the path ensemble.
    M : float
        The middle interface of the path ensemble.
    R : float
        The right interface of the path ensemble.
    percents : array of floats
        The distribution of lambda_max values for LM* paths, given at lambda 
        values 'lambs' between the middle and right interfaces.
    lambs : array of floats
        The lambda values at which the lambda_max distribution is given.
    percents2 : array of floats
        The distribution of lambda_min values for RM* paths, given at lambda 
        values 'lambs2' between the left and middle interfaces.
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
    Calculate pathlength distributions for different path types in a unified path ensemble.

    This function analyzes the pathlengths of each path type (LML, LMR, RML, RMR) 
    within a unified path ensemble, providing mean, standard deviation, and a histogram 
    of the distribution.

    Parameters
    ----------
    upe : :py:class:`.PathEnsemble`
        A unified path ensemble object.

    Returns
    -------
    data : dict
        A dictionary containing the pathlength distributions for each path type.
        Each path type (e.g., "LML") has the following keys:
            - "mean": Mean pathlength for the path type.
            - "std": Standard deviation of pathlengths for the path type.
            - "hist": Histogram of pathlengths for the path type.
            - "bin_centers": Centers of the histogram bins.

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

        
def get_global_probs_from_dict(ensemble_probabilities):
    """Calculates the global crossing probabilities for the PPTIS simulation.

    This function computes the global crossing probabilities using the recursive relations:
    - P_plus[j] = (pLMR[j-1] * P_plus[j-1]) / (pLMR[j-1] + pLML[j-1] * P_min[j-1])
    - P_min[j] = (pRML[j-1] * P_min[j-1]) / (pLMR[j-1] + pLML[j-1] * P_min[j-1])
    where P_plus[0] = 1, P_min[0] = 1, and j is the ensemble number.

    Parameters
    ----------
    ensemble_probabilities : list of dicts
        A list of dictionaries containing the local crossing probabilities for each PPTIS ensemble.
        Each dictionary has keys representing the path types (RMR, RML, LMR, LML).
        For example, ensemble_probabilities[i]['LMR']['mean'] and ensemble_probabilities[i]['LMR']['std']
        represent the mean and standard deviation of p_{i}^{mp}, respectively.

    Returns
    -------
    P_min : list of floats
        A list of probabilities, one per ensemble i, representing the probability of crossing A earlier
        than i+1, given that the path crossed i.
    P_plus : list of floats
        A list of probabilities, one per ensemble i, representing the probability of crossing i+1 earlier
        than A, given that the path crossed i.
    P_cross : list of floats
        A list of probabilities, one per ensemble i, representing the TIS probability of crossing i+1.
    """

    P_plus, P_min, P_cross = [1.0], [1.0], [1.0, ensemble_probabilities[1]['LMR']]
    for i, probabilities in enumerate(ensemble_probabilities):
        if i <= 1:  # Skip the first two ensembles as they are initial conditions
            continue
        # Calculate the global crossing probabilities
        P_plus.append((probabilities['LMR'] * P_plus[-1]) / (probabilities['LMR'] + probabilities['LML'] * P_min[-1]))
        P_min.append((probabilities['RML'] * P_min[-1]) / (probabilities['LMR'] + probabilities['LML'] * P_min[-1]))
        # Calculate the TIS probabilities
        P_cross.append(ensemble_probabilities[1]['LMR'] * P_plus[-1])
    return P_min, P_plus, P_cross

def get_global_probs_from_local(p_minus_plus, p_minus_minus, p_plus_plus, p_plus_minus):  # was: get_global_probz
    """Computes the global crossing probabilities for the PPTIS simulation from local probabilities.

    This function follows the recursive relations:
    - P_plus[j] = (pLMR[j-1] * P_plus[j-1]) / (pLMR[j-1] + pLML[j-1] * P_min[j-1])
    - P_min[j] = (pRML[j-1] * P_min[j-1]) / (pLMR[j-1] + pLML[j-1] * P_min[j-1])
    where P_plus[0] = 1, P_min[0] = 1, and j is the ensemble number.

    Parameters
    ----------
    p_minus_plus : list of floats
        The local probability of having type LMR (Left-to-Right crossing) for each ensemble.
    p_minus_minus : list of floats
        The local probability of having type LML (Left-to-Left crossing) for each ensemble.
    p_plus_plus : list of floats
        The local probability of having type RMR (Right-to-Right crossing) for each ensemble.
    p_plus_minus : list of floats
        The local probability of having type RML (Right-to-Left crossing) for each ensemble.

    Returns
    -------
    P_min : list of floats
        A list of probabilities, one per ensemble i, representing the probability of crossing A earlier
        than i+1, given that the path crossed i.
    P_plus : list of floats
        A list of probabilities, one per ensemble i, representing the probability of crossing i+1 earlier
        than A, given that the path crossed i.
    P_cross : list of floats
        A list of probabilities, one per ensemble i, representing the TIS probability of crossing i+1.

    Notes
    -----
    If any input probability list contains NaN values, the function returns [NaN, NaN, NaN].
    """

    # Check for NaN values in input probabilities
    if np.isnan(p_minus_plus).any() or np.isnan(p_minus_minus).any() or \
       np.isnan(p_plus_plus).any() or np.isnan(p_plus_minus).any():
        return [np.nan, np.nan, np.nan]

    P_plus, P_min, P_cross = [1.0], [1.0], [1.0, p_minus_plus[0]]
    for i, (pmp, pmm, _, ppm) in enumerate(zip(p_minus_plus, p_minus_minus, p_plus_plus, p_plus_minus)):
        if i == 0:  # Skip the first ensemble as it is an initial condition
            continue
        # Check for division by zero or NaN in calculations
        if pmp + pmm * P_min[-1] == 0 or np.isnan(pmp) or np.isnan(pmm * P_min[-1]):
            return [np.nan, np.nan, np.nan]
        # Calculate the global crossing probabilities
        P_plus.append((pmp * P_plus[-1]) / (pmp + pmm * P_min[-1]))
        P_min.append((ppm * P_min[-1]) / (pmp + pmm * P_min[-1]))
        # Calculate the TIS probabilities
        P_cross.append(p_minus_plus[0] * P_plus[-1])
    return P_min, P_plus, P_cross


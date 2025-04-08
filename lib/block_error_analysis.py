"""Script to read and loop the cycles and do recursive block error analysis
"""

import os
import glob
import warnings
import numpy as np
import datetime

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Reading
from .reading import read_inputfile, get_LMR_interfaces, read_pathensemble, get_weights
from .repptis_pathlengths import set_taus, collect_tau, collect_tau1, collect_tau2
from .repptis_analysis import ACCFLAGS, REJFLAGS

# REPPTIS analysis
from .repptis_analysis import get_local_probs, get_global_probs_from_dict

# MSM functions
from .repptis_msm import construct_M, construct_M_N3, global_pcross_msm
from .repptis_msm import mfpt_to_first_last_state, construct_tau_vector
from .istar_analysis import construct_M_istar, compute_weight_matrices, compute_weight_matrix, get_transition_probs_weights, global_pcross_msm_star

# Writing output
from .writing import write_running_estimates, write_plot_block_error


def recursive_blocker(k):
    """
    Computes block averages using a recursive blocking formula.

    This function calculates block averages based on the method described in 
    Vervust et al. (PyRETIS3, 2024). The formula used is:

        k_r[j] = (j + 1) * k[j] - j * k[j - 1]

    which accounts for Python's zero-based indexing.

    Parameters
    ----------
    k : list or np.ndarray
        Sequence of values for which block averaging is computed.

    Returns
    -------
    k_r : list
        List of computed block-averaged values.
    """
    k_r = [k[0]]  # Initialize with the first element
    k = np.array(k)

    for j in range(1, len(k)):
        k_r.append((j + 1) * k[j] - j * k[j - 1])

    return k_r


def load_txt_data(filename):
    """
    Reads a text file containing three columns (cycle, Tau, Pcross)
    and loads the data into three separate numpy arrays.

    Parameters
    ----------
    filename : str
        Name or path of the text file to be loaded.

    Returns
    -------
    cycles : np.ndarray
        Array of cycle numbers.
        
    tau : np.ndarray
        Array of Tau values (NaNs are preserved).
        
    pcross : np.ndarray
        Array of Pcross values.
    """
    data = np.genfromtxt(filename, skip_header=1, dtype=float, names=["cycle", "tau", "pcross"])
    cycles = data["cycle"]
    tau = data["tau"]
    pcross = data["pcross"]
    
    return cycles, tau, pcross


def block_error_analysis(path_ensembles, interfaces, interval, load=False):
    """
    Conducts block error analysis on Tau and Pcross values.

    This function loads precomputed Tau and Pcross running estimates from a txt file if available 
    and valid. Otherwise, it recalculates them using the `calculate_running_estimate` function.
    It then performs block error analysis on the obtained values.

    Parameters
    ----------
    path_ensembles : :py:class:`.PathEnsemble`  
        The loaded path ensemble object.  

    interfaces : list of float  
        A list of interface positions extracted from the data.  

    interval : int
        The interval used for running estimates.

    load : bool, optional (default=False)
        Whether to load data from the txt file if available.

    Returns
    -------
    None
        The function performs block error analysis but does not return a value.
    """

    filename = f"pcross_tau_interval_{interval}.txt"

    if load and os.path.exists(filename):
        # Attempt to load data from the file
        print("The data file exists, reading...")
        _, taus, pcross = load_txt_data(filename)

        # Validate the loaded data: check for empty values or NaNs
        # if taus is None or pcross is None or np.isnan(taus).any() or np.isnan(pcross).any():
        if taus is None or pcross is None:
            print("Invalid data in file, recalculating...")

            # If data is invalid, recalculate running estimates
            _, taus, pcross, _, _, _, _, _ = calculate_running_estimate(path_ensembles, interfaces, interval)
    else:
        # If loading is disabled or the file doesn't exist, calculate running estimates
        print("First time calculating the data file ...")
        _, taus, pcross, _, _, _, _, _ = calculate_running_estimate(path_ensembles, interfaces, interval)

    block_error_calculation(taus, interval, "Tau")
    block_error_calculation(pcross, interval, "Pcross")

def block_error_analysis_staple(path_ensembles, interfaces, interval, load=False, pl=True):
    """
    Conducts block error analysis on Tau and Pcross values.

    This function loads precomputed Tau and Pcross running estimates from a txt file if available 
    and valid. Otherwise, it recalculates them using the `calculate_running_estimate` function.
    It then performs block error analysis on the obtained values.

    Parameters
    ----------
    path_ensembles : :py:class:`.PathEnsemble`  
        The loaded path ensemble object.  

    interfaces : list of float  
        A list of interface positions extracted from the data.  

    interval : int
        The interval used for running estimates.

    load : bool, optional (default=False)
        Whether to load data from the txt file if available.

    Returns
    -------
    None
        The function performs block error analysis but does not return a value.
    """

    filename = f"pcross_tau_interval_{interval}.txt"

    if load and os.path.exists(filename):
        # Attempt to load data from the file
        print("The data file exists, reading...")
        _, taus, pcross = load_txt_data(filename)

        # Validate the loaded data: check for empty values or NaNs
        # if taus is None or pcross is None or np.isnan(taus).any() or np.isnan(pcross).any():
        if taus is None or pcross is None:
            print("Invalid data in file, recalculating...")

            # If data is invalid, recalculate running estimates
            _, _, p_staple, pmms, pmps, ppms, ppps, Pcrossfulls_repptis, pcrepptis_MSM, pcstaple_MSM = calculate_running_estimate_staple(path_ensembles, interfaces, interval, pl=pl)
    else:
        # If loading is disabled or the file doesn't exist, calculate running estimates
        print("First time calculating the data file ...")
        _, _, p_staple, pmms, pmps, ppms, ppps, Pcrossfulls_repptis, pcrepptis_MSM, pcstaple_MSM = calculate_running_estimate_staple(path_ensembles, interfaces, interval, pl=pl)

    block_error_calculation(np.array(Pcrossfulls_repptis)[:,-1], interval, "Pcross_repptis")
    block_error_calculation(pcrepptis_MSM, interval, "Pcross_repp_MSM")
    block_error_calculation(np.array(pcstaple_MSM)[:, -1], interval, "Pcross")


def block_error_calculation(running_estimate, interval, label):
    """
    Performs block error analysis on a running estimate.

    This method applies block averaging to reduce statistical correlations 
    and estimate the relative error of the final running estimate.

    Parameters
    ----------
    running_estimate : list or np.ndarray
        A sequence of running estimates for which the error is analyzed.

    Returns
    -------
    rel_errors : list of float
        A list of relative errors corresponding to different block lengths.
    """

    min_block_number = 5  # Hard-coded based on Titus' code
    max_block_length = len(running_estimate) // min_block_number  # Determine max block size
    best_avg = running_estimate[-1]  # Use the last value as the best available average

    rel_errors = []
    
    for block_length in range(1, max_block_length + 1):
        # Extract every `block_length`-th element to form blocks
        blocked_estimate = running_estimate[block_length::block_length]  
        
        # Perform recursive blocking to obtain block-averaged values
        blocked_values = recursive_blocker(blocked_estimate)  

        # Compute absolute and relative errors
        abs_err = np.sqrt(np.var(blocked_values, ddof=1) / len(blocked_values))  
        rel_err = abs_err / best_avg  
        
        rel_errors.append(rel_err)

    write_plot_block_error(f"block_error_analysis_{label}_{interval}", running_estimate, rel_errors, interval)

    return rel_errors


def shallow_copy(obj):
    """
    Creates a shallow copy of an object by duplicating its attributes.

    Parameters
    ----------
    obj : object
        The object to be copied.

    Returns
    -------
    new_obj : object
        A new instance of the same type with a shallow copy of attributes.
    """
    new_obj = type(obj)()
    
    if hasattr(obj, '__dict__'):
        new_obj.__dict__ = obj.__dict__.copy()
    else:
        raise TypeError("Object does not support attribute dictionaries.")
    
    return new_obj


def pathensembles_nskip(obj, nskip):
    """
    Truncates specified attributes of a path ensemble object.

    This function modifies the given object by retaining only the first `nskip` 
    elements of certain key attributes.

    Parameters
    ----------
    obj : object
        The path ensemble object with attributes to be truncated.

    nskip : int
        The number of elements to retain for each attribute.

    Returns
    -------
    None
        The function modifies the object in place.
    """
    keys = ['cyclenumbers', 'flags', 'generation', 'lengths', 'lmrs', 'weights', 'dirs', 'lambmaxs', 'lambmins']
    
    for key in keys:
        if hasattr(obj, key):
            attr = getattr(obj, key)
            setattr(obj, key, attr[:nskip])
        else:
            print(f"WARNING: Object does not have attribute '{key}'.")


def load_path_ensembles(indir, load=False):
    """
    Loads and processes path ensembles from a specified directory.

    This function retrieves path ensemble data and interface definitions  
    from the given working directory, preparing them for further analysis.  
    It can either load pre-existing `.npy` files or generate them if they do not exist.

    Parameters
    ----------
    indir : str  
        Path to the working directory containing path ensemble data.  

    load : bool  
        If True, loads previously created `.npy` files.  
        If False, generates new `.npy` files, which will take longer.  

    Returns
    -------
    pathensembles_original : :py:class:`.PathEnsemble`  
        The loaded path ensemble object.  

    interfaces : list of float  
        A list of interface positions extracted from the data.  
    """
    # Set the working directory
    inputfile = f"{indir}/repptis.rst"
    os.chdir(indir)

    # Get sorted list of folders (excluding index 0)
    folders = sorted(glob.glob(f"{indir}/0[0-9][0-9]"))

    # Read input data
    interfaces, zero_left, _ = read_inputfile(inputfile)
    LMR_interfaces, LMR_strings = get_LMR_interfaces(interfaces, zero_left)

    # Initialize path ensembles
    pathensembles_original = []
    for i, folder in enumerate(folders):
        pe = read_pathensemble(f"{folder}/pathensemble.txt")
        pe.set_name(folder)
        pe.set_interfaces([LMR_interfaces[i], LMR_strings[i]])
        pe.set_in_zero_minus(i == 0)
        pe.set_in_zero_plus(i == 1)
        
        # Set weights and orders
        weights, _ = get_weights(pe.flags, ACCFLAGS, REJFLAGS, verbose=False)
        pe.set_weights(weights)
        if load == True:
            pe.set_orders(load=True, acc_only=True)
        else:
            pe.set_orders(load=False, acc_only=True, save=True)

        pathensembles_original.append(pe)
    
    return pathensembles_original, interfaces


def calculate_running_estimate(pathensembles_original, interfaces, interval=1):
    """
    Computes running estimates of key parameters from path ensembles.

    This function iterates over the cycle number, progressively truncating the 
    `pathensembles_original` object up to the given cycle number. At each step, 
    it calculates various parameters, including probabilities and time constants (taus). 
    
    Since this process involves recalculating parameters at each cycle, it can be 
    computationally expensive, particularly for large numbers of cycles (e.g., 30,000). 
    The slowest part of the function is `set_taus`. To mitigate this, an interval parameter 
    is introduced, allowing calculations to be performed at specified intervals 
    (e.g., every 10 cycles if `interval=10`), improving efficiency.

    Using intervals greater than 1 will affect the block error analysis since fewer 
    data points are used. However, if the data comes from an MD time series, which 
    typically exhibits high correlation, the difference between using an interval of 
    1 and 10 should be minimal. 

    At the same block length, the exact error value will be preserved. For example, 
    the relative error for an interval of 1 with a block length of 100 will be equal 
    to that of an interval of 10 with a block length of 10.

    Parameters
    ----------
    pathensembles_original : :py:class:`.PathEnsemble`
        The original path ensembles containing trajectory data.

    interfaces : list of float
        A list of interface positions used for computing transition probabilities.

    interval : int, optional, default=1
        The step size for computing running estimates. Setting `interval > 1` reduces 
        the number of computations by skipping intermediate cycles.

    Returns
    -------
    cycles : list of int
        List of cycle numbers processed. Useful when `interval > 1`.

    taus : list of float
        Running estimates of the Tau+ values.

    pcross : list of float
        Running estimates of the crossing probabilities.

    pmms : list of tuple
        Running estimates of P_minus_minus (LML) values.
        Each tuple contains `len(interfaces) - 1` elements.

    pmps : list of tuple
        Running estimates of P_minus_plus (LMR) values.
        Each tuple contains `len(interfaces) - 1` elements.

    ppms : list of tuple
        Running estimates of P_plus_minus (RML) values.
        Each tuple contains `len(interfaces) - 1` elements.

    ppps : list of tuple
        Running estimates of P_plus_plus (RMR) values.
        Each tuple contains `len(interfaces) - 1` elements.

    Pcrossfulls : list of tuple
        Running estimates of the global crossing probabilities for each interface.
        Each tuple contains `len(interfaces)` elements.
    """

    # Prepare result storage
    cycles, taus, pcross = [], [], []
    pmms, pmps, ppms, ppps, Pcrossfulls = [], [], [], [], []
    pathtypes = ("LML", "LMR", "RML", "RMR")
    
    # Iterate over cycle numbers with interval steps
    for nskip in range(1, pathensembles_original[0].cyclenumbers[-1] + interval, interval):
        pathensembles = [shallow_copy(pe) for pe in pathensembles_original]
        data = {i: {} for i in range(len(pathensembles))}

        for i, pe in enumerate(pathensembles):
            pathensembles_nskip(pe, nskip)
            set_taus(pe)

            # Skip [0-] index as it is not used for Pcross calculations
            # if i == 0:
            #     continue  

            # Compute local probabilities
            plocfull = get_local_probs(pe, tr=False)
            data[i]["full"] = {ptype: plocfull[ptype] for ptype in pathtypes}

        # Extract global probabilities
        psfull = [
            {ptype: data[i]["full"][ptype] for ptype in ("LMR", "RML", "RMR", "LML")}
            for i in range(len(pathensembles))
        ]
        # _, _, Pcrossfull = get_globall_probs(psfull)
        _, _, Pcrossfull = get_global_probs_from_dict(psfull)

        # Extract probability distributions
        pmp, pmm, ppp, ppm = zip(*[
            (data[i]["full"]["LMR"], data[i]["full"]["LML"],
             data[i]["full"]["RMR"], data[i]["full"]["RML"])
            for i in range(1, len(pathensembles))
        ])

        # Construct transition matrix M
        N, NS = len(interfaces), 4 * len(interfaces) - 5
        if N > 3:
            M = construct_M(pmm, pmp, ppm, ppp, N)
        else:
            M = construct_M_N3(pmm, pmp, ppm, ppp, N)

        # Collect tau values
        tau_mm, tau_mp, tau_pm, tau_pp = collect_tau(pathensembles)
        tau1_mm, tau1_mp, tau1_pm, tau1_pp = collect_tau1(pathensembles)
        tau2_mm, tau2_mp, tau2_pm, tau2_pp = collect_tau2(pathensembles)

        # Construct tau vectors
        tau = construct_tau_vector(N, NS, tau_mm, tau_mp, tau_pm, tau_pp)
        tau1 = construct_tau_vector(N, NS, tau1_mm, tau1_mp, tau1_pm, tau1_pp)
        tau2 = construct_tau_vector(N, NS, tau2_mm, tau2_mp, tau2_pm, tau2_pp)
        tau_m = tau - tau1 - tau2

        # Compute Mean First Passage Time (MFPT) and global cross probability
        _, _, h1, _ = mfpt_to_first_last_state(M, tau1, tau_m, tau2)
        _, _, y1, _ = global_pcross_msm(M)

        # Print cycle information
        print(f"{nskip:5d} cycles, tau {h1[0][0]:.8f}, Pcross {y1[0][0]:.8f}")

        # Store results
        cycles.append(nskip)
        taus.append(h1[0][0])
        pcross.append(y1[0][0])
        pmms.append(pmm)
        pmps.append(pmp)
        ppms.append(ppm)
        ppps.append(ppp)
        Pcrossfulls.append(Pcrossfull)

    # Write to file
    write_running_estimates(f"pcross_tau_interval_{interval}.txt", cycles, taus, "Tau", pcross, "Pcross")
    write_running_estimates(f"ploc_interval_{interval}.txt", cycles,
        np.array(pmms), "P_LML",
        np.array(pmps), "P_LMR",
        np.array(ppms), "P_RML",
        np.array(ppps), "P_RMR",
        np.array(Pcrossfulls)[:, 1:], "P_Cross"
    )
    return cycles, taus, pcross, pmms, pmps, ppms, ppps, Pcrossfulls


def calculate_running_estimate_staple(pathensembles_original, interfaces, interval=1, pl=True):
    """
    Computes running estimates of key parameters from path ensembles.

    This function iterates over the cycle number, progressively truncating the 
    `pathensembles_original` object up to the given cycle number. At each step, 
    it calculates various parameters, including probabilities and time constants (taus). 
    
    Since this process involves recalculating parameters at each cycle, it can be 
    computationally expensive, particularly for large numbers of cycles (e.g., 30,000). 
    The slowest part of the function is `set_taus`. To mitigate this, an interval parameter 
    is introduced, allowing calculations to be performed at specified intervals 
    (e.g., every 10 cycles if `interval=10`), improving efficiency.

    Using intervals greater than 1 will affect the block error analysis since fewer 
    data points are used. However, if the data comes from an MD time series, which 
    typically exhibits high correlation, the difference between using an interval of 
    1 and 10 should be minimal. 

    At the same block length, the exact error value will be preserved. For example, 
    the relative error for an interval of 1 with a block length of 100 will be equal 
    to that of an interval of 10 with a block length of 10.

    Parameters
    ----------
    pathensembles_original : :py:class:`.PathEnsemble`
        The original path ensembles containing trajectory data.

    interfaces : list of float
        A list of interface positions used for computing transition probabilities.

    interval : int, optional, default=1
        The step size for computing running estimates. Setting `interval > 1` reduces 
        the number of computations by skipping intermediate cycles.

    Returns
    -------
    cycles : list of int
        List of cycle numbers processed. Useful when `interval > 1`.

    taus : list of float
        Running estimates of the Tau+ values.

    pcross : list of float
        Running estimates of the crossing probabilities.

    pmms : list of tuple
        Running estimates of P_minus_minus (LML) values.
        Each tuple contains `len(interfaces) - 1` elements.

    pmps : list of tuple
        Running estimates of P_minus_plus (LMR) values.
        Each tuple contains `len(interfaces) - 1` elements.

    ppms : list of tuple
        Running estimates of P_plus_minus (RML) values.
        Each tuple contains `len(interfaces) - 1` elements.

    ppps : list of tuple
        Running estimates of P_plus_plus (RMR) values.
        Each tuple contains `len(interfaces) - 1` elements.

    Pcrossfulls : list of tuple
        Running estimates of the global crossing probabilities for each interface.
        Each tuple contains `len(interfaces)` elements.
    """

    # Prepare result storage
    cycles, taus = [], []
    Pcrossfulls_repptis, pcrepptis_MSM, pcstaple_MSM = [], [], []
    pmms, pmps, ppms, ppps, p_staple = [], [], [], [], []
    pathtypes = ("LML", "LMR", "RML", "RMR")
    
    # Iterate over cycle numbers with interval steps
    for nskip in range(1, pathensembles_original[0].cyclenumbers[-1] + interval, interval):
        pathensembles = [shallow_copy(pe) for pe in pathensembles_original]
        data = {i: {} for i in range(len(pathensembles))}

        for i, pe in enumerate(pathensembles):
            pathensembles_nskip(pe, nskip)
            if pl:
                set_taus(pe)

            # Skip [0-] index as it is not used for Pcross calculations
            # if i == 0:
            #     continue  

            # Compute REPPTIS local probabilities
            plocfull = get_local_probs(pe, tr=False)
            data[i]["full"] = {ptype: plocfull[ptype] for ptype in pathtypes}

        # Extract global probabilities
        psfull = [
            {ptype: data[i]["full"][ptype] for ptype in ("LMR", "RML", "RMR", "LML")}
            for i in range(len(pathensembles))
        ]
        _, _, Pcrossfull = get_global_probs_from_dict(psfull)

        # Extract probability distributions
        pmp, pmm, ppp, ppm = zip(*[
            (data[i]["full"]["LMR"], data[i]["full"]["LML"],
             data[i]["full"]["RMR"], data[i]["full"]["RML"])
            for i in range(1, len(pathensembles))
        ])

        # Construct transition matrix M
        N, NS = len(interfaces), 4 * len(interfaces) - 5
        M = construct_M(pmm, pmp, ppm, ppp, N)

        # Collect tau values
        if pl:
            tau_mm, tau_mp, tau_pm, tau_pp = collect_tau(pathensembles)
            tau1_mm, tau1_mp, tau1_pm, tau1_pp = collect_tau1(pathensembles)
            tau2_mm, tau2_mp, tau2_pm, tau2_pp = collect_tau2(pathensembles)

            # Construct tau vectors
            tau = construct_tau_vector(N, NS, tau_mm, tau_mp, tau_pm, tau_pp)
            tau1 = construct_tau_vector(N, NS, tau1_mm, tau1_mp, tau1_pm, tau1_pp)
            tau2 = construct_tau_vector(N, NS, tau2_mm, tau2_mp, tau2_pm, tau2_pp)
            tau_m = tau - tau1 - tau2

            # Compute Mean First Passage Time (MFPT) and global cross probability
            _, _, h1, _ = mfpt_to_first_last_state(M, tau1, tau_m, tau2)

        _, _, y1_repp, _ = global_pcross_msm(M)


        NS = 2*N
        wi = compute_weight_matrices(pathensembles, interfaces, len(interfaces), True)
        pi = get_transition_probs_weights(wi)
        M = construct_M_istar(pi, max(4, 2*len(interfaces)), len(interfaces))

        plocMSM = np.ones(len(interfaces))
        for lint in range(1, len(interfaces)):
            Mi = M[np.r_[0:2+lint, 1+N:N+lint, -1]]
            Mi = Mi[:, np.r_[0:2+lint, 1+N:N+lint+1]]
            Msum = M[np.r_[0:2+lint, 1+N:N+lint+1]]
            Mi[:, -1] = np.sum(Msum[:, N+lint:], axis=1)

            z1, z2, y1, y2 = global_pcross_msm_star(Mi)
            plocMSM[lint] = y1[0][0]

        pcrosslocMSM = np.empty(len(plocMSM))

        for i in range (len(pcrosslocMSM)):
            pcrosslocMSM[i] = plocMSM[i]
            # pcrosslocMSM[i] = plocMSM[i]/np.prod(pcrosslocMSM[:i])
        # print(pcrosslocMSM)
        # Compute taus for pathlength analysis
        # timestamp = datetime.now().strftime("%H:%M:%S")
        print("\n" + 15*"-")
        print(f"{nskip:5d} cycles, Plocs {plocMSM}, Pcross {y1[0][0]:.8f}" + (f"tau {h1[0][0]:.8f}" if pl else ""))
        print(15*"-" + "\n")

        # Store results
        cycles.append(nskip)
        if pl:
            taus.append(h1[0][0])
        pcrepptis_MSM.append(y1_repp[0][0])
        pcstaple_MSM.append(pcrosslocMSM)
        p_staple.append(pi)
        pmms.append(pmm)
        pmps.append(pmp)
        ppms.append(ppm)
        ppps.append(ppp)
        Pcrossfulls_repptis.append(Pcrossfull)

        # Not sure if we need this, need to check later
        del data
        del Mi, wi, pi, z1, z2, y1, y2,N, NS
        del Pcrossfull
        del plocMSM, pcrosslocMSM

    # repptisploc = []
    # for i, pe in enumerate(pathensembles_original):
    #     # REPPTIS p_loc
    #     repptisploc.append(get_local_probs(pe, tr=False))

    # _, _, reppfull = get_global_probs_from_dict(repptisploc)
    
    # # Write REPPTIS results to log file
    # print("\ninfunc:REPPTIS global crossing probabilities:")
    # for i, p in enumerate(reppfull):
    #     print(f"Interface {i}: {p:.8f}")

    # repptisploc = []
    # for i, pe in enumerate(pathensembles):
    #     # REPPTIS p_loc
    #     repptisploc.append(get_local_probs(pe, tr=False))

    # _, _, reppfull = get_global_probs_from_dict(repptisploc)
    
    # # Write REPPTIS results to log file
    # print("\ninfunc2:REPPTIS global crossing probabilities:")
    # for i, p in enumerate(reppfull):
    #     print(f"Interface {i}: {p:.8f}")


    # Write to file
    if pl:
        write_running_estimates(f"pcross_tau_interval_{interval}.txt", cycles, taus, "Tau", pcstaple_MSM, "Pcross")
    else:
        write_running_estimates(f"pcross_tau_interval_{interval}.txt", cycles, np.array(pcstaple_MSM)[:, 1:], "Pcross")
    write_running_estimates(f"ploc_interval_{interval}.txt", cycles,
        np.array(pmms), "P_LML",
        np.array(pmps), "P_LMR",
        np.array(ppms), "P_RML",
        np.array(ppps), "P_RMR",
        # np.array(p_staple), "Pi_staple",
        np.array(Pcrossfulls_repptis)[:, 1:], "P_Cross_native",
        np.array(pcrepptis_MSM), "P_cross_REPPTISMSM"
    )
    return cycles, taus, p_staple, pmms, pmps, ppms, ppps, Pcrossfulls_repptis, pcrepptis_MSM, pcstaple_MSM
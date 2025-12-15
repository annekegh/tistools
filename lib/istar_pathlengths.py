"""
Path length analysis module for interface-based Markov State Models (iSTAR).

This module provides functions to analyze path lengths in transition interface sampling
using the iSTAR approach, which constructs Markov state models at the interfaces and
analyzes transitions between them based on turn points at interfaces.
"""

import numpy as np
import logging
from .reading import set_flags_ACC_REJ

# Hard-coded rejection flags found in output files
ACCFLAGS, REJFLAGS = set_flags_ACC_REJ() 

def construct_tau_vector(N, NS, taumm, taump, taupm, taupp):
    """
    Construct a tau vector from given path lengths for iSTAR analysis.

    In the iSTAR approach, path lengths are categorized by the direction of turns
    at interfaces. This function constructs a unified tau vector from path length
    components for different turn types.

    Parameters
    ----------
    N : int
        Number of interfaces.
    NS : int
        Number of state segments in the iSTAR model.
    taumm, taump, taupm, taupp : list of float
        Path lengths for different turn type combinations:
        - taumm: minus to minus (backward to backward)
        - taump: minus to plus (backward to forward)
        - taupm: plus to minus (forward to backward)
        - taupp: plus to plus (forward to forward)

    Returns
    -------
    np.ndarray
        Constructed tau vector for the iSTAR model.

    Raises
    ------
    ValueError
        If the input lengths do not match the expected values.
    """
    if N < 3:
        raise ValueError("N must be at least 3.")
    if NS != 4 * N - 5:
        raise ValueError(f"NS must be 4 * N - 5 = {4 * N - 5}.")
    if len(taumm) != N or len(taump) != N or len(taupm) != N or len(taupp) != N:
        raise ValueError("Input lengths must match the expected values.")

    # Unravel the values into one vector
    tau = np.zeros(NS)
    # [0-]
    tau[0] = taupp[0]
    # [0+-]
    tau[1] = taumm[1]
    tau[2] = taump[1]
    tau[3] = taupm[1]
    # [1+-] etc
    for i in range(1, N - 2):
        tau[4 * i] = taumm[i + 1]
        tau[4 * i + 1] = taump[i + 1]
        tau[4 * i + 2] = taupm[i + 1]
        tau[4 * i + 3] = taupp[i + 1]
    # [(N-2)^(-1)]
    tau[-3] = taumm[-1]
    tau[-2] = taump[-1]
    # B
    tau[-1] = 0.  # whatever
    return tau

def set_tau_first_hit_interface_distrib(pe, do_last=True):
    """
    Set the average path length before the next interface is reached for each path type.

    In iSTAR, we track the time to reach the next interface (first hit) from the 
    current interface, categorized by path direction types.

    Parameters
    ----------
    pe : :py:class:`.PathEnsemble`
        The path ensemble object.
    do_last : bool, optional
        If True, also compute tau2 (time after last interface crossing). Default is True.

    Returns
    -------
    None
    """
    pe.tau1 = []
    pe.tau1avg = {"LML": None, "LMR": None, "RML": None, "RMR": None}
    # Select the accepted paths
    for i in range(len(pe.cyclenumbers)):
        if pe.flags[i] != "ACC" or pe.generation[i] == "ld":
            pe.tau1.append(0)
            continue 
        pe.tau1.append(get_tau1_path(pe.orders[i], pe.lmrs[i], pe.interfaces[0]))
    pe.tau1 = np.array(pe.tau1) 

    # Get the average tau1 for each path type. Each path has a weight w.
    for ptype in ("LML", "LMR", "RML", "RMR"):
        totweight = np.sum(pe.weights[pe.lmrs == ptype])
        if totweight != 0:  # Make sure total weight is not zero
            pe.tau1avg[ptype] = np.average(pe.tau1[pe.lmrs == ptype], weights=pe.weights[pe.lmrs == ptype])

    if not do_last:
        return

    pe.tau2 = []
    pe.tau2avg = {"LML": None, "LMR": None, "RML": None, "RMR": None}
    # Select the accepted paths
    for i in range(len(pe.cyclenumbers)):
        if pe.flags[i] != "ACC" or pe.generation[i] == "ld":
            pe.tau2.append(0)
            continue
        pe.tau2.append(get_tau2_path(pe.orders[i], pe.lmrs[i], pe.interfaces[0]))
    pe.tau2 = np.array(pe.tau2)

    # Get the average tau2 for each path type. Each path has a weight w.
    for ptype in ("LML", "LMR", "RML", "RMR"):
        totweight = np.sum(pe.weights[pe.lmrs == ptype])
        if totweight != 0:  # Make sure total weight is not zero
            pe.tau2avg[ptype] = np.average(pe.tau2[pe.lmrs == ptype], weights=pe.weights[pe.lmrs == ptype])

def get_tau1_path(orders, ptype, intfs):
    """
    Return the number of steps it took for this path to cross to the next interface.

    In iSTAR, this measures the time from one interface turn to reaching the next
    interface in the path trajectory.

    Parameters
    ----------
    orders : np.ndarray
        Order parameters for the path.
    ptype : str
        Path type (e.g., "LMR", "LML", "RML", "RMR").
    intfs : list of float
        Interface values.

    Returns
    -------
    int
        Number of steps to cross to the next interface.

    Raises
    ------
    ValueError
        If the path type is unknown.
    """
    if ptype in ("LMR", "LML"):
        a = np.where(orders[:, 0] >= intfs[0])[0][0]  # L->M->. cross
        b = np.where(orders[:, 0] >= intfs[1])[0][0]  # L->M->. cross
        return b - a
    elif ptype in ("RML", "RMR"):
        a = np.where(orders[:, 0] <= intfs[2])[0][0]  # .<-M<-R cross
        b = np.where(orders[:, 0] <= intfs[1])[0][0]  # .<-M<-R cross
        return b - a
    else:
        raise ValueError(f"Unknown path type {ptype}")

def get_tau2_path(orders, ptype, intfs):
    """
    Return the number of steps in the path after the last crossing of the final interface.

    In iSTAR, this measures the time from the last interface crossing to the path endpoint.

    Parameters
    ----------
    orders : np.ndarray
        Order parameters for the path.
    ptype : str
        Path type (e.g., "LMR", "LML", "RML", "RMR").
    intfs : list of float
        Interface values.

    Returns
    -------
    int
        Number of steps after the last interface crossing.

    Raises
    ------
    ValueError
        If the path type is unknown.
    """
    if ptype in ("LML", "RML"):
        a = np.where(orders[::-1, 0] >= intfs[0])[0][0]  # L<-M<-. cross
        b = np.where(orders[::-1, 0] >= intfs[1])[0][0]  # L<-M<-. cross
        return b - a
    elif ptype in ("LMR", "RMR"):
        a = np.where(orders[::-1, 0] <= intfs[2])[0][0]  # .->M->R cross
        b = np.where(orders[::-1, 0] <= intfs[1])[0][0]  # .->M->R cross
        return b - a
    else:
        raise ValueError(f"Unknown path type {ptype}")

def get_tau_path(orders, ptype, intfs):
    """
    Return the total number of steps in the path, excluding the start and end points.

    Parameters
    ----------
    orders : np.ndarray
        Order parameters for the path.
    ptype : str
        Path type (e.g., "LMR", "LML", "RML", "RMR").
    intfs : list of float
        Interface values.

    Returns
    -------
    int
        Total number of steps in the path.

    Raises
    ------
    ValueError
        If the path type is unknown.
    """
    if ptype in ("LMR", "LML"):
        a1 = np.where(orders[:, 0] >= intfs[0])[0][0]  # L->M->. cross
    elif ptype in ("RML", "RMR"):
        a1 = np.where(orders[:, 0] <= intfs[2])[0][0]  # .<-M<-R cross
    else:
        raise ValueError(f"Unknown path type {ptype}")
    # Cut off piece at end
    if ptype in ("LML", "RML"):
        a2 = np.where(orders[::-1, 0] >= intfs[0])[0][0]  # L<-M<-. cross
    elif ptype in ("LMR", "RMR"):
        a2 = np.where(orders[::-1, 0] <= intfs[2])[0][0]  # .->M->R cross
    else:
        raise ValueError(f"Unknown path type {ptype}")
    b = len(orders)  # len(pe.orders[i]) = path length of path i
    return b - a1 - a2

def set_tau_distrib(pe):
    """
    Set the average total path length for each path type.

    Parameters
    ----------
    pe : :py:class:`.PathEnsemble`
        The path ensemble object.

    Returns
    -------
    None
    """
    pe.tau = []
    pe.tauavg = {"LML": None, "LMR": None, "RML": None, "RMR": None, "L*L": None, "R*R": None}
    # Determine path types
    if pe.in_zero_minus:
        if pe.has_zero_minus_one:
            ptypes = ["LML", "LMR", "RML", "RMR", "L*L", "R*R"]
        else:
            ptypes = ["RMR",]
    else:
        ptypes = ["LML", "LMR", "RML", "RMR"]
    # Select the accepted paths and collect path lengths
    for i in range(len(pe.cyclenumbers)):
        if pe.flags[i] != "ACC" or pe.generation[i] == "ld":
            pe.tau.append(0)
            continue 
        pe.tau.append(get_tau_path(pe.orders[i], pe.lmrs[i], pe.interfaces[0]))
    pe.tau = np.array(pe.tau)

    # Get the average tau for each path type. Each path has a weight w.
    for ptype in ptypes:
        totweight = np.sum(pe.weights[pe.lmrs == ptype])
        if totweight != 0:  # Make sure total weight is not zero
            pe.tauavg[ptype] = np.average(pe.tau[pe.lmrs == ptype], weights=pe.weights[pe.lmrs == ptype])

def collect_tau(pathensembles):
    """
    Compute average path lengths for all path ensembles.

    Parameters
    ----------
    pathensembles : list of :py:class:`.PathEnsemble`
        List of path ensemble objects.

    Returns
    -------
    tuple
        Average path lengths (taumm, taump, taupm, taupp).
    """
    print("Collect tau")
    taumm = np.zeros(len(pathensembles))
    taump = np.zeros(len(pathensembles))
    taupm = np.zeros(len(pathensembles))
    taupp = np.zeros(len(pathensembles))
    
    for i in range(len(pathensembles)):
        print("ensemble", i, pathensembles[i].name)
        taumm[i] = pathensembles[i].tauavg['LML']
        taump[i] = pathensembles[i].tauavg['LMR']
        taupm[i] = pathensembles[i].tauavg['RML']
        taupp[i] = pathensembles[i].tauavg['RMR']
    return taumm, taump, taupm, taupp

def collect_tau1(pathensembles):
    """
    Compute and collect average time to reach the next interface for all path ensembles.

    In iSTAR, this metric captures the transition time from one interface to the next,
    which is essential for constructing the time-dependent transition probabilities.

    Parameters
    ----------
    pathensembles : list of :py:class:`.PathEnsemble`
        List of path ensemble objects.

    Returns
    -------
    tuple
        Average times to reach the next interface (tau1_mm, tau1_mp, tau1_pm, tau1_pp).
    """
    print("Collect tau1")
    tau1_mm = np.zeros(len(pathensembles))
    tau1_mp = np.zeros(len(pathensembles))
    tau1_pm = np.zeros(len(pathensembles))
    tau1_pp = np.zeros(len(pathensembles)) 

    for i in range(len(pathensembles)):
        tau1_mm[i] = pathensembles[i].tau1avg['LML']
        tau1_mp[i] = pathensembles[i].tau1avg['LMR']
        tau1_pm[i] = pathensembles[i].tau1avg['RML']
        tau1_pp[i] = pathensembles[i].tau1avg['RMR']
    return tau1_mm, tau1_mp, tau1_pm, tau1_pp

def collect_tau2(pathensembles):
    """
    Compute and collect average time after the last interface crossing for all path ensembles.

    In iSTAR, this represents the final segment of path trajectories after leaving
    the last interface before reaching the endpoint.

    Parameters
    ----------
    pathensembles : list of :py:class:`.PathEnsemble`
        List of path ensemble objects.

    Returns
    -------
    tuple
        Average times after the last interface crossing (tau2_mm, tau2_mp, tau2_pm, tau2_pp).
    """
    print("Collect tau2")
    tau2_mm = np.zeros(len(pathensembles))
    tau2_mp = np.zeros(len(pathensembles))
    tau2_pm = np.zeros(len(pathensembles))
    tau2_pp = np.zeros(len(pathensembles))
    
    for i in range(len(pathensembles)):
        tau2_mm[i] = pathensembles[i].tau2avg['LML']
        tau2_mp[i] = pathensembles[i].tau2avg['LMR']
        tau2_pm[i] = pathensembles[i].tau2avg['RML']
        tau2_pp[i] = pathensembles[i].tau2avg['RMR']
    return tau2_mm, tau2_mp, tau2_pm, tau2_pp

def collect_taum(pathensembles):
    """
    Compute and collect average time between the first and last interface crossing for all path ensembles.

    In iSTAR analysis, this represents the time spent between interface turns,
    which is crucial for understanding transition dynamics.

    Parameters
    ----------
    pathensembles : list of :py:class:`.PathEnsemble`
        List of path ensemble objects.

    Returns
    -------
    tuple
        Average times between interface crossings (taum_mm, taum_mp, taum_pm, taum_pp).
    """
    print("Collect taum")
    taum_mm = np.zeros(len(pathensembles))
    taum_mp = np.zeros(len(pathensembles))
    taum_pm = np.zeros(len(pathensembles))
    taum_pp = np.zeros(len(pathensembles))

    for i in range(len(pathensembles)):
        pe = pathensembles[i]
        # Check if tau exists for this path type, then rest should exist too
        if pe.tauavg['LML'] is not None:
            taum_mm[i] = pathensembles[i].tauavg['LML'] - pathensembles[i].tau1avg['LML'] - pathensembles[i].tau2avg['LML']
        if pe.tauavg['LMR'] is not None:            
            taum_mp[i] = pathensembles[i].tauavg['LMR'] - pathensembles[i].tau1avg['LMR'] - pathensembles[i].tau2avg['LMR']
        if pe.tauavg['RML'] is not None:
            taum_pm[i] = pathensembles[i].tauavg['RML'] - pathensembles[i].tau1avg['RML'] - pathensembles[i].tau2avg['RML']
        if pe.tauavg['RMR'] is not None:
            taum_pp[i] = pathensembles[i].tauavg['RMR'] - pathensembles[i].tau1avg['RMR'] - pathensembles[i].tau2avg['RMR']

    return taum_mm, taum_mp, taum_pm, taum_pp

def set_taus(pe):
    """
    Set the average path lengths before and after interface crossings, and the average total path length.

    This function computes all three tau metrics for iSTAR analysis:
    - tau1: time to reach the next interface
    - tau2: time after the last interface crossing
    - tau: total path length

    Parameters
    ----------
    pe : :py:class:`.PathEnsemble`
        The path ensemble object.

    Returns
    -------
    None
    """
    pe.tau1 = []
    pe.tau2 = []
    pe.tau = []
    
    pe.tau1avg = {"LML": None, "LMR": None, "RML": None, "RMR": None}
    pe.tau2avg = {"LML": None, "LMR": None, "RML": None, "RMR": None}
    pe.tauavg = {"LML": None, "LMR": None, "RML": None, "RMR": None, "L*L": None, "R*R": None}

    # Determine path types
    if pe.in_zero_minus:
        if pe.has_zero_minus_one:
            ptypes = ["LML", "LMR", "RML", "RMR", "L*L", "R*R"]
        else:
            ptypes = ["RMR",]
    else:
        ptypes = ["LML", "LMR", "RML", "RMR"]

    # Loop over all paths, compute tau1, tau2, and total tau
    for i in range(len(pe.cyclenumbers)):
        if pe.flags[i] != "ACC" or pe.generation[i] == "ld":
            # If not accepted or if generation is "ld", set zero for all values
            pe.tau1.append(0)
            pe.tau2.append(0)
            pe.tau.append(0)
            continue
        
        # Compute tau1 and tau2
        tau1_value = get_tau1_path(pe.orders[i], pe.lmrs[i], pe.interfaces[0])
        tau2_value = get_tau2_path(pe.orders[i], pe.lmrs[i], pe.interfaces[0])

        # Compute total tau (the path length)im
        tau_value = get_tau_path(pe.orders[i], pe.lmrs[i], pe.interfaces[0])

        # Append results
        pe.tau1.append(tau1_value)
        pe.tau2.append(tau2_value)
        pe.tau.append(tau_value)

    # Convert lists to numpy arrays for efficient processing
    pe.tau1 = np.array(pe.tau1)
    pe.tau2 = np.array(pe.tau2)
    pe.tau = np.array(pe.tau)

    # Calculate the average tau1, tau2, and total tau for each path type
    for ptype in ptypes:
        # Total weight for each path type
        totweight = np.sum(pe.weights[pe.lmrs == ptype])
        
        # Compute average tau1 for the current path type
        if totweight != 0:  # Ensure total weight is not zero
            pe.tau1avg[ptype] = np.average(pe.tau1[pe.lmrs == ptype],
                                           weights=pe.weights[pe.lmrs == ptype])
            pe.tau2avg[ptype] = np.average(pe.tau2[pe.lmrs == ptype],
                                           weights=pe.weights[pe.lmrs == ptype])
            pe.tauavg[ptype] = np.average(pe.tau[pe.lmrs == ptype],
                                          weights=pe.weights[pe.lmrs == ptype])


# Backward compatibility alias for REPPTIS code
set_tau_first_hit_M_distrib = set_tau_first_hit_interface_distrib

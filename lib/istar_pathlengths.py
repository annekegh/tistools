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

#======================================
# Mean first passage times
#======================================

def mfpt_to_absorbing_staple(M, tau1, taum, tau2, absor, kept, doprint=False, remove_initial_m=True):
    """
    Compute the mean first passage time (MFPT) to absorbing states.

    This function calculates the MFPT to reach any of the specified absorbing states 
    in a Markov process, both unconditionally (G) and conditionally on leaving the 
    current state (H).
    
    Supports both vector-valued tau (tau[i] for state i) and matrix-valued tau 
    (tau[i,j] for transition from state i to state j). Matrix-valued tau provides 
    more accurate estimates by using transition-specific path lengths.

    Parameters
    ----------
    M : np.ndarray
        The transition matrix of the Markov process, shape (NS, NS).
    tau1 : np.ndarray
        Time before the first interface crossing. Can be:
        - 1D array of shape (NS,): tau1[i] is time for state i
        - 2D array of shape (NS, NS): tau1[i,j] is time for transition i→j
    taum : np.ndarray
        Time spent between the first and last interface crossing. Same shape options as tau1.
    tau2 : np.ndarray
        Time after the last interface crossing. Same shape options as tau1.
    absor : list or np.ndarray
        Indices of absorbing states.
    kept : list or np.ndarray
        Indices of nonboundary (non-absorbing) states.
    doprint : bool, optional
        If True, prints intermediate computation details. Default is False.
    remove_initial_m : bool or str, optional
        If True or "m", the middle part (taum) is removed from the initial state's MFPT.
        Default is True.

    Raises
    ------
    ValueError
        If the transition matrix has fewer than three states.
        If the absorbing and nonboundary states do not partition the state space.

    Returns
    -------
    g1 : np.ndarray
        An array of size (n_absorb, 1) containing unconditional MFPTs for absorbing states.
    g2 : np.ndarray
        An array of size (n_kept, 1) containing unconditional MFPTs for nonboundary states.
    h1 : np.ndarray
        An array of size (n_absorb, 1) containing conditional MFPTs for absorbing states.
    h2 : np.ndarray
        An array of size (n_kept, 1) containing conditional MFPTs for nonboundary states.

    Notes
    -----
    For matrix-valued tau, the effective time from state i is computed as:
        tau_eff[i] = sum_j M[i,j] * tau[i,j]
    This accounts for the fact that path lengths depend on both the starting 
    and ending states of each transition.
    
    The MFPT equations are:
    - g1 = 0 (boundary condition for absorbing states)
    - (I - Mp) g2 = D g1 + tp  (solve for nonboundary states)
    - h1 = M11 g1 + E g2 + t1  (conditional MFPT for absorbing)
    - h2 = D g1 + Mp g2 + tp   (conditional MFPT for nonboundary)
    """
    NS = len(M)
    if NS < 3:
        raise ValueError(f"Transition matrix must have at least 3 states, but got {NS}.")

    check_valid_indices(M, absor, kept)

    if len(M) != len(absor) + len(kept):
        raise ValueError("The number of states must be the sum of absorbing and nonboundary states.")

    # Check if tau is matrix-valued (2D) or vector-valued (1D)
    is_matrix_tau = (taum.ndim == 2)
    
    if is_matrix_tau:
        # Matrix-valued tau: compute effective tau weighted by transition probabilities
        # tau_eff[i] = sum_j M[i,j] * tau[i,j]
        taum2 = taum + tau2  # Total time excluding the initial phase (matrix)
        
        # Compute effective times for absorbing states (transitions to all states)
        t1 = np.sum(M[np.ix_(absor, range(NS))] * taum2[np.ix_(absor, range(NS))], axis=1).reshape(-1, 1)
        # Compute effective times for nonboundary states (transitions to all states)
        tp = np.sum(M[np.ix_(kept, range(NS))] * taum2[np.ix_(kept, range(NS))], axis=1).reshape(-1, 1)
        
        # For removing initial middle part, use diagonal elements (self-contribution)
        # or weighted average depending on interpretation
        st1 = np.sum(M[np.ix_(absor, range(NS))] * taum[np.ix_(absor, range(NS))], axis=1).reshape(-1, 1)
        stp = np.sum(M[np.ix_(kept, range(NS))] * taum[np.ix_(kept, range(NS))], axis=1).reshape(-1, 1)
    else:
        # Vector-valued tau (original behavior)
        taum2 = taum + tau2  # Total time excluding the initial phase
        t1 = taum2[absor].reshape(len(absor), 1)
        tp = taum2[kept].reshape(len(kept), 1)
        st1 = taum[absor].reshape(len(absor), 1)
        stp = taum[kept].reshape(len(kept), 1)

    # Extract the submatrix for nonboundary states
    Mp = np.take(np.take(M, kept, axis=0), kept, axis=1)

    # Extract transition probabilities between nonboundary and absorbing states
    D = np.take(np.take(M, kept, axis=0), absor, axis=1)  # Transitions from nonboundary to absorbing states
    E = np.take(np.take(M, absor, axis=0), kept, axis=1)  # Transitions from absorbing to nonboundary states
    M11 = np.take(np.take(M, absor, axis=0), absor, axis=1)  # Transitions within absorbing states

    a = np.identity(len(Mp)) - Mp  # Compute (I - Mp) for solving equations

    # Compute G vector (unconditional MFPT)
    g1 = np.zeros((len(absor), 1))  # Boundary condition: g1 is initialized to zero
    g2 = np.linalg.solve(a, np.dot(D, g1) + tp)  # Solve (I - Mp) g2 = D g1 + tp

    # Compute H vector (conditional MFPT)
    h1 = np.dot(M11, g1) + np.dot(E, g2) + t1  # Conditional MFPT for absorbing states
    h2 = np.dot(D, g1) + np.dot(Mp, g2) + tp  # Conditional MFPT for nonboundary states

    # Remove the average time of middle part (m) of the initial state if specified
    if remove_initial_m:
        h1 -= st1
        h2 -= stp

    if doprint:
        print("Eigenvalues of Mp:")
        print(np.linalg.eigvals(Mp))
        print("Eigenvalues of (I - Mp):")
        print(np.linalg.eigvals(a))
        print("Transition matrix components:")
        print("D:\n", D)
        print("E:\n", E)
        print("M11:\n", M11)
        print("Time vectors (tau m2):")
        print("t1:\n", t1)
        print("tp:\n", tp)
        print("Vectors:")
        print("g1:\n", g1)
        print("g2:\n", g2)
        print("h1:\n", h1)
        print("h2:\n", h2)
        print("Verification (should be close to 0):", np.sum((g2 - h2) ** 2))
        if is_matrix_tau:
            print("Using matrix-valued tau for transition-specific path lengths")

    return g1, g2, h1, h2


def mfpt_to_first_last_staple(M, tau1, taum, tau2, doprint=False):
    """
    Compute the mean first passage time (MFPT) to reach either state 0 or state -1.

    This function calculates the MFPT to reach the first (0) or last (NS-1) state 
    in a Markov process. It does so by treating these two states as absorbing 
    and solving for the expected first passage times.
    
    Supports both vector-valued tau (shape NS) and matrix-valued tau (shape NS×NS)
    for more accurate estimates using transition-specific path lengths.

    Parameters
    ----------
    M : np.ndarray
        The transition matrix of the Markov process, shape (NS, NS).
    tau1 : np.ndarray
        Time before the first interface crossing. Shape (NS,) or (NS, NS).
    taum : np.ndarray
        Time between first and last interface crossing. Shape (NS,) or (NS, NS).
    tau2 : np.ndarray
        Time after the last interface crossing. Shape (NS,) or (NS, NS).
    doprint : bool, optional
        If True, prints intermediate computation details. Default is False.

    Raises
    ------
    ValueError
        If the transition matrix has fewer than three states.

    Returns
    -------
    g1 : np.ndarray
        MFPT for the absorbing states (0 and NS-1).
    g2 : np.ndarray
        MFPT for nonboundary states.
    h1 : np.ndarray
        Conditional MFPT for the absorbing states.
    h2 : np.ndarray
        Conditional MFPT for nonboundary states.

    Notes
    -----
    - The key result is `h1[0]`, which gives the MFPT from state 0 to either 0 or -1, 
      given that the process leaves state 0.
    - Calls `mfpt_to_absorbing_staple` with `remove="m"` to exclude the intermediate 
      passage time contribution from the calculations.
    - For more accurate estimates, use matrix-valued tau created with 
      `construct_tau_vector_staple(..., as_matrix=True)` or `construct_tau_matrix_staple()`.
    """
    NS = len(M)
    if NS < 3:
        raise ValueError(f"Transition matrix must have at least 3 states, but got {NS}.")

    absor = np.array([0, NS - 1])
    kept = np.array([i for i in range(NS) if i not in absor])

    return mfpt_to_absorbing_staple(M, tau1, taum, tau2, absor, kept, doprint=doprint, remove_initial_m="m")


def mfpt_istar(M, tau_interface, use_matrix_tau=True, doprint=False):
    """
    Compute MFPT for iSTAR model from interface-pair path lengths.
    
    This is a convenience function that handles the conversion from 
    interface-indexed tau values to MSM state-indexed values.

    Parameters
    ----------
    M : np.ndarray
        Transition matrix of shape (2N, 2N) from construct_M_istar.
    tau_interface : dict
        Dictionary with keys 'tau1', 'taum', 'tau2', each containing
        an (N, N) matrix of path lengths indexed by (start_intf, end_intf).
    use_matrix_tau : bool, optional
        If True (default), use transition-specific path lengths for more 
        accurate MFPT estimates. If False, use state-averaged values.
    doprint : bool, optional
        If True, print intermediate results. Default is False.

    Returns
    -------
    g1, g2, h1, h2 : np.ndarray
        MFPT results. h1[0] is the key result: MFPT from state A.

    Example
    -------
    >>> # Compute tau matrices from path ensembles
    >>> tau_data = set_taus_staple(pathensembles, interfaces)
    >>> # Build MSM transition matrix
    >>> M = construct_M_istar(P, 2*N, N)
    >>> # Compute MFPT
    >>> g1, g2, h1, h2 = mfpt_istar(M, tau_data, use_matrix_tau=True)
    >>> print(f"MFPT from A: {h1[0]}")
    """
    NS = len(M)
    N = NS // 2
    
    tau1_intf = tau_interface['tau1']
    taum_intf = tau_interface['tau'] - tau_interface['tau1'] - tau_interface['tau2']
    tau2_intf = tau_interface['tau2']
    
    if use_matrix_tau:
        # Convert to MSM state-indexed matrices
        tau1 = construct_tau_matrix_staple(tau1_intf, N)
        taum = construct_tau_matrix_staple(taum_intf, N)
        tau2 = construct_tau_matrix_staple(tau2_intf, N)
    else:
        # Convert to state-averaged vectors
        tau1 = construct_tau_vector_staple(N, NS, tau1_intf)
        taum = construct_tau_vector_staple(N, NS, taum_intf)
        tau2 = construct_tau_vector_staple(N, NS, tau2_intf)
    
    return mfpt_to_first_last_staple(M, tau1, taum, tau2, doprint=doprint)


def construct_tau_matrix_staple(tau_interface, N):
    """
    Construct a tau matrix indexed by MSM states from interface-pair tau values.
    
    The iSTAR MSM has 2N states representing turns at N interfaces:
    - States 0 to N-1: "left turns" (arriving from the left) at interfaces 0 to N-1
    - States N to 2N-1: "right turns" (arriving from the right) at interfaces 0 to N-1
    
    The input tau_interface[i,j] contains path lengths for transitions between 
    interface i and interface j. This function maps these to the MSM state space.
    
    Parameters
    ----------
    tau_interface : np.ndarray
        Matrix of shape (N, N) where tau_interface[i,j] is the path length for 
        transitions from interface i to interface j.
    N : int
        Number of interfaces.
    
    Returns
    -------
    np.ndarray
        Matrix of shape (2N, 2N) where result[s1,s2] is the path length for 
        transition from MSM state s1 to MSM state s2.
    
    Notes
    -----
    The state mapping follows the construct_M_istar convention:
    - State 0: [0-] (left turn at interface 0, i.e., return to A)
    - State 1: [0+] (right turn at interface 0, leaving A)  
    - States 2 to N-1: left turns at interfaces 1 to N-2
    - State N: left turn at interface N-1 (at B)
    - States N+1 to 2N-2: right turns at interfaces 1 to N-2
    - State 2N-1: right turn at interface N-1 (reached B)
    
    The tau values depend on the starting interface (where the turn is) and the 
    ending interface (where the next turn will be).
    """
    NS = 2 * N
    tau_msm = np.zeros((NS, NS))
    
    # Map interface transitions to MSM state transitions
    # For each MSM state s, we need to determine which interface it represents
    # and what direction the path is going
    
    for s_from in range(NS):
        for s_to in range(NS):
            # Determine the interface indices for source and target states
            if s_from == 0:
                intf_from = -1
            elif 2 <= s_from < N+1:
                # Left turn states (0 to N-1) - turn happened at interface s_from
                intf_from = s_from - 2
            elif s_from == NS-1:
                intf_from = -2
            elif s_from >= N+1:
                # Right turn states (N to 2N-1) - turn happened at interface (s_from - N)
                intf_from = s_from - N
            else:
                continue
            
            if s_to == 1:
                intf_to = 0
            elif s_to == NS-1:
                intf_to = -1
            elif s_to < N+1:
                # Target is a left turn state - next turn at interface s_to
                intf_to = s_to - 2
            elif s_to >= N+1:
                # Target is a right turn state - next turn at interface (s_to - N)
                intf_to = s_to - N
            else:
                continue
            
            # Get the tau value for this interface pair
            if 0 <= intf_from < N and 0 <= intf_to < N:
                tau_msm[s_from, s_to] = tau_interface[intf_from, intf_to]
    
    return tau_msm


def _compute_start_end_indices(pe, interfaces):
    """
    Compute start and end interface indices for all paths in an ensemble.
    
    Parameters
    ----------
    pe : :py:class:`.PathEnsemble`
        The path ensemble object.
    interfaces : list of float
        Interface values sorted in ascending order.
    
    Returns
    -------
    tuple of np.ndarray
        (start_indices, end_indices) arrays for all paths.
    """
    start_indices = []
    end_indices = []
    for i in range(len(pe.cyclenumbers)):
        if pe.flags[i] != "ACC" or pe.generation[i] == "ld":
            start_indices.append(-1)
            end_indices.append(-1)
        elif pe.in_zero_minus:
            start_indices.append(-1)
            end_indices.append(0)  # Special case for paths in the zero-minus state TODO: implement l_-1?
        else:
            start, end = get_start_end_interfaces(
                pe.lambmins[i], pe.lambmaxs[i], pe.dirs[i], interfaces
            )
            start_indices.append(start)
            end_indices.append(end)
    return np.array(start_indices), np.array(end_indices)

def construct_tau_vector_staple(N, NS, tau_matrix):
    """
    Construct tau vector for iSTAR MFPT analysis from interface-pair path lengths.

    In iSTAR, path lengths are categorized by (start, end) interface pairs.
    This function converts these to the format needed for MFPT calculations.

    Parameters
    ----------
    N : int
        Number of interfaces.
    NS : int
        Number of states in the MSM (should be 2*N).
    tau_matrix : np.ndarray
        Matrix of path lengths for (start, end) interface pairs.
        Shape should be (N, N) where tau_matrix[i,j] is the path length
        for transitions from interface i to interface j.

    Returns
    -------
    np.ndarray
        If as_matrix=False: 1D array of shape (NS,) with average tau per state.
        If as_matrix=True: 2D array of shape (NS, NS) with tau per transition.

    Raises
    ------
    ValueError
        If the dimensions don't match expected values.
    """
    if N < 3:
        raise ValueError("N must be at least 3.")
    if NS != 2 * N:
        raise ValueError(f"NS must be 2 * N = {2 * N}.")
    if tau_matrix.shape != (N, N):
        raise ValueError(f"Input tau_matrix must have shape ({N}, {N}), got {tau_matrix.shape}.")
    
    # Return (NS,) vector with average tau per starting state
    # Average over destination states weighted by uniform distribution
    # (actual weighting should use transition probabilities)
    tau = np.zeros(NS)
    for s in range(NS):
        if s < N:
            intf = s  # Left turn at interface s
        else:
            intf = s - N  # Right turn at interface (s - N)
        # Average over all possible destination interfaces
        tau[s] = np.nanmean(tau_matrix[intf, :])
    return tau

def set_tau_first_hit_interface_distrib(pe, interfaces, do_last=True):
    """
    Set the average path length before the next interface is reached for each (start, end) pair.

    In iSTAR, we track the time to reach the next interface (first hit) from the 
    current interface, categorized by start and end interface indices.

    Parameters
    ----------
    pe : :py:class:`.PathEnsemble`
        The path ensemble object.
    interfaces : list of float
        Interface values sorted in ascending order.
    do_last : bool, optional
        If True, also compute tau2 (time after last interface crossing). Default is True.

    Returns
    -------
    None
    """
    n_intf = len(interfaces)
    pe.tau1 = []
    pe.tau1avg = np.zeros((n_intf+1, n_intf))  # Matrix indexed by (start, end)
    start_indices = []
    end_indices = []
    
    # Select the accepted paths and compute start/end indices
    for i in range(len(pe.cyclenumbers)):
        if pe.flags[i] != "ACC" or pe.generation[i] == "ld":
            pe.tau1.append(0)
            start_indices.append(-1)
            end_indices.append(-1)
            continue
        # Determine start and end interfaces based on path properties
        if pe.in_zero_minus:
            start = -1
            end = 0  # Special case for paths in the zero-minus state TODO: implement l_-1?
        else:
            start, end = get_start_end_interfaces(
                pe.lambmins[i], pe.lambmaxs[i], pe.dirs[i], interfaces
            )
        start_indices.append(start)
        end_indices.append(end)
        pe.tau1.append(get_tau1_path(pe.orders[i], start, end, interfaces))
    
    pe.tau1 = np.array(pe.tau1)
    start_indices = np.array(start_indices)
    end_indices = np.array(end_indices)

    # Get the average tau1 for each (start, end) pair
    accmask = (pe.flags == "ACC") & (pe.generation != "ld")
    for start in range(-1, n_intf):
        for end in range(n_intf):
            mask = accmask & (start_indices == start) & (end_indices == end)
            totweight = np.sum(pe.weights[mask])
            if totweight > 0:
                pe.tau1avg[start+1, end] = np.average(pe.tau1[mask], weights=pe.weights[mask])

    if not do_last:
        return

    pe.tau2 = []
    pe.tau2avg = np.zeros((n_intf+1, n_intf))  # Matrix indexed by (start, end)
    
    # Select the accepted paths
    for i in range(len(pe.cyclenumbers)):
        if pe.flags[i] != "ACC" or pe.generation[i] == "ld":
            pe.tau2.append(0)
            continue
        start, end = start_indices[i], end_indices[i]
        pe.tau2.append(get_tau2_path(pe.orders[i], start, end, interfaces))
    pe.tau2 = np.array(pe.tau2)

    # Get the average tau2 for each (start, end) pair
    for start in range(-1, n_intf):
        for end in range(n_intf):
            mask = accmask & (start_indices == start) & (end_indices == end)
            totweight = np.sum(pe.weights[mask])
            if totweight > 0:
                pe.tau2avg[start+1, end] = np.average(pe.tau2[mask], weights=pe.weights[mask])

def get_tau1_path(orders, start, end, intfs):
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
    if start in (-1, 0, len(intfs)-1):
        return 0  # No next interface to cross
    
    elif start < end:
        s_idx = np.where(orders[:, 0] <= intfs[start])[-1][0]  # Last index at or before starting interface
        a = np.where(orders[:, 0] <= intfs[start+1])[0][0]  # Next crossing of start interface
        b = np.where(orders[s_idx:, 0] >= intfs[start+1])[0][0] + s_idx  # Next crossing of end interface
    
    elif start > end:
        s_idx = np.where(orders[:, 0] >= intfs[start])[-1][0]  # Last index at or before starting interface
        a = np.where(orders[:, 0] >= intfs[start-1])[0][0]  # Next crossing of start interface
        b = np.where(orders[s_idx:, 0] <= intfs[start-1])[0][0] + s_idx  # Next crossing of end interface
    
    else:
        raise ValueError(f"Unknown start/end combination: {start}, {end}")
    return b - a


def get_tau2_path(orders, start, end, intfs):
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
    orders_rev = orders[::-1, 0]
    if end in (-1, 0, len(intfs)-1):
        return 0  # No last interface crossing to consider
    
    elif start < end:
        e_idx = np.where(orders_rev >= intfs[end])[-1][0]  # Last index at or before ending interface
        a = np.where(orders_rev >= intfs[end-1])[0][0]  # Last crossing of end interface
        b = np.where(orders_rev[e_idx:] <= intfs[end-1])[0][0] + e_idx  # Last crossing of start interface
    
    elif start > end:
        e_idx = np.where(orders_rev <= intfs[end])[-1][0]  # Last index at or before ending interface
        a = np.where(orders_rev <= intfs[end+1])[0][0]  # Last crossing of end interface
        b = np.where(orders_rev[e_idx:] >= intfs[end+1])[0][0] + e_idx  # Last crossing of start interface
    
    else:
        raise ValueError(f"Unknown start/end combination: {start}, {end}")
    return b - a

def get_tau_path(orders, start, end, intfs):
    """
    Return the total number of steps in the path, excluding the start and end points.

    Parameters
    ----------
    orders : np.ndarray
        Order parameters for the path.
    lambmin : float
        Minimum order parameter value of the path.
    lambmax : float
        Maximum order parameter value of the path.
    intfs : list of float
        Interface values.

    Returns
    -------
    int
        Total number of steps in the path.

    Raises
    ------
    ValueError
        If the start/end combination is invalid.
    """
    if start == end == 0:
        a1 = np.where(orders[:, 0] >= intfs[0])[0][0]  
        a2 = np.where(orders[::-1, 0] >= intfs[0])[0][0]  
    elif start == -1:
        # TODO implement l_-1?
        a1 = np.where(orders[:, 0] <= intfs[0])[0][0]  
        a2 = np.where(orders[::-1, 0] <= intfs[0])[0][0]  
    elif start < end:
        a1 = np.where(orders[:, 0] <= intfs[start+1])[0][0] 
        a2 = np.where(orders[::-1, 0] >= intfs[end-1])[0][0] 
    elif start > end:
        a1 = np.where(orders[:, 0] >= intfs[start-1])[0][0]  
        a2 = np.where(orders[::-1, 0] <= intfs[end+1])[0][0]        
    else:
        raise ValueError(f"Unknown start/end combination: {start}, {end}")
    
    if 0 in (start, end) or len(intfs)-1 in (start, end):
        if start == 0:
            a1 = np.where(orders[:, 0] >= intfs[0])[0][0]
        elif start == len(intfs)-1:
            a1 = np.where(orders[:, 0] <= intfs[-1])[0][0]
        if end == 0:
            a2 = np.where(orders[::-1, 0] >= intfs[0])[0][0]
        elif end == len(intfs)-1:
            a2 = np.where(orders[::-1, 0] <= intfs[-1])[0][0]
            
    b = len(orders)  # len(pe.orders[i]) = path length of path i
    return b - a1 - a2

def set_tau_distrib(pe, interfaces):
    """
    Set the average total path length for each (start, end) interface pair.

    Parameters
    ----------
    pe : :py:class:`.PathEnsemble`
        The path ensemble object.
    interfaces : list of float
        Interface values sorted in ascending order.

    Returns
    -------
    None
    """
    n_intf = len(interfaces)
    pe.tau = []
    pe.tauavg = np.zeros((n_intf+1, n_intf))  # Matrix indexed by (start, end)
    
    # Compute start/end indices for each path
    start_indices = []
    end_indices = []
    for i in range(len(pe.cyclenumbers)):
        if pe.flags[i] != "ACC" or pe.generation[i] == "ld":
            start_indices.append(-1)
            end_indices.append(-1)
        elif pe.in_zero_minus:
            start_indices.append(-1)
            end_indices.append(0)  # Special case for paths in the zero-minus state TODO: implement l_-1?
        else:
            start, end = get_start_end_interfaces(
                pe.lambmins[i], pe.lambmaxs[i], pe.dirs[i], interfaces
            )
            start_indices.append(start)
            end_indices.append(end)
    start_indices = np.array(start_indices)
    end_indices = np.array(end_indices)
    
    # Select the accepted paths and collect path lengths
    for i in range(len(pe.cyclenumbers)):
        if pe.flags[i] != "ACC" or pe.generation[i] == "ld":
            pe.tau.append(0)
            continue
        start, end = start_indices[i], end_indices[i]
        pe.tau.append(get_tau_path(pe.orders[i], start, end, interfaces))
    pe.tau = np.array(pe.tau)

    # Get the average tau for each (start, end) pair
    accmask = (pe.flags == "ACC") & (pe.generation != "ld")
    for start in range(-1, n_intf):
        for end in range(n_intf):
            mask = accmask & (start_indices == start) & (end_indices == end)
            totweight = np.sum(pe.weights[mask])
            if totweight > 0:
                pe.tauavg[start+1, end] = np.average(pe.tau[mask], weights=pe.weights[mask])

def collect_tau_staple(pathensembles, interfaces):
    """
    Compute average path lengths for all path ensembles as a matrix.

    Parameters
    ----------
    pathensembles : list of :py:class:`.PathEnsemble`
        List of path ensemble objects.
    interfaces : list of float
        Interface values sorted in ascending order.

    Returns
    -------
    np.ndarray
        Combined average path lengths matrix of shape (n_interfaces, n_interfaces),
        where tau_combined[start, end] is the weighted average across all ensembles.
    """
    print("Collect tau")
    n_intf = len(interfaces)
    n_ens = len(pathensembles)
    
    # Collect tau matrices and weights from all ensembles
    tau_matrices = np.zeros((n_ens, n_intf+1, n_intf))
    weights = np.zeros((n_ens, n_intf+1, n_intf))
    
    for i, pe in enumerate(pathensembles):
        print("ensemble", i, pe.name)
        if hasattr(pe, 'tauavg') and isinstance(pe.tauavg, np.ndarray):
            tau_matrices[i] = np.nan_to_num(pe.tauavg, nan=0.0)
            # Compute start/end indices and weights for each (start, end) pair
            accmask = (pe.flags == "ACC") & (pe.generation != "ld")
            start_indices, end_indices = _compute_start_end_indices(pe, interfaces)
            for start in range(-1, n_intf):
                for end in range(n_intf):
                    mask = accmask & (start_indices == start) & (end_indices == end)
                    weights[i, start+1, end] = np.sum(pe.weights[mask])
    
    # Compute weighted average across ensembles
    tau_combined = np.zeros((n_intf+1, n_intf))
    for start in range(-1, n_intf):
        for end in range(n_intf):
            total_weight = np.sum(weights[:, start+1, end])
            if total_weight > 0:
                tau_combined[start+1, end] = np.sum(
                    tau_matrices[:, start+1, end] * weights[:, start+1, end]
                ) / total_weight
    
    return tau_combined

def collect_tau1_staple(pathensembles, interfaces):
    """
    Compute and collect average time to reach the next interface for all path ensembles.

    In iSTAR, this metric captures the transition time from one interface to the next,
    which is essential for constructing the time-dependent transition probabilities.

    Parameters
    ----------
    pathensembles : list of :py:class:`.PathEnsemble`
        List of path ensemble objects.
    interfaces : list of float
        Interface values sorted in ascending order.

    Returns
    -------
    np.ndarray
        Combined average tau1 matrix of shape (n_interfaces, n_interfaces),
        where tau1_combined[start, end] is the weighted average across all ensembles.
    """
    print("Collect tau1")
    n_intf = len(interfaces)
    n_ens = len(pathensembles)
    
    # Collect tau1 matrices and weights from all ensembles
    tau1_matrices = np.zeros((n_ens, n_intf+1, n_intf))
    weights = np.zeros((n_ens, n_intf+1, n_intf))
    
    for i, pe in enumerate(pathensembles):
        if hasattr(pe, 'tau1avg') and isinstance(pe.tau1avg, np.ndarray):
            tau1_matrices[i] = np.nan_to_num(pe.tau1avg, nan=0.0)
            # Compute start/end indices and weights for each (start, end) pair
            accmask = (pe.flags == "ACC") & (pe.generation != "ld")
            start_indices, end_indices = _compute_start_end_indices(pe, interfaces)
            for start in range(-1, n_intf):
                for end in range(n_intf):
                    mask = accmask & (start_indices == start) & (end_indices == end)
                    weights[i, start+1, end] = np.sum(pe.weights[mask])
    
    # Compute weighted average across ensembles
    tau1_combined = np.zeros((n_intf+1, n_intf))
    for start in range(-1, n_intf):
        for end in range(n_intf):
            total_weight = np.sum(weights[:, start+1, end])
            if total_weight > 0:
                tau1_combined[start+1, end] = np.sum(
                    tau1_matrices[:, start+1, end] * weights[:, start+1, end]
                ) / total_weight
    
    return tau1_combined

def collect_tau2_staple(pathensembles, interfaces):
    """
    Compute and collect average time after the last interface crossing for all path ensembles.

    In iSTAR, this represents the final segment of path trajectories after leaving
    the last interface before reaching the endpoint.

    Parameters
    ----------
    pathensembles : list of :py:class:`.PathEnsemble`
        List of path ensemble objects.
    interfaces : list of float
        Interface values sorted in ascending order.

    Returns
    -------
    np.ndarray
        Combined average tau2 matrix of shape (n_interfaces, n_interfaces),
        where tau2_combined[start, end] is the weighted average across all ensembles.
    """
    print("Collect tau2")
    n_intf = len(interfaces)
    n_ens = len(pathensembles)
    
    # Collect tau2 matrices and weights from all ensembles
    tau2_matrices = np.zeros((n_ens, n_intf+1, n_intf))
    weights = np.zeros((n_ens, n_intf+1, n_intf))
    
    for i, pe in enumerate(pathensembles):
        if hasattr(pe, 'tau2avg') and isinstance(pe.tau2avg, np.ndarray):
            tau2_matrices[i] = np.nan_to_num(pe.tau2avg, nan=0.0)
            # Compute start/end indices and weights for each (start, end) pair
            accmask = (pe.flags == "ACC") & (pe.generation != "ld")
            start_indices, end_indices = _compute_start_end_indices(pe, interfaces)
            for start in range(-1, n_intf):
                for end in range(n_intf):
                    mask = accmask & (start_indices == start) & (end_indices == end)
                    weights[i, start+1, end] = np.sum(pe.weights[mask])
    
    # Compute weighted average across ensembles
    tau2_combined = np.zeros((n_intf+1, n_intf))
    for start in range(-1, n_intf):
        for end in range(n_intf):
            total_weight = np.sum(weights[:, start+1, end])
            if total_weight > 0:
                tau2_combined[start+1, end] = np.sum(
                    tau2_matrices[:, start+1, end] * weights[:, start+1, end]
                ) / total_weight
    
    return tau2_combined

def collect_taum_staple(pathensembles, interfaces):
    """
    Compute and collect average time between the first and last interface crossing for all path ensembles.

    In iSTAR analysis, this represents the time spent between interface turns,
    which is crucial for understanding transition dynamics. Computed as tau - tau1 - tau2.

    Parameters
    ----------
    pathensembles : list of :py:class:`.PathEnsemble`
        List of path ensemble objects.
    interfaces : list of float
        Interface values sorted in ascending order.

    Returns
    -------
    np.ndarray
        Combined average taum matrix of shape (n_interfaces, n_interfaces),
        where taum_combined[start, end] is the weighted average across all ensembles.
    """
    print("Collect taum")
    n_intf = len(interfaces)
    
    # Get the combined tau matrices
    tau_combined = collect_tau_staple(pathensembles, interfaces)
    tau1_combined = collect_tau1_staple(pathensembles, interfaces)
    tau2_combined = collect_tau2_staple(pathensembles, interfaces)
    
    # Compute taum = tau - tau1 - tau2
    taum_combined = tau_combined - tau1_combined - tau2_combined
    
    return taum_combined

def get_start_end_interfaces(lambmin, lambmax, direction, interfaces):
        """
        Extract the start and end interface indices for a path based on lambda bounds and direction.
        
        Parameters
        ----------
        lambmin : float
            Minimum order parameter value of the path.
        lambmax : float
            Maximum order parameter value of the path.
        direction : int
            Path direction: 1 for forward, -1 for backward.
        interfaces : list of float
            Interface values sorted in ascending order.
        
        Returns
        -------
        tuple of int
            (start, end) interface indices where:
            - 0 means before/at first interface
            - len(interfaces)-1 means at/after last interface
            - Other values indicate the interface region
        """
        if direction == 1:  # forward direction
            if lambmin <= interfaces[0]:
                start = 0
            else:
                start = np.searchsorted(interfaces, lambmin, side='left')
            if lambmax >= interfaces[-1]:
                end = len(interfaces) - 1
            else:
                end = np.searchsorted(interfaces, lambmax, side='right') - 1
        else:  # backward direction
            if lambmin <= interfaces[0]:
                end = 0
            else:
                end = np.searchsorted(interfaces, lambmin, side='left')
            if lambmax >= interfaces[-1]:
                start = len(interfaces) - 1
            else:
                start = np.searchsorted(interfaces, lambmax, side='right') - 1
        
        return start, end

def set_taus_staple(pathensembles, interfaces):
    """
    Set the average path lengths before and after interface crossings, and the average total path length.

    This function computes all three tau metrics for iSTAR analysis:
    - tau1: time to reach the next interface
    - tau2: time after the last interface crossing
    - tau: total path length

    Parameters
    ----------
    pathensembles : list of :py:class:`.PathEnsemble`
        List of path ensemble objects.
    interfaces : list of float
        Interface values sorted in ascending order.

    Returns
    -------
    dict
        Dictionary with 'tau1', 'tau2', 'tau' containing the weighted averages
        across all ensembles as matrices of shape (n_interfaces, n_interfaces).
    """
    n_intf = len(interfaces)
    tau_avg = {}
    totweights = []
    
    for pe in pathensembles:
        pe.tau1 = []
        pe.tau2 = []
        pe.tau = []
        start_indices = []
        end_indices = []
        pe.tau1avg = np.zeros((n_intf+1, n_intf))
        pe.tau2avg = np.zeros((n_intf+1, n_intf))
        pe.tauavg = np.zeros((n_intf+1, n_intf))

        # Loop over all paths, compute start/end indices and tau values
        for i in range(len(pe.cyclenumbers)):
            if pe.flags[i] != "ACC" or pe.generation[i] == "ld":
                # If not accepted or if generation is "ld", set zero for all values
                pe.tau1.append(0)
                pe.tau2.append(0)
                pe.tau.append(0)
                start_indices.append(-1)
                end_indices.append(-1)
                continue
            
            if pe.in_zero_minus:
                start = -1
                end = 0  # Special case for paths in the zero-minus state TODO: implement l_-1?
            else:
                # Determine start and end interface indices
                start, end = get_start_end_interfaces(
                    pe.lambmins[i], pe.lambmaxs[i], pe.dirs[i], interfaces
                )
            start_indices.append(start)
            end_indices.append(end)
            
            # Compute tau1, tau2, and total tau
            tau1_value = get_tau1_path(pe.orders[i], start, end, interfaces)
            tau2_value = get_tau2_path(pe.orders[i], start, end, interfaces)
            tau_value = get_tau_path(pe.orders[i], start, end, interfaces)

            # Append results
            pe.tau1.append(tau1_value)
            pe.tau2.append(tau2_value)
            pe.tau.append(tau_value)

        # Convert lists to numpy arrays for efficient processing
        pe.tau1 = np.array(pe.tau1)
        pe.tau2 = np.array(pe.tau2)
        pe.tau = np.array(pe.tau)
        start_indices = np.array(start_indices)
        end_indices = np.array(end_indices)

        # Calculate the average tau1, tau2, and total tau for each (start, end) pair
        accmask = (pe.flags == "ACC") & (pe.generation != "ld")
        
        for start in range(-1, n_intf):
            for end in range(n_intf):
                mask = accmask & (start_indices == start) & (end_indices == end)
                totweight = np.sum(pe.weights[mask])
                if totweight > 0:
                    pe.tau1avg[start+1, end] = np.average(pe.tau1[mask], weights=pe.weights[mask])
                    pe.tau2avg[start+1, end] = np.average(pe.tau2[mask], weights=pe.weights[mask])
                    pe.tauavg[start+1, end] = np.average(pe.tau[mask], weights=pe.weights[mask])
        
        totweights.append(np.sum(pe.weights[accmask]))
        
    # Now compute the overall averages across all path ensembles
    totweights = np.array(totweights)
    total_weight = np.sum(totweights)
    if total_weight == 0:
        logging.warning("Total weight across all path ensembles is zero. Cannot compute average taus.")
        return tau_avg
    
    # Compute weighted average matrices
    tau1_matrices = np.array([np.nan_to_num(pe.tau1avg, nan=0.0) for pe in pathensembles])
    tau2_matrices = np.array([np.nan_to_num(pe.tau2avg, nan=0.0) for pe in pathensembles])
    tau_matrices = np.array([np.nan_to_num(pe.tauavg, nan=0.0) for pe in pathensembles])
    
    tau_avg['tau1'] = np.average(tau1_matrices, axis=0, weights=totweights)
    tau_avg['tau2'] = np.average(tau2_matrices, axis=0, weights=totweights)
    tau_avg['tau'] = np.average(tau_matrices, axis=0, weights=totweights)
    
    return tau_avg


def check_valid_indices(M, absor, kept):
    """
    Validate the indices of absorbing and non-absorbing states in a Markov process.

    This function ensures that the indices of absorbing (`absor`) and 
    non-absorbing (`kept`) states are correctly defined and do not overlap.

    Parameters
    ----------
    M : np.ndarray
        The transition matrix of the Markov process.
    absor : array-like
        Indices of absorbing states.
    kept : array-like
        Indices of non-absorbing (nonboundary) states.

    Raises
    ------
    ValueError
        If there are duplicate indices in `absor` or `kept`.
    ValueError
        If any index in `absor` or `kept` is out of bounds.
    ValueError
        If any state is neither in `absor` nor in `kept`, or if a state is in both.

    Notes
    -----
    - The function checks that each state is assigned to exactly one of `absor` or `kept`.
    - Ensures indices are valid (non-negative and within bounds).
    - Prevents duplicate entries in either `absor` or `kept`.
    """
    NS = len(M)

    if len(set(absor)) != len(absor):
        raise ValueError("Duplicate indices found in `absor`.")

    if len(set(kept)) != len(kept):
        raise ValueError("Duplicate indices found in `kept`.")

    if min(absor) < 0 or max(absor) >= NS:
        raise ValueError("Indices in `absor` must be within the valid range [0, NS-1].")

    if min(kept) < 0 or max(kept) >= NS:
        raise ValueError("Indices in `kept` must be within the valid range [0, NS-1].")

    for i in range(NS):
        if (i in absor) == (i in kept):  
            raise ValueError(f"State {i} must be in either `absor` or `kept`, but not both.")


def get_pieces_matrix(M, absor, kept):
    """
    Extract submatrices from a transition matrix corresponding to absorbing and nonboundary states.

    This function partitions a given transition matrix `M` into four submatrices:
    - `Mp`: The transition matrix for nonboundary (non-absorbing) states.
    - `D`: The transitions from nonboundary states to absorbing states.
    - `E`: The transitions from absorbing states to nonboundary states.
    - `M11`: The transition matrix for absorbing states.

    Parameters
    ----------
    M : np.ndarray
        The transition matrix of the Markov process.
    absor : array-like
        Indices of absorbing states.
    kept : array-like
        Indices of non-absorbing (nonboundary) states.

    Raises
    ------
    ValueError
        If `absor` and `kept` contain invalid or overlapping indices.
    ValueError
        If `absor` contains only one element, reshaping is required for proper computation.

    Returns
    -------
    Mp : np.ndarray
        Submatrix corresponding to nonboundary-to-nonboundary state transitions.
    D : np.ndarray
        Submatrix representing transitions from nonboundary to absorbing states.
    E : np.ndarray
        Submatrix representing transitions from absorbing to nonboundary states.
    M11 : np.ndarray
        Submatrix corresponding to transitions between absorbing states.
    """
    check_valid_indices(M, absor, kept)

    # Extract matrix blocks
    Mp = np.take(np.take(M, kept, axis=0), kept, axis=1)
    D = np.take(np.take(M, kept, axis=0), absor, axis=1)
    E = np.take(np.take(M, absor, axis=0), kept, axis=1)
    M11 = np.take(np.take(M, absor, axis=0), absor, axis=1)

    # Handle edge case where there's only one absorbing state
    if len(absor) == 1:
        D = D.reshape(-1, 1)  # Ensure D is a column vector
        E = E.reshape(1, -1)  # Ensure E is a row vector
        M11 = M11.reshape(1, 1)  # Ensure M11 is a 1x1 matrix

    return Mp, D, E, M11


def get_pieces_vector(vec, absor, kept):
    """
    Extract sub-vectors from a given vector corresponding to absorbing and nonboundary states.

    This function partitions a given vector `vec` into two sub-vectors:
    - `v1`: Elements corresponding to absorbing states.
    - `v2`: Elements corresponding to nonboundary (non-absorbing) states.

    Parameters
    ----------
    vec : np.ndarray
        A 1D array representing state values.
    absor : array-like
        Indices of absorbing states.
    kept : array-like
        Indices of non-absorbing (nonboundary) states.

    Raises
    ------
    ValueError
        If `absor` and `kept` contain invalid or overlapping indices.
    ValueError
        If `vec` is not a 1D array.
    ValueError
        If `vec` does not have the expected length matching `absor` + `kept`.

    Returns
    -------
    v1 : np.ndarray
        A column vector (`len(absor) × 1`) containing elements from `vec` at `absor` indices.
    v2 : np.ndarray
        A column vector (`len(kept) × 1`) containing elements from `vec` at `kept` indices.
    """
    # Ensure `vec` is a 1D array
    vec = np.asarray(vec)
    if vec.ndim != 1:
        raise ValueError(f"Expected `vec` to be 1D, but got shape {vec.shape}")

    # Ensure the length of vec matches the expected total size
    NS = len(vec)
    if NS != len(absor) + len(kept):
        raise ValueError(f"Expected `vec` to have length {len(absor) + len(kept)}, but got {NS}")

    check_valid_indices(vec, absor, kept)  # Use a dummy matrix for validation

    # Extract vector components
    v1 = vec[np.array(absor)].reshape(len(absor), 1)
    v2 = vec[np.array(kept)].reshape(len(kept), 1)

    return v1, v2

# Backward compatibility alias for REPPTIS code
set_tau_first_hit_M_distrib = set_tau_first_hit_interface_distrib

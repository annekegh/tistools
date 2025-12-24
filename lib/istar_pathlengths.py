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
def mfpt_to_absorbing_staple_balanced(M, tau1, taum, tau2, absor, kept, 
                                       weights=None, doprint=False):
    """
    Compute MFPT using M-weighted and MC-weighted balanced boundary times.
    
    This version uses both the transition matrix M and MC weights to properly
    average tau1 and tau2 for each transition, giving a physically accurate 
    estimate without arbitrary averaging or remove_initial_m corrections.
    
    Parameters
    ----------
    M : np.ndarray
        Transition matrix of shape (NS, NS).
    tau1 : np.ndarray
        tau1 matrix in interface space, shape (n_intf+1, n_intf).
    taum : np.ndarray
        taum matrix in interface space, shape (n_intf+1, n_intf).
    tau2 : np.ndarray
        tau2 matrix in interface space, shape (n_intf+1, n_intf).
    absor : list
        Indices of absorbing states.
    kept : list
        Indices of non-absorbing states.
    weights : np.ndarray, optional
        MC weights for each (start, end) pair, shape (n_intf+1, n_intf).
        From collect_weights_staple(). If None, uses M as weights only.
    doprint : bool
        Print debug information.
    
    Returns
    -------
    g1, g2, h1, h2 : np.ndarray
        MFPT results.
    
    Notes
    -----
    The path time for each transition is: taum + tau_boundary
    where tau_boundary = weighted_average(tau1, tau2) for that specific transition.
    
    This approach:
    - Avoids the arbitrary choice between tau1 and tau2
    - Uses both measurements weighted by their sampling quality (MC weights)
    - Weights by transition probabilities (M) to reflect actual dynamics
    - No need for remove_initial_m correction
    """
    NS = len(M)
    N = NS // 2
    
    if NS < 3:
        raise ValueError(f"Transition matrix must have at least 3 states, got {NS}.")
    
    check_valid_indices(M, absor, kept)
    
    # Construct tau matrices in MSM space
    tau_boundary = construct_tau_boundary_matrix_staple(tau1, tau2, M, N, weights)
    taum_msm = construct_tau_matrix_staple(taum, N)
    
    # Total time per transition = taum + tau_boundary
    tau_total = taum_msm + tau_boundary
    
    # Compute effective times weighted by transition probabilities
    t1 = np.sum(M[np.ix_(absor, range(NS))] * tau_total[np.ix_(absor, range(NS))], axis=1).reshape(-1, 1)
    tp = np.sum(M[np.ix_(kept, range(NS))] * tau_total[np.ix_(kept, range(NS))], axis=1).reshape(-1, 1)
    
    # Extract submatrices
    Mp = np.take(np.take(M, kept, axis=0), kept, axis=1)
    D = np.take(np.take(M, kept, axis=0), absor, axis=1)
    E = np.take(np.take(M, absor, axis=0), kept, axis=1)
    M11 = np.take(np.take(M, absor, axis=0), absor, axis=1)
    
    a = np.identity(len(Mp)) - Mp
    
    # Solve for G (unconditional MFPT)
    g1 = np.zeros((len(absor), 1))
    g2 = np.linalg.solve(a, np.dot(D, g1) + tp)
    
    # Compute H (conditional MFPT)
    h1 = np.dot(M11, g1) + np.dot(E, g2) + t1
    h2 = np.dot(D, g1) + np.dot(Mp, g2) + tp
    
    if doprint:
        print("Using mfpt_to_absorbing_staple_balanced (M + MC weighted)")
        print(f"tau_boundary sample values:")
        for i in range(min(5, NS)):
            for j in range(min(5, NS)):
                if tau_boundary[i, j] > 0:
                    print(f"  tau_boundary[{i},{j}] = {tau_boundary[i,j]:.4f}")
        print(f"h1: {h1.T}")
        print(f"h2: {h2.T}")
    
    return g1, g2, h1, h2


def mfpt_to_first_last_staple_balanced(M, tau1, taum, tau2, weights=None, doprint=False):
    """
    Compute MFPT to reach state 0 or state -1 using balanced boundary times.
    
    This version averages tau1 and tau2 per transition using both M and MC weights
    for a balanced estimate, avoiding the need to choose between them or apply 
    remove_initial_m corrections.
    
    Parameters
    ----------
    M : np.ndarray
        Transition matrix of shape (NS, NS).
    tau1 : np.ndarray
        tau1 matrix in interface space, shape (n_intf+1, n_intf).
    taum : np.ndarray
        taum matrix in interface space, shape (n_intf+1, n_intf).
    tau2 : np.ndarray
        tau2 matrix in interface space, shape (n_intf+1, n_intf).
    weights : np.ndarray, optional
        MC weights for each (start, end) pair, shape (n_intf+1, n_intf).
        From collect_weights_staple(). If None, uses M as weights only.
    doprint : bool
        Print debug information.
    
    Returns
    -------
    g1, g2, h1, h2 : np.ndarray
        MFPT results. h1[0] is MFPT from [0+] to A or B.
    
    See Also
    --------
    mfpt_to_first_last_staple : Original version using tau2 only.
    collect_weights_staple : Function to collect MC weights from path ensembles.
    """
    NS = len(M)
    if NS < 3:
        raise ValueError(f"Transition matrix must have at least 3 states, got {NS}.")
    
    absor = np.array([0, NS - 1])
    kept = np.array([i for i in range(NS) if i not in absor])
    
    return mfpt_to_absorbing_staple_balanced(
        M, tau1, taum, tau2, absor, kept,
        weights=weights, doprint=doprint
    )


def mfpt_istar_balanced(M, tau_interface, pathensembles, interfaces, doprint=False):
    """
    Compute MFPT for iSTAR model using balanced boundary times with MC weights.
    
    This is a convenience function that:
    1. Collects MC weights from path ensembles
    2. Uses balanced averaging of tau1 and tau2 per transition
    3. Weights by both M (transition probabilities) and MC weights
    
    Parameters
    ----------
    M : np.ndarray
        Transition matrix of shape (2N, 2N) from construct_M_istar.
    tau_interface : dict
        Dictionary with keys 'tau1', 'tau2', and either 'taum' or 'tau'.
        Each contains an (N+1, N) matrix of path lengths indexed by (start+1, end).
        If 'taum' is not present, it will be computed as tau - tau1 - tau2.
    pathensembles : list of :py:class:`.PathEnsemble`
        List of path ensemble objects (for MC weights).
    interfaces : list of float
        Interface values sorted in ascending order.
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
    >>> # Compute MFPT with balanced boundary times
    >>> g1, g2, h1, h2 = mfpt_istar_balanced(M, tau_data, pathensembles, interfaces)
    >>> print(f"MFPT from A: {h1[0]}")
    """
    # Collect MC weights
    weights = collect_weights_staple(pathensembles, interfaces)
    
    # Extract tau components
    tau1 = tau_interface['tau1']
    tau2 = tau_interface['tau2']
    
    # Get taum - either provided directly or compute from tau - tau1 - tau2
    if 'taum' in tau_interface:
        taum = tau_interface['taum']
    elif 'tau' in tau_interface:
        # Compute from provided values to ensure consistency
        taum = tau_interface['tau'] - tau1 - tau2
    else:
        raise ValueError("tau_interface must contain either 'taum' or 'tau'")
    
    return mfpt_to_first_last_staple_balanced(
        M, tau1, taum, tau2, weights=weights, doprint=doprint
    )


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
            if (s_from == 0 and s_to == 1) or (abs(s_to - s_from) < N-2 and (s_from > 2 or s_to > 2)):
                continue
            else:
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
                elif 2 < s_to < N+1:
                    # Target is a left turn state - next turn at interface s_to
                    intf_to = s_to - 2
                elif s_to >= N+1:
                    # Target is a right turn state - next turn at interface (s_to - N)
                    intf_to = s_to - N
                else:
                    continue
                
                tau_msm[s_from, s_to] = tau_interface[intf_from+1, intf_to]
    
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
        Shape should be (N+1, N) where tau_matrix[start+1, end] is the path length
        for transitions from interface start to interface end (start ranges from -1 to N-1).

    Returns
    -------
    np.ndarray
        1D array of shape (NS,) with average tau per MSM state.

    Raises
    ------
    ValueError
        If the dimensions don't match expected values.
    
    Notes
    -----
    The MSM state mapping follows construct_M_istar convention:
    - State 0: [0-] (special state, maps to start=-1)
    - States 1 to N: left turn states at interfaces 0 to N-1
    - States N+1 to 2N-1: right turn states at interfaces 1 to N-1
    - State 2N-1: [B] (reached state B, maps to interface N-1)
    """
    if N < 3:
        raise ValueError("N must be at least 3.")
    if NS != 2 * N:
        raise ValueError(f"NS must be 2 * N = {2 * N}.")
    if tau_matrix.shape != (N+1, N):
        raise ValueError(f"Input tau_matrix must have shape ({N+1}, {N}), got {tau_matrix.shape}.")
    
    # Return (NS,) vector with average tau per starting state
    # Average over destination states (actual weighting should use transition probabilities)
    tau = np.zeros(NS)
    
    for s in range(NS):
        # Map MSM state to interface index, matching construct_tau_matrix_staple logic
        if s == 0:
            intf_from = -1  # [0-] state
        elif 2 <= s < N + 1:
            intf_from = s - 2  # Left turn states
        elif s == NS - 1:
            intf_from = -2  # [B] state - no valid tau
        elif s >= N + 1:
            intf_from = s - N  # Right turn states
        else:
            intf_from = -2  # Invalid
        
        if intf_from >= -1:
            # Average over all possible destination interfaces
            if intf_from < N+1:
                row = tau_matrix[intf_from + 1, intf_from + 1:]
            else:
                row = tau_matrix[intf_from + 1, :intf_from + 1]
            valid = row[row > 0]
            tau[s] = np.mean(valid) if len(valid) > 0 else 0.0
        else:
            tau[s] = 0.0
    
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
        pe.tau1.append(get_tau1_staple(pe.orders[i], start, end, interfaces))
    
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
        pe.tau2.append(get_tau2_staple(pe.orders[i], start, end, interfaces))
    pe.tau2 = np.array(pe.tau2)

    # Get the average tau2 for each (start, end) pair
    for start in range(-1, n_intf):
        for end in range(n_intf):
            mask = accmask & (start_indices == start) & (end_indices == end)
            totweight = np.sum(pe.weights[mask])
            if totweight > 0:
                pe.tau2avg[start+1, end] = np.average(pe.tau2[mask], weights=pe.weights[mask])

def get_tau1_staple(orders, start, end, intfs):
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


def get_tau2_staple(orders, start, end, intfs):
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

def get_tau_staple(orders, start, end, intfs):
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
        pe.tau.append(get_tau_staple(pe.orders[i], start, end, interfaces))
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


def average_tau2_with_tau1(tau1_matrix, tau2_matrix, weights_tau1=None, weights_tau2=None):
    """
    Average tau2 values with corresponding tau1 values from complementary paths.
    
    In iSTAR, tau2[start_A, end_A] measures the time after the last crossing of 
    interface (end_A - 1) for paths going from start_A to end_A. Meanwhile, 
    tau1[start_B, end_B] measures the time to cross interface (start_B + 1) for 
    paths going from start_B to end_B.
    
    When end_A == start_B (and the directions are opposite), these two measurements
    describe the same physical region around an interface, just measured from 
    different path directions. Averaging them can provide a better estimate.
    
    The matching is:
    - tau2[start_A, end_A] with tau1[end_A, end_B] (where start_A < end_A and end_A > end_B)
    - tau2[start_A, end_A] with tau1[end_A, end_B] (where start_A > end_A and end_A < end_B)
    
    Parameters
    ----------
    tau1_matrix : np.ndarray
        Matrix of tau1 values, shape (n_intf+1, n_intf).
        tau1_matrix[start+1, end] is tau1 for paths from interface start to end.
    tau2_matrix : np.ndarray
        Matrix of tau2 values, shape (n_intf+1, n_intf).
        tau2_matrix[start+1, end] is tau2 for paths from interface start to end.
    weights_tau1 : np.ndarray, optional
        Weight matrix for tau1 values, same shape as tau1_matrix.
        If None, equal weights are assumed.
    weights_tau2 : np.ndarray, optional
        Weight matrix for tau2 values, same shape as tau2_matrix.
        If None, equal weights are assumed.
    
    Returns
    -------
    np.ndarray
        Averaged tau2 matrix of shape (n_intf+1, n_intf), where each tau2 value
        is averaged with its corresponding tau1 value (if available).
    
    Notes
    -----
    The physical interpretation:
    - For a path A going right (start < end), tau2_A is the time spent near 
      interface (end - 1) after the last crossing.
    - For a path B going left starting at that same interface (start_B = end_A), 
      tau1_B is the time to cross interface (start_B - 1) = (end_A - 1).
    - Both measure time near the same interface, so averaging can reduce noise.
    
    When tau1 and tau2 regions overlap (negative taum), this averaging helps
    provide a consistent estimate of the boundary region time.
    
    Example
    -------
    >>> # tau2 for path (2 -> 4) measures time near interface 3 after last crossing
    >>> # tau1 for path (4 -> 2) measures time to cross interface 3
    >>> # These describe the same region, so we average them
    >>> tau2_avg = average_tau2_with_tau1(tau1_matrix, tau2_matrix)
    """
    n_intf_plus1, n_intf = tau2_matrix.shape
    
    if weights_tau1 is None:
        weights_tau1 = np.ones_like(tau1_matrix)
    if weights_tau2 is None:
        weights_tau2 = np.ones_like(tau2_matrix)
    
    # Set zero weights where values are zero
    weights_tau1 = weights_tau1 * (tau1_matrix > 0)
    weights_tau2 = weights_tau2 * (tau2_matrix > 0)
    
    tau2_averaged = np.zeros_like(tau2_matrix)
    
    for start_A in range(-1, n_intf):
        for end_A in range(n_intf):
            tau2_val = tau2_matrix[start_A + 1, end_A]
            w2 = weights_tau2[start_A + 1, end_A]
            
            if tau2_val == 0:
                # No tau2 for this (start, end) pair
                continue
            
            # Find the matching tau1: paths that start at end_A and go in opposite direction
            # For paths going right (start_A < end_A), match with paths going left from end_A
            # For paths going left (start_A > end_A), match with paths going right from end_A
            
            start_B = end_A  # The matching path starts where path A ends
            
            # Find all tau1 values from paths starting at start_B going in opposite direction
            matching_tau1 = []
            matching_weights = []
            
            if start_A < end_A:
                # Path A goes right, so path B should go left (end_B < start_B)
                for end_B in range(start_B):
                    if tau1_matrix[start_B + 1, end_B] > 0:
                        matching_tau1.append(tau1_matrix[start_B + 1, end_B])
                        matching_weights.append(weights_tau1[start_B + 1, end_B])
            elif start_A > end_A:
                # Path A goes left, so path B should go right (end_B > start_B)
                for end_B in range(start_B + 1, n_intf):
                    if tau1_matrix[start_B + 1, end_B] > 0:
                        matching_tau1.append(tau1_matrix[start_B + 1, end_B])
                        matching_weights.append(weights_tau1[start_B + 1, end_B])
            
            if len(matching_tau1) > 0:
                # Compute weighted average of matching tau1 values
                matching_tau1 = np.array(matching_tau1)
                matching_weights = np.array(matching_weights)
                avg_tau1 = np.average(matching_tau1, weights=matching_weights)
                w1 = np.sum(matching_weights)
                
                # Average tau2 with the averaged tau1
                total_weight = w1 + w2
                if total_weight > 0:
                    tau2_averaged[start_A + 1, end_A] = (avg_tau1 * w1 + tau2_val * w2) / total_weight
                else:
                    tau2_averaged[start_A + 1, end_A] = tau2_val
            else:
                # No matching tau1 found, keep original tau2
                tau2_averaged[start_A + 1, end_A] = tau2_val
    
    return tau2_averaged


def collect_tau2_averaged_staple(pathensembles, interfaces):
    """
    Collect tau2 values averaged with corresponding tau1 values from complementary paths.
    
    This function first collects tau1, tau2, and their weights from all path ensembles,
    then calls average_tau2_with_tau1 to compute the averaged tau2 matrix.
    
    Parameters
    ----------
    pathensembles : list of :py:class:`.PathEnsemble`
        List of path ensemble objects.
    interfaces : list of float
        Interface values sorted in ascending order.
    
    Returns
    -------
    np.ndarray
        Averaged tau2 matrix of shape (n_intf+1, n_intf).
    
    See Also
    --------
    average_tau2_with_tau1 : The underlying averaging function.
    collect_tau2_staple : Collect tau2 without averaging.
    """
    print("Collect tau2 (averaged with tau1)")
    n_intf = len(interfaces)
    n_ens = len(pathensembles)
    
    # Collect tau1 and tau2 matrices and weights from all ensembles
    tau1_matrices = np.zeros((n_ens, n_intf+1, n_intf))
    tau2_matrices = np.zeros((n_ens, n_intf+1, n_intf))
    weights = np.zeros((n_ens, n_intf+1, n_intf))
    
    for i, pe in enumerate(pathensembles):
        accmask = (pe.flags == "ACC") & (pe.generation != "ld")
        start_indices, end_indices = _compute_start_end_indices(pe, interfaces)
        
        if hasattr(pe, 'tau1avg') and isinstance(pe.tau1avg, np.ndarray):
            tau1_matrices[i] = np.nan_to_num(pe.tau1avg, nan=0.0)
        if hasattr(pe, 'tau2avg') and isinstance(pe.tau2avg, np.ndarray):
            tau2_matrices[i] = np.nan_to_num(pe.tau2avg, nan=0.0)
        
        for start in range(-1, n_intf):
            for end in range(n_intf):
                mask = accmask & (start_indices == start) & (end_indices == end)
                weights[i, start+1, end] = np.sum(pe.weights[mask])
    
    # First compute the combined tau1 and tau2 matrices (weighted across ensembles)
    tau1_combined = np.zeros((n_intf+1, n_intf))
    tau2_combined = np.zeros((n_intf+1, n_intf))
    weights_combined = np.zeros((n_intf+1, n_intf))
    
    for start in range(-1, n_intf):
        for end in range(n_intf):
            total_weight = np.sum(weights[:, start+1, end])
            weights_combined[start+1, end] = total_weight
            if total_weight > 0:
                tau1_combined[start+1, end] = np.sum(
                    tau1_matrices[:, start+1, end] * weights[:, start+1, end]
                ) / total_weight
                tau2_combined[start+1, end] = np.sum(
                    tau2_matrices[:, start+1, end] * weights[:, start+1, end]
                ) / total_weight
    
    # Now average tau2 with tau1
    tau2_averaged = average_tau2_with_tau1(
        tau1_combined, tau2_combined, 
        weights_tau1=weights_combined, weights_tau2=weights_combined
    )
    
    return tau2_averaged


def collect_weights_staple(pathensembles, interfaces):
    """
    Collect total MC weights for each (start, end) interface pair across all ensembles.
    
    Parameters
    ----------
    pathensembles : list of :py:class:`.PathEnsemble`
        List of path ensemble objects.
    interfaces : list of float
        Interface values sorted in ascending order.
    
    Returns
    -------
    np.ndarray
        Weight matrix of shape (n_intf+1, n_intf) where weights[start+1, end]
        is the total weight for paths from interface start to end.
    """
    n_intf = len(interfaces)
    weights_combined = np.zeros((n_intf + 1, n_intf))
    
    for pe in pathensembles:
        accmask = (pe.flags == "ACC") & (pe.generation != "ld")
        start_indices, end_indices = _compute_start_end_indices(pe, interfaces)
        
        for start in range(-1, n_intf):
            for end in range(n_intf):
                mask = accmask & (start_indices == start) & (end_indices == end)
                weights_combined[start + 1, end] += np.sum(pe.weights[mask])
    
    return weights_combined


def construct_tau_boundary_matrix_staple(tau1_matrix, tau2_matrix, M, N, weights=None):
    """
    Construct boundary time matrix using M-weighted averaging of tau1 and tau2.
    
    For each MSM state s_from, the boundary time is computed by:
    1. tau2 from paths that ARRIVED at s_from (weighted by incoming transition probabilities)
    2. tau1 from paths that DEPART from s_from (weighted by outgoing transition probabilities)
    
    These are averaged using M and optionally MC weights, giving a physically accurate estimate.
    
    Parameters
    ----------
    tau1_matrix : np.ndarray
        tau1 values in interface space, shape (n_intf+1, n_intf).
    tau2_matrix : np.ndarray
        tau2 values in interface space, shape (n_intf+1, n_intf).
    M : np.ndarray
        Transition matrix of shape (2N, 2N).
    N : int
        Number of interfaces.
    weights : np.ndarray, optional
        MC weights for each (start, end) pair, shape (n_intf+1, n_intf).
        If None, uses transition matrix M as weights only.
    
    Returns
    -------
    np.ndarray
        Boundary time matrix of shape (2N, 2N) in MSM state space.
    
    Notes
    -----
    The key insight is that tau2 from an arriving path and tau1 from a departing
    path both measure time spent near the same interface (the turn point).
    By averaging them weighted by transition probabilities and MC weights,
    we get a more accurate and balanced estimate of the boundary region time.
    """
    NS = 2 * N
    
    # Convert tau1 and tau2 to MSM space
    tau1_msm = construct_tau_matrix_staple(tau1_matrix, N)
    tau2_msm = construct_tau_matrix_staple(tau2_matrix, N)
    
    # Convert weights to MSM space if provided
    if weights is not None:
        weights_msm = construct_tau_matrix_staple(weights, N)
    else:
        weights_msm = None
    
    tau_boundary = np.zeros((NS, NS))
    
    for s_from in range(NS):
        # Compute weighted average of tau2 for paths arriving at s_from
        # Weight by: M[s_prev, s_from] * weights_msm[s_prev, s_from]
        incoming_M = M[:, s_from]
        tau2_arriving = tau2_msm[:, s_from]
        
        if weights_msm is not None:
            incoming_weights = incoming_M * weights_msm[:, s_from]
        else:
            incoming_weights = incoming_M.copy()
        
        # Mask for valid tau2 values
        valid_tau2 = tau2_arriving > 0
        total_incoming = np.sum(incoming_weights[valid_tau2])
        
        if total_incoming > 0:
            tau2_avg = np.sum(incoming_weights[valid_tau2] * tau2_arriving[valid_tau2]) / total_incoming
            w2_total = total_incoming
        else:
            tau2_avg = 0.0
            w2_total = 0.0
        
        # For each outgoing transition, average tau2_avg with tau1
        for s_to in range(NS):
            tau1_val = tau1_msm[s_from, s_to]
            
            # Weight for tau1: M[s_from, s_to] * weights_msm[s_from, s_to]
            if weights_msm is not None:
                w1 = M[s_from, s_to] * weights_msm[s_from, s_to]
            else:
                w1 = M[s_from, s_to]
            
            # Only count if tau1 is valid
            if tau1_val <= 0:
                w1 = 0.0
            
            total_weight = w1 + w2_total
            if total_weight > 0:
                tau_boundary[s_from, s_to] = (tau1_val * w1 + tau2_avg * w2_total) / total_weight
            elif tau1_val > 0:
                tau_boundary[s_from, s_to] = tau1_val
            elif tau2_avg > 0:
                tau_boundary[s_from, s_to] = tau2_avg
    
    return tau_boundary


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
            tau1_value = get_tau1_staple(pe.orders[i], start, end, interfaces)
            tau2_value = get_tau2_staple(pe.orders[i], start, end, interfaces)
            tau_value = get_tau_staple(pe.orders[i], start, end, interfaces)

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

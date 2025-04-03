"""
Analysis module for interface-based Markov State Models (iSTAR) in transition interface sampling.

This module provides functions to analyze transition paths, calculate memory effects, and
extract global crossing probabilities using the iSTAR approach. The module builds on
tools from repptis_analysis to process path ensembles.

iSTAR (interface-based State TrAnsition netwoRk (lol?)) is a specific approach to analyzing
transition interface sampling (TIS) simulations by constructing a Markov state model
at the interfaces. This allows for efficient calculation of transition probabilities
and rates between different states in complex molecular systems.
"""

from json import load
import numpy as np
from .reading import *
import logging
from .repptis_analysis import *
from .repptis_msm import *
import matplotlib.pyplot as plt
import deeptime as dpt

# Hard-coded rejection flags found in output files
ACCFLAGS, REJFLAGS = set_flags_ACC_REJ() 

# logger
logger = logging.getLogger(__name__)

def global_pcross_msm_star(M, doprint=False):
    """
    Calculate global crossing probabilities from a transition matrix using the iSTAR approach.
    
    This function computes the probability to arrive at state -1 before reaching state 0,
    given that we start at state 0 and leave it, which represents the crossing probability
    from state 0 to -1. It implements the mathematical formalism of the iSTAR approach where
    the system is modeled as a Markov state model with absorbing boundary conditions.
    
    The calculation involves:
    1. Extracting submatrices from the full transition matrix
    2. Solving a linear system of equations (I-Mp)z2 = D·z1
    3. Computing final probability vectors using matrix algebra
    
    Parameters
    ----------
    M : np.ndarray
        Transition matrix representing the Markov state model. This should be a square matrix
        where element M[i,j] represents the probability of transitioning from state i to state j.
    doprint : bool, optional
        If True, prints detailed information about the calculation including eigenvalues,
        submatrices, and verification of the solution. Default is False.
    
    Returns
    -------
    tuple
        A tuple containing the following vectors used in the calculation:
        z1 : np.ndarray
            First solution vector (boundary states)
        z2 : np.ndarray
            Second solution vector (intermediate states)
        y1 : np.ndarray 
            First result vector (boundary states)
        y2 : np.ndarray
            Second result vector (intermediate states)
            
    Notes
    -----
    The mathematical foundation is based on the solution of the committor probability
    in a Markov process, which gives the probability of reaching one state before another
    when starting from an intermediate state.
    """
    # Get number of states
    NS = len(M)
    if NS <= 2:
        raise ValueError("Transition matrix must have at least 3 states for iSTAR analysis")

    # Extract submatrices from the transition matrix
    Mp = M[2:-1, 2:-1]  # Transition probabilities between intermediate states
    a = np.identity(NS-3) - Mp  # I - Mp, used for solving linear system
    
    # Extract other submatrices from the transition matrix
    D = M[2:-1, np.array([0, -1])]  # Transitions from intermediate states to boundary states
    E = M[np.array([0, -1]), 2:-1]  # Transitions from boundary states to intermediate states
    M11 = M[np.array([0, -1]), np.array([0, -1])]  # Transitions between boundary states
    
    # Compute Z vector - solve the linear system
    z1 = np.array([[0], [1]])  # Initial condition
    z2 = np.linalg.solve(a, np.dot(D, z1))  # Solve (I-Mp)z2 = D·z1
    
    # Compute H vector - final probability distribution
    y1 = np.dot(M11, z1) + np.dot(E, z2)  # Transitions to boundary states
    y2 = np.dot(D, z1) + np.dot(Mp, z2)   # Transitions to intermediate states
    
    if doprint:
        # Print eigenvalue analysis
        print("\n=== Eigenvalue Analysis ===")
        vals, vecs = np.linalg.eig(Mp)
        print("Mp eigenvalues:", np.round(vals, 4))
        
        vals, vecs = np.linalg.eig(a)
        print("(I-Mp) eigenvalues:", np.round(vals, 4))
        
        # Print matrix components with clear formatting
        print("\n=== Matrix Components ===")
        print("D (transitions from intermediate to boundary states):")
        print(np.array2string(D, precision=4, suppress_small=True))
        
        print("\nE (transitions from boundary to intermediate states):")
        print(np.array2string(E, precision=4, suppress_small=True))
        
        print("\nM11 (transitions between boundary states):")
        print(np.array2string(M11, precision=4, suppress_small=True))
        
        # Print solution vectors with clear labels
        print("\n=== Solution Vectors ===")
        print("z1 (boundary states solution):")
        print(np.array2string(z1, precision=4))
        
        print("\nz2 (intermediate states solution):")
        print(np.array2string(z2, precision=4))
        
        # Print result vectors with clear labels
        print("\n=== Result Vectors ===")
        print("y1 (boundary states result):")
        print(np.array2string(y1, precision=4))
        
        print("\ny2 (intermediate states result):")
        print(np.array2string(y2, precision=4))
        
        # Verification check
        diff_norm = np.sum((y2-z2)**2)
        print("\n=== Verification ===")
        print(f"||y2-z2||² = {diff_norm:.6e}")
        if diff_norm < 1e-10:
            print("✓ Verification passed: z2 and y2 are identical (as expected)")
        else:
            print("⚠ Verification failed: z2 and y2 differ")
    
    return z1, z2, y1, y2

def construct_M_istar(P, NS, N):
    """
    Construct transition matrix M for the iSTAR model from interface-to-interface transition probabilities.
    
    This function builds a complete transition matrix for the iSTAR model based on the provided
    path transition probabilities between interfaces. The resulting matrix represents the Markov state model
    where states are defined at each interface, with special handling for boundary states.
    
    Parameters
    ----------
    P : np.ndarray
        Array of probabilities for paths between interfaces. P[i,j] represents the 
        probability of a path transitioning from interface i to interface j.
    NS : int
        Dimension of the Markov state model, typically 2*N when N>=2. This defines
        the size of the resulting transition matrix.
    N : int
        Number of interfaces in the system. This determines how the transition matrix
        is structured with specific patterns for interface transitions.
    
    Returns
    -------
    np.ndarray
        Transition matrix M for the iSTAR model with dimensions (NS, NS).
    
    Notes
    -----
    The transition matrix follows a specific structure required for the iSTAR analysis:
    - Special handling for states [0-] and [0*+-]
    - Structured transitions between neighboring interfaces
    - The matrix satisfies Markovian properties where each row sums to 1
    
    Raises
    ------
    ValueError
        If the dimensions of P do not match N, or if NS is not equal to 2*N
    """
    # Validate input dimensions
    if P.shape[0] != N:
        raise ValueError(f"Number of interfaces ({N}) must match P matrix first dimension ({P.shape[0]})")
    if P.shape[1] != N:
        raise ValueError(f"Number of interfaces ({N}) must match P matrix second dimension ({P.shape[1]})")
    if NS != 2*N:
        raise ValueError(f"NS ({NS}) must be equal to 2*N (2*{N}={2*N})")

    # Construct transition matrix
    M = np.zeros((NS, NS))
    
    # States [0-] and [0*+-]
    M[0, 2] = 1            # Transition from state 0 to state 2
    M[2, 0] = P[0, 0]      # Transition from state 2 to state 0
    M[2, N+1:] = P[0, 1:]  # Transitions from state 2 to states N+1 and beyond
    M[1, 0] = 1            # Transition from state 1 to state 0
    M[-1, 0] = 1           # Transition from last state to state 0
    M[N+1:-1, 1] = P[1:-1, 0]  # Transitions from states N+1 and beyond to state 2

    # Set up transitions for other states
    for i in range(1, N):
        M[2+i, N+i:2*N] = P[i, i:]  # Transitions from state 2+i to states N+i and beyond
        if i < N-1:
            M[N+i, 3:2+i] = P[i, 1:i]   # Transitions from state N+i to states 3 through 2+i

    # Nonsampled paths
    if not M[N, -1] >= 0:
        M[N, -1] = 1
    
    # Check rows that sum to zero and set diagonal to 1
    # for i in range(NS):
    #     row_sum = np.sum(M[i])
    #     if row_sum == 0:
    #         M[i, i] = 1  # Set diagonal entry to 1 for rows that sum to zero
            
    # Normalize transition probabilities
        # for i in range(NS):
        #         row_sum = np.sum(M[i])
        # if row_sum > 0:
        #     M[i] = M[i] / row_sum

    # return np.delete(np.delete(M, N, 0), N, 1)
    return M

def get_transition_probs_weights(w_path):
    """
    Calculate local crossing probabilities between interfaces using the recursive probability approach.
    
    This function implements a specific algorithm for calculating transition probabilities
    between interfaces based on weighted path counts. It first computes a matrix q(i,k) representing
    direct transition probabilities, then uses these to derive the full probability matrix p(i,k)
    representing the probability to cross from interface i to interface k through any intermediate path.
    
    Parameters
    ----------
    w_path : dict
        Dictionary containing weighted path counts between interfaces. Each element w_path[i][j][k]
        represents the weighted count of paths that start at ensemble i, from interface j,
        and reach interface k.
    
    Returns
    -------
    np.ndarray
        Matrix p where p[i][k] represents the probability of transitions from interface i
        to interface k through any intermediate path. This accounts for both direct and
        indirect transitions.
    
    Notes
    -----
    The calculation proceeds in two steps:
    1. Compute matrix q where q[i][k] represents the probability of a direct transition
       from interface i to interface k
    2. Compute matrix p using the recursive relation that accounts for all possible paths
       between interfaces
    
    The algorithm handles both forward (i < k) and backward (i > k) transitions differently,
    accounting for the directional nature of interface crossing.
    """
    # Initialize the probability matrix
    n_int = list(w_path.values())[0].shape[0]
    p = np.empty([n_int, n_int])
    q = np.ones([n_int, n_int])
    
    # Calculate q(i,k) - probability to go from i to k via direct transitions
    for i in range(n_int):
        for k in range(n_int):
            counts = np.zeros(2)
            if i == k:
                # Self-transitions
                if i == 0:
                    q[i][k] = 1  # Probability to return to 0 is 1
                    continue
                else:
                    q[i][k] = 0  # No self-transitions for other interfaces
                    continue
            elif i == 0 and k == 1:
                # Special case: transition from 0 to 1
                q[i][k] = (np.sum(w_path[i+1][i][k:])) / (np.sum(w_path[i+1][i][k-1:]))
                continue
            elif i < k:
                # Forward transitions (L→R)
                for pe_i in range(i+1, k+1):
                    if pe_i > n_int-1:
                        break
                    counts += [np.sum(w_path[pe_i][i][k:]), np.sum(w_path[pe_i][i][k-1:])]
                    # print(pe_i-1, i, k, np.sum(w_path[pe_i][i][k:])/np.sum(w_path[pe_i][i][k-1:]), np.sum(w_path[pe_i][i][k-1:]))
            elif i > k:
                # Backward transitions (R→L)
                for pe_i in range(k+2, i+2):
                    if pe_i > n_int-1:
                        break
                    counts += [np.sum(w_path[pe_i][i][:k+1]), np.sum(w_path[pe_i][i][:k+2])]
                    # print(pe_i-1, i, k, np.sum(w_path[pe_i][i][:k+1])/np.sum(w_path[pe_i][i][:k+2]), np.sum(w_path[pe_i][i][:k+2]))

            q[i][k] = counts[0] / counts[1] if counts[1] > 0 else 0
            if 0 in counts:
                print(f"Warning: Zero count detected for q[{i}][{k}] = {q[i][k]}, counts = {counts}")

        # Artificial probability to allow computation of p[i][k] for last interface
        q[-1][-2] = 1
    
    print("\nIntermediate transition probabilities (q matrix):")
    print(np.array2string(q, precision=4, suppress_small=True))
    
    # Calculate final transition probabilities p(i,k) from q values
    for i in range(n_int):
        for k in range(n_int):
            if i < k:
                # Forward transitions
                if k == n_int-1:
                    p[i][k] = np.prod(q[i][i+1:k+1])
                else:
                    p[i][k] = np.prod(q[i][i+1:k+1]) * (1-q[i][k+1])
            elif k < i:
                # Backward transitions
                if k == 0:
                    p[i][k] = np.prod(q[i][k:i])
                else:
                    p[i][k] = np.prod(q[i][k:i]) * (1-q[i][k-1])
                # if i == n_int-1:
                #     p[i][k] = 0
            else:
                # Self-transitions
                if i == 0:
                    p[i][k] = 1-q[i][1]
                else:
                    p[i][k] = 0
    
    print("\nFinal transition probabilities (p matrix):")
    print(np.array2string(p, precision=4, suppress_small=True))
    print("\nLocal crossing probabilities computed successfully")
    
    return p

def get_transition_probs_interm(w_path, weights=None, tr=False):
    """
    Calculate transition probabilities between interfaces using a weighted path averaging approach.
    
    This function provides an alternative method to get_transition_probs_weights() for calculating
    transition probabilities, by explicitly considering intermediate interfaces. For each pair
    of interfaces (i,k), it:
    1. Calculates the probability of reaching each intermediate interface j from i
    2. Calculates the probability of reaching k from each intermediate interface j
    3. Computes a weighted average of these probabilities to determine p(i,k)
    
    Parameters
    ----------
    w_path : dict
        Dictionary containing weighted path counts between interfaces. Each element w_path[i][j][k]
        represents the weighted count of paths that start at ensemble i, from interface j,
        and reach interface k.
    weights : np.ndarray, optional
        Path weights for each transition. If None, equal weights are used. These weights
        determine the importance of each path in the calculation.
    tr : bool, optional
        If True, time-reversal symmetry is enforced, making p(i,j) = p(j,i). This is useful
        when the system should obey detailed balance. Default is False.
    
    Returns
    -------
    np.ndarray
        Matrix p where p[i][k] represents the probability of transitions from interface i
        to interface k, calculated using the weighted averaging approach.
    
    Notes
    -----
    This method differs from get_transition_probzz() in that it explicitly considers all possible
    intermediate interfaces and computes weighted averages rather than using a recursive formula.
    It may provide more accurate results when the path statistics are limited or when there are
    complex transition mechanisms.
    """
    # Get shape parameters
    sh = list(w_path.values())[0].shape
    p = np.empty([sh[0], sh[0]])
    
    # Calculate transition probabilities for each pair of interfaces
    for i in range(sh[0]):
        for k in range(sh[1]):
            if i == k:
                # Self-transitions
                if i == 0:
                    p[i][k] = np.sum(w_path[i+1][i][k]) / np.sum(w_path[i+1][i][i:]) if np.sum(w_path[i+1][i][i:]) != 0 else 0
                else:
                    p[i][k] = 0
            elif i < k:
                # Forward transitions (i → k where i < k)
                p_reachedj = np.empty(k-i+1)
                p_jtillend = np.empty(k-i+1)
                w_reachedj = np.empty(k-i+1)
                w_jtillend = np.empty(k-i+1)
                
                # Calculate probabilities for each intermediate interface j
                for j in range(i, k+1):
                    p_reachedj[j-i] = np.sum(w_path[i+1][i][j:]) / np.sum(w_path[i+1][i][i:]) if np.sum(w_path[i+1][i][i:]) != 0 else 0
                    w_reachedj[j-i] = np.sum(w_path[i+1][i][i:])
                    
                    if j < sh[0]-1:
                        p_jtillend[j-i] = np.sum(w_path[j+1][i][k]) / np.sum(w_path[j+1][i][i:]) if np.sum(w_path[j+1][i][i:]) != 0 else 0
                        w_jtillend[j-i] = np.sum(w_path[j+1][i][i:])
                    else: 
                        p_jtillend[j-i] = 1
                        w_jtillend[j-i] = 1
                
                # Debug output: Transition probability calculation details
                print(f"\n--- Transition from i={i} to k={k} (distance: {k-i}) ---")
                print(f"Probabilities of reaching intermediate interfaces:")
                print(f"  P_i(j reached):  {np.array2string(p_reachedj, precision=4)}")
                print(f"  P_j(k):          {np.array2string(p_jtillend, precision=4)}")
                print(f"\nCombined probabilities:")
                print(f"  P_i(k) raw:      {np.array2string(p_reachedj*p_jtillend, precision=4)}")
                print(f"  Weights:         {np.array2string(w_reachedj*w_jtillend, precision=4)}")
                
                weighted_avg = np.average(p_reachedj * p_jtillend, weights=w_reachedj*w_jtillend) if np.sum(w_reachedj*w_jtillend) != 0 else 0
                normal_avg = np.average(p_reachedj * p_jtillend)
                
                print(f"\nFinal probabilities:")
                print(f"  P_i(k) weighted: {weighted_avg:.4f}")
                print(f"  P_i(k) normal:   {normal_avg:.4f}")
                
                # Calculate weighted average of transition probability
                p[i][k] = np.average(p_reachedj * p_jtillend, weights=w_reachedj*w_jtillend) if np.sum(w_reachedj*w_jtillend) != 0 else 0                
            elif i > k:
                # Backward transitions (i → k where i > k)
                p_reachedj = np.empty(i-k+1)
                p_jtillend = np.empty(i-k+1)
                w_reachedj = np.empty(i-k+1)
                w_jtillend = np.empty(i-k+1)
                
                # Calculate probabilities for each intermediate interface j
                for j in range(k, i+1):
                    if i < sh[0]-1:
                        p_reachedj[j-k] = np.sum(w_path[i+1][i][:j+1]) / np.sum(w_path[i+1][i][:i+1]) if np.sum(w_path[i+1][i][:i+1]) != 0 else 0
                        p_jtillend[j-k] = np.sum(w_path[j+1][i][k]) / np.sum(w_path[j+1][i][:i+1]) if np.sum(w_path[j+1][i][:i+1]) != 0 else 0
                        w_reachedj[j-k] = np.sum(w_path[i+1][i][:i+1])
                        w_jtillend[j-k] = np.sum(w_path[j+1][i][:i+1])
                    else: 
                        p_reachedj[j-k] = 0
                        p_jtillend[j-k] = 0
                        w_reachedj[j-k] = 0
                        w_jtillend[j-k] = 0
                
                # Debug output: Transition probability calculation details
                print(f"\n--- Transition from i={i} to k={k} (distance: {i-k}) ---")
                print(f"Probabilities of reaching intermediate interfaces:")
                print(f"  P_i(j reached):  {np.array2string(p_reachedj, precision=4)}")
                print(f"  P_j(k):          {np.array2string(p_jtillend, precision=4)}")
                print(f"\nCombined probabilities:")
                print(f"  P_i(k) raw:      {np.array2string(p_reachedj*p_jtillend, precision=4)}")
                print(f"  Weights:         {np.array2string(w_reachedj*w_jtillend, precision=4)}")
                
                weighted_avg = np.average(p_reachedj * p_jtillend, weights=w_reachedj*w_jtillend) if np.sum(w_reachedj*w_jtillend) != 0 else 0
                normal_avg = np.average(p_reachedj * p_jtillend)
                
                print(f"\nFinal probabilities:")
                print(f"  P_i(k) weighted: {weighted_avg:.4f}")
                print(f"  P_i(k) normal:   {normal_avg:.4f}")
                
                # Calculate weighted average of transition probability
                p[i][k] = np.average(p_reachedj * p_jtillend, weights=w_reachedj*w_jtillend) if np.sum(w_reachedj*w_jtillend) != 0 else 0

    print("Local crossing probabilities computed")
    return p

def get_simple_probs(w_path):
    """
    Calculate simplified transition probabilities between interfaces using direct path counting.
    
    This method provides a more straightforward approach than get_transition_probs() or 
    get_transition_probzz() by directly calculating the ratio of paths going from i to k
    over all paths starting from i, with minimal pre-processing or weighting schemes.
    
    The calculation is tailored to specific interface relationship patterns:
    - For i < k (forward transitions): Different handling based on interface position
    - For k < i (backward transitions): Special case for the last interface
    - For self-transitions (i = k): Special handling for the first interface
    
    Parameters
    ----------
    w_path : dict
        Dictionary containing weighted path counts between interfaces. Each element w_path[i][j][k]
        represents the weighted count of paths that start at ensemble i, from interface j,
        and reach interface k.
    
    Returns
    -------
    np.ndarray
        Matrix p where p[i][k] represents the simplified transition probability from interface i
        to interface k, based on direct path counting.
    
    Notes
    -----
    This function is useful when:
    - A quick estimate of transition probabilities is needed
    - The simulation data is limited and cannot support more complex analyses
    - The system is relatively simple and does not require detailed treatment of intermediate paths
    
    The calculation explicitly handles special cases for boundary interfaces.
    """
    n_int = list(w_path.values())[0].shape[0]

    # Initialize probability matrix
    p = np.empty([n_int, n_int])

    # Calculate transition probabilities for each pair of interfaces
    for i in range(n_int):
        for k in range(n_int):
            if i < k:
                # Forward transitions
                if i == 0 or i >= n_int-2:
                    if k == n_int-1:
                        p[i][k] = np.sum(w_path[i+1][i][k:]) / np.sum(w_path[i+1][i][i:])
                    else:
                        p[i][k] = (w_path[i+1][i][k]) / np.sum(w_path[i+1][i][i:])
                else:
                    # Use both i+1 and i+2 ensembles for calculation
                    p[i][k] = (w_path[i+1][i][k] + w_path[i+2][i][k]) / (np.sum(w_path[i+1][i][i:]) + np.sum(w_path[i+2][i][i:]))
            elif k < i:
                # Backward transitions
                if i == n_int-1:
                    p[i][k] = 0
                else:
                    p[i][k] = (w_path[i+1][i][k] + w_path[i][i][k]) / (np.sum(w_path[i+1][i][:i]) + np.sum(w_path[i][i][:i]))
            else:
                # Self-transitions
                if i == 0:
                    p[i][k] = w_path[i+1][i][k] / np.sum(w_path[i+1][i][i:])
                else:
                    p[i][k] = 0
                    
    # Print the matrix p in a more readable format
    print("\nLocal crossing probability matrix:")
    print(np.array2string(p, precision=4, suppress_small=True))
    print("\nLocal crossing probabilities computed successfully")
    
    return p

def get_summed_probs(pes, interfaces, weights=None, dbens=False):
    """
    Calculate transition probabilities by summing path counts across all ensembles.
    
    This function provides a global view of transition probabilities by aggregating
    path statistics from multiple path ensembles. It processes each path ensemble 
    to extract path counts for each interface transition, then combines them to 
    compute overall transition probabilities.
    
    The function handles:
    - Forward transitions (i < k): Paths that cross from lower to higher interfaces
    - Backward transitions (k < i): Paths that cross from higher to lower interfaces
    - Special cases for boundary interfaces
    - Self-transitions (i = k)
    
    Parameters
    ----------
    pes : list
        List of :py:class:`.PathEnsemble` objects. Each object represents a collection of paths
        from a specific simulation or interface.
    interfaces : list
        List of interface positions, typically lambda values that define the interfaces
        in order parameter space.
    weights : dict, optional
        Dictionary of weights for each path ensemble. If None, weights are calculated
        from the acceptance/rejection statistics of each ensemble.
    dbens : bool, optional
        If True, enables detailed debugging output for each ensemble, showing path counts
        and probability calculations. Default is False.
    
    Returns
    -------
    np.ndarray
        Matrix p where p[i][k] represents the probability of transitions from interface i
        to interface k, calculated by summing path counts across all ensembles.
    
    Notes
    -----
    This method is especially useful for:
    - Combining data from multiple simulations or replicas
    - Analyzing systems with wide interface spacing
    - Getting a comprehensive picture of transition behavior across the entire order parameter space
    
    The function identifies and reports transitions with zero weights, which can indicate
    sampling problems or path classifications that need attention.
    """
    masks = {}
    w_path = {}

    # Process each path ensemble
    for i, pe in enumerate(pes):
        # Get the lmr masks, weights, ACCmask, and loadmask of the paths
        masks[i] = get_lmr_masks(pe)
        if weights is None:
            w, ncycle_true = get_weights(pe.flags, ACCFLAGS, REJFLAGS, verbose=False)
            assert ncycle_true == pe.ncycle
        else:
            w = weights[i]
            
        accmask = get_flag_mask(pe, "ACC")
        loadmask = get_generation_mask(pe, "ld")
        msg = f"Ensemble {pe.name[-3:]} has {len(w)} paths.\n The total "+\
                    f"weight of the ensemble is {np.sum(w)}\nThe total amount of "+\
                    f"accepted paths is {np.sum(accmask)}\nThe total amount of "+\
                    f"load paths is {np.sum(loadmask)}"
        logger.debug(msg)

        # Initialize weight path matrix for this ensemble
        w_path[i] = np.zeros([len(interfaces), len(interfaces)])
        
        # Calculate weights for each transition i→k
        for j in range(len(interfaces)):
            for k in range(len(interfaces)):
                if j == k:
                    # Self-transitions
                    if j == 0:
                        if i == 1:
                            start_cond = pe.lambmins <= interfaces[j]
                            end_cond = np.logical_and(pe.lambmaxs >= interfaces[1], pe.lambmaxs <= interfaces[2])
                            w_path[i][j][k] = np.sum(select_with_masks(w, [masks[i]["LML"], accmask, ~loadmask])) +\
                                np.sum(select_with_masks(w, [start_cond, end_cond, accmask, ~loadmask]))
                        elif i == 2:
                            w_path[i][j][k] = np.sum(select_with_masks(w, [masks[i]["LML"], accmask, ~loadmask]))
                        continue
                    else:
                        w_path[i][j][k] = 0  
                        continue
                elif j < k:
                    # Forward transitions (j → k where j < k)
                    dir_mask = pe.dirs == 1
                    if j == 0:
                        start_cond = pe.lambmins <= interfaces[j]
                        if k == 1:
                            continue
                    else: 
                        start_cond = np.logical_and(pe.lambmins <= interfaces[j], pe.lambmins >= interfaces[j-1])
                    if k == len(interfaces)-1:
                        end_cond = pe.lambmaxs >= interfaces[k]
                    else: 
                        end_cond = np.logical_and(pe.lambmaxs >= interfaces[k], pe.lambmaxs <= interfaces[k+1])
                
                    w_path[i][j][k] = np.sum(select_with_masks(w, [start_cond, end_cond, dir_mask, accmask, ~loadmask]))
                else:
                    # Backward transitions (j → k where j > k)
                    dir_mask = pe.dirs == -1
                    if k == 0:
                        start_cond = pe.lambmins <= interfaces[k]
                        if j == 1:
                            continue
                    else: 
                        start_cond = np.logical_and(pe.lambmins <= interfaces[k], pe.lambmins >= interfaces[k-1])
                    if j == len(interfaces)-1:
                        end_cond = pe.lambmaxs >= interfaces[j]
                    else: 
                        end_cond = np.logical_and(pe.lambmaxs >= interfaces[j], pe.lambmaxs <= interfaces[j+1])

                    w_path[i][j][k] = np.sum(select_with_masks(w, [start_cond, end_cond, dir_mask, accmask, ~loadmask]))
                    
        print(f"Sum weights ensemble i: ", np.sum(w_path[i]))

    # Initialize the probability matrix
    p = np.zeros([len(interfaces), len(interfaces)])
    
    # Calculate probabilities from summed weights across all ensembles
    for i in range(len(interfaces)):
        for k in range(len(interfaces)):
            counts = np.zeros(2)
            if i < k:
                # Forward transitions
                for pe_i in range(i+1, k+2):
                    if pe_i > len(interfaces)-1:
                        break
                    counts += [w_path[pe_i][i][k], np.sum(w_path[pe_i][i][i:])]
            elif k < i:
                # Backward transitions
                if i == len(interfaces)-1:
                    p[i][k] = 0
                    continue
                for pe_i in range(min(k+1, i+1), i+2):
                    if pe_i > len(interfaces)-1:
                        break
                    counts += [w_path[pe_i][i][k], np.sum(w_path[pe_i][i][:i+1])]
            else:
                # Self-transitions
                if i == 0:
                    counts += [w_path[1][i][k], np.sum(w_path[1][i][i:])]
                else:
                    counts[1] += 1
                    
            p[i][k] = counts[0] / counts[1] if counts[1] != 0 else 0
            if counts[1] == 0:
                print("No weights for this transition: ", p[i][k], counts, i, k)
                
    # Print the matrix p in a more readable format
    print("\nLocal crossing probability matrix:")
    print(np.array2string(p, precision=4, suppress_small=True))
    print("\nLocal crossing probabilities computed successfully")
    
    return p

def compute_weight_matrices(pes, interfaces, n_int=None, tr=False, weights=None):
    """
    Compute weight matrices for transition analysis across all path ensembles.
    
    This function processes each path ensemble to calculate weight matrices that represent
    the effective number of transitions between each pair of interfaces. These weights 
    form the foundation for subsequent probability calculations and transition analysis.
    
    The function applies sophisticated filtering based on:
    - Path direction (forward/backward)
    - Special treatment for interface boundary conditions
    - Path types (LML, LMR, RML, RMR)
    - Acceptance/rejection status and loading conditions
    
    Parameters
    ----------
    pes : list
        List of :py:class:`.PathEnsemble` objects. Each object represents a collection of paths
        from a specific simulation or interface.
    interfaces : list
        List of interface positions, typically lambda values that define the interfaces
        in order parameter space.
    n_int : int, optional
        Number of interfaces to consider. If None, uses the length of pes. This allows
        analysis of a subset of interfaces if desired.
    tr : bool, optional
        If True, enforces time-reversal symmetry in the weights by symmetrizing
        the weight matrices. Default is False.
    weights : dict, optional
        Dictionary of pre-computed weights. If None, weights are calculated using
        the staple method that accounts for path multiplicities. 
        
    Returns
    -------
    dict
        Dictionary X where X[i] is the weight matrix for ensemble i. Each element X[i][j][k]
        represents the weighted count of paths starting from interface j and reaching
        interface k within ensemble i.
        
    Notes
    -----
    The weight calculation incorporates:
    - Path direction detection (using pe.dirs)
    - Specific handling of different path types (LML, LMR, RML, RMR)
    - Interface boundary conditions and special cases
    - High acceptance weight adjustment for certain path types
    
    This function is fundamental to the iSTAR analysis pipeline, as the weight matrices
    serve as input for all subsequent probability calculations.
    """
    masks = {}
    w_path = {}
    X = {}
    if n_int is None:
        n_int = len(pes)
        
    # Process each path ensemble
    for i, pe in enumerate(pes):
        # Get the lmr masks, weights, ACCmask, and loadmask of the paths
        masks[i] = get_lmr_masks(pe)
        if weights is None:
            # Calculate weights using the staple method
            w = get_weights_staple(i, pe.flags, pe.generation, pe.lmrs, n_int, ACCFLAGS, REJFLAGS, verbose=True)
        else:
            w = weights[i]
            
        accmask = get_flag_mask(pe, "ACC")
        loadmask = get_generation_mask(pe, "ld")
        msg = f"Ensemble {pe.name[-3:]} has {len(w)} paths.\n The total "+\
                    f"weight of the ensemble is {np.sum(w)}\nThe total amount of "+\
                    f"accepted paths is {np.sum(accmask)}\nThe total amount of "+\
                    f"load paths is {np.sum(loadmask)}"
        logger.debug(msg)
        
        # Initialize weight matrices
        w_path[i] = np.zeros([len(interfaces), len(interfaces)])
        X[i] = np.zeros([len(interfaces), len(interfaces)])

        # Calculate weights for each transition j→k
        for j in range(len(interfaces)):
            for k in range(len(interfaces)):
                if j == k:
                    # Self-transitions
                    if i == 1 and j == 0:
                        w_path[i][j][k] = np.sum(select_with_masks(w, [masks[i]["LML"], accmask, ~loadmask]))
                        continue
                    else:
                        w_path[i][j][k] = 0
                elif j < k:
                    # Forward transitions (j → k where j < k)
                    if j == 0 and k == 1:
                        if i != 2:
                            dir_mask = pe.dirs == 1
                        elif i == 2:
                            dir_mask = masks[i]["LML"]
                    elif j == len(interfaces)-2 and k == len(interfaces)-1:
                        dir_mask = masks[i]["RMR"]
                    else:
                        dir_mask = pe.dirs == 1
                        # dir_mask = np.logical_or(pe.dirs == 1, masks[i]["LML"])                        
                    if j == 0:
                        start_cond = pe.lambmins <= interfaces[j]
                    else: 
                        start_cond = np.logical_and(pe.lambmins <= interfaces[j], pe.lambmins >= interfaces[j-1])
                    if k == len(interfaces)-1:
                        end_cond = pe.lambmaxs >= interfaces[k]
                    else: 
                        end_cond = np.logical_and(pe.lambmaxs >= interfaces[k], pe.lambmaxs <= interfaces[k+1])
                
                    w_path[i][j][k] = np.sum(select_with_masks(w, [start_cond, end_cond, dir_mask, accmask, ~loadmask]))
                else:
                    # Backward transitions (j → k where j > k)
                    if j == 1 and k == 0:
                        if i != 2:
                            dir_mask = pe.dirs == -1
                            # dir_mask = np.logical_or(pe.dirs == -1, masks[i]["RMR"])
                        elif i == 2:
                            dir_mask = masks[i]["LML"]
                    elif j == len(interfaces)-2 and k == len(interfaces)-1:
                        dir_mask = masks[i]["RMR"]
                    else:
                        dir_mask = pe.dirs == -1
                        # dir_mask = np.logical_or(pe.dirs == -1, masks[i]["RMR"])                        
                    if k == 0:
                        start_cond = pe.lambmins <= interfaces[k]
                    else: 
                        start_cond = np.logical_and(pe.lambmins <= interfaces[k], pe.lambmins >= interfaces[k-1])
                    if j == len(interfaces)-1:
                        end_cond = pe.lambmaxs >= interfaces[j]
                    else: 
                        end_cond = np.logical_and(pe.lambmaxs >= interfaces[j], pe.lambmaxs <= interfaces[j+1])

                    w_path[i][j][k] = np.sum(select_with_masks(w, [start_cond, end_cond, dir_mask, accmask, ~loadmask]))

        print(f"Sum weights ensemble {i}: ", np.sum(w_path[i]))

    X = w_path
    for i in range(len(pes)):
        if tr:
            if i == 2 and X[i][0, 1] == 0:     # In [1*] all LML paths are classified as 1 -> 0 (for now).
                X[i][1, 0] *= 2     # Time reversal needs to be adjusted to compensate for this
            elif i == len(interfaces)-1 and X[i][-1, -2] == 0:
                X[i][-2, -1] *= 2
            X[i] = (X[i] + X[i].T) / 2.0   # Properly symmetrize the matrix
    return X

def compute_weight_matrix(pe, pe_id, interfaces, tr=False, weights=None):
    """
    Compute weight matrix for a single path ensemble.
    
    This function is a specialized version of compute_weight_matrices that processes
    a single path ensemble rather than a list. It calculates a matrix where each element
    represents the weighted count of paths transitioning between interfaces within the
    given ensemble.
    
    Parameters
    ----------
    pe : :py:class:`.PathEnsemble`
        :py:class:`.PathEnsemble` object representing the collection of paths to analyze.
    pe_id : int
        ID of the path ensemble, which determines specific handling rules and
        filtering conditions based on the ensemble's position in the interface sequence.
    interfaces : list
        List of interface positions, typically lambda values that define the interfaces
        in order parameter space.
    tr : bool, optional
        If True, enforces time-reversal symmetry in the weights by symmetrizing
        the weight matrix. Default is False.
    weights : np.ndarray, optional
        Pre-computed weights for the paths in this ensemble. If None, weights are 
        calculated using the staple method within the function.
    
    Returns
    -------
    np.ndarray
        Weight matrix X_path where each element X_path[j][k] represents the weighted
        count of paths starting from interface j and reaching interface k within
        the specified ensemble.
        
    Notes
    -----
    This function applies specific filtering and classification rules based on:
    - Path direction (using pe.dirs)
    - Special cases for boundary ensembles (e.g., pe_id == 1)
    - Interface crossing conditions for path endpoints
    - Acceptance/rejection status and loading conditions
    
    The weight matrix serves as the foundation for calculating transition probabilities
    in subsequent analysis steps.
    """
    # Get the lmr masks, weights, ACCmask, and loadmask of the paths
    masks = get_lmr_masks(pe)
    if weights is None:
        w = get_weights_staple(pe_id, pe.flags, pe.generation, pe.lmrs, len(interfaces), ACCFLAGS, REJFLAGS, verbose=False)
    else:
        w = weights
    accmask = get_flag_mask(pe, "ACC")
    loadmask = get_generation_mask(pe, "ld")
    msg = f"Ensemble {pe.name[-3:]} has {len(w)} paths.\n The total "+\
                f"weight of the ensemble is {np.sum(w)}\nThe total amount of "+\
                f"accepted paths is {np.sum(accmask)}\nThe total amount of "+\
                f"load paths is {np.sum(loadmask)}"
    logger.debug(msg)

    X_path = np.zeros([len(interfaces), len(interfaces)])

    for j in range(len(interfaces)):
        for k in range(len(interfaces)):
            if j == k:
                # Self-transitions
                if pe_id == 1 and j == 0:
                    X_path[j][k] = np.sum(select_with_masks(w, [masks["LML"], accmask, ~loadmask]))
                    continue
                else:
                    X_path[j][k] = 0  
            elif j < k:
                # Forward transitions (j → k where j < k)
                if j == 0 and k == 1:
                    if pe_id != 2:
                        dir_mask = pe.dirs == 1
                    elif pe_id == 2:
                        dir_mask = masks["LML"]
                elif j == len(interfaces)-2 and k == len(interfaces)-1:
                    dir_mask = masks["RMR"]
                else:
                    dir_mask = pe.dirs == 1
                    # dir_mask = np.logical_or(pe.dirs == 1, masks["LML"])                        
                if j == 0:
                    start_cond = pe.lambmins <= interfaces[j]
                else: 
                    start_cond = np.logical_and(pe.lambmins <= interfaces[j], pe.lambmins >= interfaces[j-1])
                if k == len(interfaces)-1:
                    end_cond = pe.lambmaxs >= interfaces[k]
                else: 
                    end_cond = np.logical_and(pe.lambmaxs >= interfaces[k], pe.lambmaxs <= interfaces[k+1])
            
                X_path[j][k] = np.sum(select_with_masks(w, [start_cond, end_cond, dir_mask, accmask, ~loadmask]))
            else:
                # Backward transitions (j → k where j > k)
                if j == 1 and k == 0:
                    if pe_id != 2:
                        dir_mask = pe.dirs == -1
                        # dir_mask = np.logical_or(pe.dirs == -1, masks["RMR"])
                    elif pe_id == 2:
                        dir_mask = masks["LML"]
                elif j == len(interfaces)-2 and k == len(interfaces)-1:
                    dir_mask = masks["RMR"]
                else:
                    dir_mask = pe.dirs == -1
                    # dir_mask = np.logical_or(pe.dirs == -1, masks["RMR"])                        
                if k == 0:
                    start_cond = pe.lambmins <= interfaces[k]
                else: 
                    start_cond = np.logical_and(pe.lambmins <= interfaces[k], pe.lambmins >= interfaces[k-1])
                if j == len(interfaces)-1:
                    end_cond = pe.lambmaxs >= interfaces[j]
                else: 
                    end_cond = np.logical_and(pe.lambmaxs >= interfaces[j], pe.lambmaxs <= interfaces[j+1])

                X_path[j][k] = np.sum(select_with_masks(w, [start_cond, end_cond, dir_mask, accmask, ~loadmask]))

    print(f"Sum weights ensemble {pe_id}: {np.sum(X_path):.4f}")

    # Apply time-reversal symmetry if requested
    if tr:
        if pe_id == 2 and X_path[0, 1] == 0:     # In [1*] all LML paths are classified as 1 -> 0 (for now)
            X_path[1, 0] *= 2     # Time reversal needs to be adjusted to compensate for this
        elif pe_id == len(interfaces)-1 and X_path[-1, -2] == 0:
            X_path[-2, -1] *= 2
        X_path = (X_path + X_path.T) # Properly symmetrize the matrix
    
    return X_path


def get_weights_staple(pe_i, flags, gen, ptypes, n_pes, ACCFLAGS, REJFLAGS, verbose=True):
    """
    Calculate weights for each trajectory in a path ensemble using the staple method.
    
    The staple method is a specific approach to weighting trajectories in TIS simulations
    that accounts for the statistical multiplicity of each path. This method:
    1. Processes paths sequentially, considering acceptance/rejection status
    2. Accumulates weights for rejected paths into the previous accepted path
    3. Applies high-acceptance (HA) factor of 2 for certain LML/RMR paths to ensure proper weighting
    
    Parameters
    ----------
    pe_i : int
        Index of the path ensemble, which determines specific handling rules.
    flags : list
        List of flags indicating the status of each trajectory ('ACC' for accepted, 
        or values in REJFLAGS for rejected).
    gen : list
        List of generation types for each trajectory (e.g., 'sh' for shooting, 'wf' for waste-recycling).
    ptypes : list
        List of path types for each trajectory (e.g., "LML", "LMR", "RMR", "RML").
    n_pes : int
        Total number of path ensembles in the system.
    ACCFLAGS : list
        List of flags that indicate acceptance of a trajectory.
    REJFLAGS : list
        List of flags that indicate rejection of a trajectory.
    verbose : bool, optional
        If True, prints detailed information about the weight calculation including
        counts of accepted, rejected, and omitted paths. Default is True.
    
    Returns
    -------
    np.ndarray
        Array of weights for each trajectory, where:
        - Accepted trajectories get weights based on their statistical multiplicity
        - Rejected trajectories get weight 0
        - Certain path types get a high-acceptance factor of 2
        
    Notes
    -----
    The staple method is particularly important for:
    - Correctly accounting for the statistical weight of each path
    - Properly handling shooting moves that may have high acceptance rates
    - Ensuring statistical correctness when combining data from different ensembles
    
    Weights are crucial for unbiased estimation of transition probabilities and
    rate constants from path sampling simulations.
    
    Raises
    ------
    AssertionError
        If input arrays have inconsistent lengths or if unexpected path flags are found
    """
    ntraj = len(flags)
    
    # Validate input dimensions
    if not len(flags) == len(gen) == len(ptypes):
        raise ValueError(f"Input arrays must have same length: flags ({len(flags)}), gen ({len(gen)}), ptypes ({len(ptypes)})")
    
    # Validate first path is accepted
    if flags[0] != 'ACC':
        raise ValueError(f"First trajectory must be accepted (ACC), but found flag: {flags[0]}")
    
    weights = np.zeros(ntraj, int)

    accepted = 0
    rejected = 0
    omitted = 0

    acc_w = 0
    acc_index = 0
    tot_w = 0
    prev_ha = 1
    assert flags[0] == 'ACC'
    for i, fgp in enumerate(zip(flags, gen, ptypes)):
        flag, gg, ptype = fgp
        if flag in ACCFLAGS:
            # Store previous trajectory with accumulated weight
            weights[acc_index] = prev_ha * acc_w
            tot_w += prev_ha * acc_w
            # Info for new trajectory
            acc_index = i
            acc_w = 1
            accepted += 1
            if gg == 'sh' and \
                ((pe_i == 2 and ptype == "RMR") or\
                (pe_i == n_pes-1 and ptype == "LML") or\
                (2 < pe_i < n_pes-1 and ptype in ["LML", "RMR"])):
                prev_ha = 2
            else:
                prev_ha = 1
        elif flag in REJFLAGS:
            acc_w += 1    # Weight of previous accepted trajectory increased
            rejected += 1
        else:
            omitted += 1

    # At the end: store the last accepted trajectory with its weight
    weights[acc_index] = prev_ha * acc_w
    tot_w += prev_ha * acc_w

    if verbose:
        print("weights:")
        print("accepted     ", accepted)
        print("rejected     ", rejected)
        print("omitted      ", omitted)
        print("total trajs  ", ntraj)
        print("total weights", np.sum(weights))

    assert omitted == 0

    return weights


def plot_rv_star(pes, interfaces, numberof):
    """
    Plot representative trajectories for the iSTAR model on a phase space diagram.
    
    This function visualizes selected trajectories from path ensembles, showing how
    paths traverse the phase space between interfaces. It's useful for understanding
    the typical behavior of transition paths in the system.
    
    Parameters
    ----------
    pes : list
        List of :py:class:`.PathEnsemble` objects containing trajectory information.
    interfaces : list
        List of interface positions to be displayed as vertical lines.
    numberof : int
        Number of trajectories to plot for each ensemble.
        
    Notes
    -----
    The function:
    1. Selects trajectories that go from the first to the last interface
    2. Draws vertical lines at each interface position
    3. Plots position vs. momentum for each selected trajectory
    4. Creates a legend identifying each ensemble
    
    This visualization is particularly valuable for:
    - Understanding the typical path behavior in phase space
    - Identifying differences in path mechanisms between ensembles
    - Verifying that paths appropriately cross interfaces
    """
    cycle_nrs = {}
    fig, ax = plt.subplots()

    for i, pe in enumerate(pes):
        accmask = get_flag_mask(pe, "ACC")
        loadmask = get_generation_mask(pe, "ld")
        start_cond = pe.lambmins <= interfaces[0]
        end_cond = pe.lambmaxs >= interfaces[-1]
        dir_mask = pe.dirs == 1

        cycle_nrs[i] = select_with_masks(pe.cyclenumbers, [start_cond, end_cond, dir_mask, accmask, ~loadmask])

        ax.vlines(interfaces, -4, 4, color='black')
        linecolor = None
        count = 0 
        while True and i > 0:
            id = np.random.choice(cycle_nrs[i])
            if np.all(pe.orders[id][:, 1] >= 0):
                lines = ax.plot(pe.orders[id][:, 0], pe.orders[id][:, 1], color=linecolor, label=i)
                linecolor = lines[0].get_color()
                count += 1
            if count == numberof:
                break
    fig.legend()
    fig.show()

def plot_rv_repptis(pes, interfaces, numberof):
    """
    Plot representative trajectories for the REPPTIS (Replica Exchange Path TIS) model.
    
    This function is similar to plot_rv_star() but specifically designed for REPPTIS
    simulations, selecting paths based on REPPTIS-specific criteria.
    
    Parameters
    ----------
    pes : list
        List of :py:class:`.PathEnsemble` objects containing trajectory information from REPPTIS simulations.
    interfaces : list
        List of interface positions to be displayed as vertical lines.
    numberof : int
        Number of trajectories to plot for each ensemble.
        
    Notes
    -----
    The function:
    1. Selects trajectories based on REPPTIS-specific criteria for interface crossing
    2. Draws vertical lines at each interface position
    3. Plots position vs. momentum for each selected trajectory
    4. Creates a legend identifying each ensemble
    
    This visualization helps in comparing path behavior between different REPPTIS ensembles
    and understanding the sampling efficiency of the REPPTIS approach.
    """
    cycle_nrs = {}
    fig, ax = plt.subplots()

    for i, pe in enumerate(pes):
        accmask = get_flag_mask(pe, "ACC")
        loadmask = get_generation_mask(pe, "ld")
        start_cond = pe.lambmins <= pe.interfaces[0][0]
        end_cond = pe.lambmaxs >= pe.interfaces[0][2]

        cycle_nrs[i] = select_with_masks(pe.cyclenumbers, [start_cond, end_cond, accmask, ~loadmask])

        ax.vlines(interfaces, -4, 4, color='black')
        linecolor = None
        count = 0
        while True and i > 0:
            id = np.random.choice(cycle_nrs[i])
            if np.all(pe.orders[id][:, 1] >= 0):
                lines = ax.plot(pe.orders[id][:, 0], pe.orders[id][:, 1], color=linecolor, label=i)
                linecolor = lines[0].get_color()
                count += 1
            if count == numberof:
                break
    fig.legend()
    fig.show()

def plot_rv_comp(pes, interfaces, n_repptis, n_staple, pe_idxs=None):
    """
    Compare representative trajectories for REPPTIS and iSTAR models on a single phase space diagram.
    
    This function creates a visualization that directly compares path behavior between
    REPPTIS and iSTAR (staple) approaches, highlighting differences in sampling strategies
    and efficiency between these methods.
    
    Parameters
    ----------
    pes : list
        List of :py:class:`.PathEnsemble` objects containing trajectory information.
    interfaces : list
        List of interface positions to be displayed as vertical lines.
    n_repptis : int
        Number of REPPTIS trajectories to plot.
    n_staple : int
        Number of iSTAR (staple) trajectories to plot.
    pe_idxs : tuple, optional
        Tuple of (start, end) indices for path ensembles to compare. If None, compares all.
        This allows focusing the comparison on specific subsets of ensembles.
        
    Notes
    -----
    The function:
    1. Identifies paths matching REPPTIS and iSTAR criteria from the same ensembles
    2. Plots iSTAR paths as solid lines
    3. Plots REPPTIS paths with differentiated segments (middle segment highlighted)
    4. Uses consistent colors for paths from the same ensemble
    
    The visualization specifically highlights:
    - The full path for iSTAR trajectories
    - The key middle segment for REPPTIS trajectories (with peripheral segments dashed)
    - How each approach samples different aspects of the transition path space
    """
    if pe_idxs is None:
        pe_idxs = (0, len(pes)-1)
    assert pe_idxs[0] <= pe_idxs[1]
    cycle_nrs_repptis = {}
    cycle_nrs_staple = {}
    fig, ax = plt.subplots()
    ax.set_xlabel("Position x (=$\lambda$)")
    ax.set_ylabel("Momentum p")

    for i, pe in enumerate(pes):
        if i not in range(pe_idxs[0],pe_idxs[1]+1):
            continue
        accmask = get_flag_mask(pe, "ACC")
        loadmask = get_generation_mask(pe, "ld")
        start_cond_repptis = pe.lambmins >= interfaces[0] if i > 2 else pe.lambmins <= interfaces[0]
        end_cond_repptis = pe.lambmaxs < interfaces[-1] if i < len(interfaces)-1 else pe.lambmaxs >= interfaces[-1]
        start_cond_staple = pe.lambmins <= interfaces[0]
        end_cond_staple =  pe.lambmaxs >= interfaces[-1]
        dir_mask = pe.dirs == 1

        cycle_nrs_repptis[i] = select_with_masks(pe.cyclenumbers, [pe.lmrs == "LMR", start_cond_repptis, end_cond_repptis, dir_mask, accmask, ~loadmask])
        cycle_nrs_staple[i] = select_with_masks(pe.cyclenumbers, [pe.lmrs == "LMR", start_cond_staple, end_cond_staple, dir_mask, accmask, ~loadmask])

        ax.vlines(interfaces, -4, 4, color='black')
        linecolor = None
        count = 0 
        while True and i > 0:
            if count == n_staple:
                break
            id = np.random.choice(cycle_nrs_staple[i])
            if np.all(pe.orders[id][:, 1] >= 0):
                lines = ax.plot(pe.orders[id][:, 0], pe.orders[id][:, 1], ".-", color=linecolor, label=i)
                linecolor = lines[0].get_color()
                count += 1

        count = 0 
        it = 0
        while True and i > 0:
            if count == n_repptis:
                break
            id = np.random.choice(cycle_nrs_repptis[i])
            l_piece = pe.orders[id][:pe.istar_idx[id][0]+1, :]
            m_piece = pe.orders[id][pe.istar_idx[id][0]-1:pe.istar_idx[id][1]+2, :]
            r_piece = pe.orders[id][pe.istar_idx[id][1]:, :]
            if len(m_piece) <= 100 and len(l_piece)+len(r_piece) <= 20550:
                ax.plot(m_piece[:, 0], m_piece[:, 1], "*-", color=linecolor, label=i)
                ax.plot(l_piece[:, 0], l_piece[:, 1], "--", alpha=0.5, color=linecolor)
                ax.plot(r_piece[:, 0], r_piece[:, 1], "--", alpha=0.5, color=linecolor)
                count += 1
            it += 1
            if it > 200000:
                id = np.random.choice(pe.cyclenumbers[pe.lmrs == "LMR"])
                l_piece = pe.orders[id][:pe.istar_idx[id][0]+1, :]
                m_piece = pe.orders[id][pe.istar_idx[id][0]-1:pe.istar_idx[id][1]+2, :]
                r_piece = pe.orders[id][pe.istar_idx[id][1]:, :]
                if np.all(m_piece[:, 1] >= 0):
                    ax.plot(m_piece[:, 0], m_piece[:, 1], color=linecolor, label=i)
                    ax.plot(l_piece[:, 0], l_piece[:, 1], "--", alpha=0.5, color=linecolor)
                    ax.plot(r_piece[:, 0], r_piece[:, 1], "--", alpha=0.5, color=linecolor)
                    count += 1
    fig.legend()
    fig.show()

def display_data(pes, interfaces, n_int=None, weights=None):
    """
    Display detailed analysis of raw and weighted transition data across all path ensembles.
    
    This comprehensive function provides a deep inspection of transition data, including
    raw path counts, weighted transitions, and time-reversed statistics. It's designed
    for detailed debugging and validation of path statistics before further analysis.
    
    The function creates multiple views of the data:
    1. Raw unweighted path counts (C matrices)
    2. Path counts with new MD steps (C_md matrices)
    3. Weighted data with high acceptance weights (X matrices)
    4. Time-reversal symmetrized data (TR matrices)
    5. Combined statistics across all ensembles
    
    Parameters
    ----------
    pes : list
        List of :py:class:`.PathEnsemble` objects to analyze.
    interfaces : list
        List of interface positions that define the state boundaries.
    n_int : int, optional
        Number of interfaces to consider. If None, uses the length of pes.
    weights : dict, optional
        Dictionary of pre-computed weights. If None, weights are calculated.
    
    Returns
    -------
    tuple
        Tuple containing:
        - C: Dictionary of raw path count matrices for each ensemble
        - X: Dictionary of weighted path count matrices for each ensemble
        - W: Combined weight matrix across all ensembles
        
    Notes
    -----
    The function performs several validation checks:
    - Identifies significant discrepancies between raw and weighted counts
    - Flags transitions with poor time-reversal symmetry
    - Highlights transitions with unusual statistics
    
    This is primarily a diagnostic tool for detecting statistical anomalies
    and validating the quality of path sampling before proceeding to more
    complex analyses.
    """
    tresholdw = 0.03
    tresholdtr = 0.05
    masks = {}
    X = {}
    C = {}
    C_md = {}  # Number of paths where new MD steps are performed (shooting/WF)
    if n_int is None:
        n_int = len(pes)
    for i, pe in enumerate(pes):
        print(10*'-')
        print(f"ENSEMBLE [{i-1 if i>0 else 0}{"*" if i>0 else "-"}] | ID {i}")
        print(10*'-')
        # Get the lmr masks, weights, ACCmask, and loadmask of the paths
        masks[i] = get_lmr_masks(pe)
        if weights is None:
            w = get_weights_staple(i, pe.flags, pe.generation, pe.lmrs, n_int, ACCFLAGS, REJFLAGS, verbose=True)
        else:
            w = weights[i]
        accmask = get_flag_mask(pe, "ACC")
        loadmask = get_generation_mask(pe, "ld")
        md_mask = np.logical_or(pe.generation == "sh", pe.generation == "wf")
        if i == 1: md_mask = np.logical_or(md_mask, pe.generation == "s-")
        elif i == 0: md_mask = np.logical_or(md_mask, pe.generation == "s+")
        msg = f"Ensemble {pe.name[-3:]} has {len(w)} paths.\n The total "+\
                    f"weight of the ensemble is {np.sum(w)}\nThe total amount of "+\
                    f"accepted paths is {np.sum(accmask)}\nThe total amount of "+\
                    f"load paths is {np.sum(loadmask)}"
        logger.debug(msg)

        X[i] = np.zeros([len(interfaces), len(interfaces)])
        C[i] = np.zeros([len(interfaces), len(interfaces)])
        C_md[i] = np.zeros([len(interfaces), len(interfaces)])
        X_val = compute_weight_matrix(pe, i, interfaces, tr=False, weights=weights)

        # 1. Displaying raw data, only unweighted X_ijk
        # 2. Displaying weighted data W_ijk
        # 3. Displaying weighted data with time reversal
        no_w = np.ones_like(w)
        for j in range(len(interfaces)):
            for k in range(len(interfaces)):
                if j == k:
                        if i == 1 and j == 0:
                            X[i][j][k] = np.sum(select_with_masks(w, [masks[i]["LML"], accmask, ~loadmask]))
                            C[i][j][k] = np.sum(select_with_masks(no_w, [masks[i]["LML"], accmask, ~loadmask]))
                            C_md[i][j][k] = np.sum(select_with_masks(no_w, [masks[i]["LML"], md_mask, accmask, ~loadmask]))
                            pass
                        else:
                            X[i][j][k] = 0
                elif j < k:
                    if j == 0 and k == 1:
                        if i > 2:
                            # dir_mask = np.logical_or(pe.dirs == 1, masks[i]["LML"])
                            dir_mask = pe.dirs == 1
                        elif i == 1:
                            # dir_mask = np.full_like(pe.dirs, True)
                            dir_mask = masks[i]["LMR"]    # Distinction for 0 -> 1 paths in [0*] 
                        elif i == 2:
                            # dir_mask = pe.dirs == 1
                            dir_mask = np.full_like(pe.dirs, True)  # For now no distinct    ion yet for [1*] paths: classify all paths as 1 -> 0. Later: dir_mask = np.logical_or(pe.dirs == 1, masks[i]["LML"])
                        else:
                            dir_mask = np.full_like(pe.dirs, False)
                    elif j == 0 and k == 2:
                        if i == 1:
                            dir_mask = masks[i]["LMR"]
                        else: 
                            dir_mask = pe.dirs == 1
                    elif j == len(interfaces)-2 and k == len(interfaces)-1:
                        dir_mask = masks[i]["RMR"]
                    else:
                        dir_mask = pe.dirs == 1
                        # dir_mask = np.logical_or(pe.dirs == 1, masks[i]["LML"])
                    if j == 0:
                        start_cond = pe.lambmins <= interfaces[j]
                    else: 
                        start_cond = np.logical_and(pe.lambmins <= interfaces[j], pe.lambmins >= interfaces[j-1])
                    if k == len(interfaces)-1:
                        end_cond = pe.lambmaxs >= interfaces[k]
                    else: 
                        end_cond = np.logical_and(pe.lambmaxs >= interfaces[k], pe.lambmaxs <= interfaces[k+1])
                
                    X[i][j][k] = np.sum(select_with_masks(w, [start_cond, end_cond, dir_mask, accmask, ~loadmask])) 
                    C[i][j][k] = np.sum(select_with_masks(no_w, [start_cond, end_cond, dir_mask, accmask, ~loadmask])) 
                    C_md[i][j][k] = np.sum(select_with_masks(no_w, [start_cond, end_cond, dir_mask, md_mask, accmask, ~loadmask])) 

                else:
                    if j == 1 and k == 0:
                        if i > 2:
                            dir_mask = pe.dirs == -1
                        elif i == 1:
                            dir_mask = masks[i]["RML"]    # Distinction for 1 -> 0 paths in [0*]
                        elif i == 2:
                            dir_mask = np.full_like(pe.dirs, True)   # For now no distinction yet for [1*] paths: classify all paths as 1 -> 0. Later: check if shooting point comes before or after crossing lambda_1/lambda_max
                        else:
                            dir_mask = np.full_like(pe.dirs, False)
                    elif j == len(interfaces)-2 and k == len(interfaces)-1:
                        dir_mask = masks[i]["RMR"]
                    else:
                        dir_mask = pe.dirs == -1
                    if k == 0:
                        start_cond = pe.lambmins <= interfaces[k]
                    else: 
                        start_cond = np.logical_and(pe.lambmins <= interfaces[k], pe.lambmins >= interfaces[k-1])
                    if j == len(interfaces)-1:
                        end_cond = pe.lambmaxs >= interfaces[j]
                    else: 
                        end_cond = np.logical_and(pe.lambmaxs >= interfaces[j], pe.lambmaxs <= interfaces[j+1])

                    X[i][j][k] = np.sum(select_with_masks(w, [start_cond, end_cond, dir_mask, accmask, ~loadmask])) 
                    C[i][j][k] = np.sum(select_with_masks(no_w, [start_cond, end_cond, dir_mask, accmask, ~loadmask])) 
                    C_md[i][j][k] = np.sum(select_with_masks(no_w, [start_cond, end_cond, dir_mask, md_mask, accmask, ~loadmask])) 
        
        frac_unw = C[i]/np.sum(C[i])
        frac_w = X[i]/np.sum(X[i])
        difffrac = abs(frac_unw - frac_w)
        idx_weirdw = np.transpose((difffrac>=tresholdw).nonzero())

        tr_diff = abs(X[i]-X[i].T) / ((X[i]+X[i].T)/2.0)
        idx_tr = np.transpose((tr_diff>=tresholdtr).nonzero())
        idx_tr = set((a,b) if a<=b else (b,a) for a,b in idx_tr)

        # 1. Raw unweighted path counts
        print("\n1a. Raw data: unweighted C matrices")
        print(f"C[{i}] = ")
        print(np.array2string(C[i], precision=4, suppress_small=True))
        
        # 2. Path counts with new MD steps
        print("\n1b. Raw data: unweighted path counts with new MD steps")
        print(f"C_md[{i}] = ")
        print(np.array2string(C_md[i], precision=4, suppress_small=True))
        
        # 3. Weighted data including high acceptance weights
        print("\n2. Weighted data: including high acceptance weights")
        print(f"X[{i}] = ")
        print(np.array2string(X[i], precision=4, suppress_small=True))
        print(f"Sum weights ensemble {i}: {np.sum(X[i]):.4f}")
        
        # Warnings for significant differences between weighted and raw data
        if len(idx_weirdw) > 0:
            print("\n[WARNING] Significant differences between weighted and raw counts:")
            for idx in idx_weirdw:
                print(f"  Path {idx[0]} → {idx[1]}: raw count={C[i][idx[0]][idx[1]]:.1f}, "
                    f"weighted={X[i][idx[0]][idx[1]]:.1f}, "
                    f"fraction diff={difffrac[idx[0], idx[1]]:.4f}, "
                    f"MD paths={C_md[i][idx[0]][idx[1]]:.1f}")
        
        # 4. Weighted data with time reversal symmetry
        print("\n3a. Weighted data with time reversal")
        if i == 2 and X[i][0, 1] == 0:     # In [1*] all LML paths are classified as 1 -> 0 (for now).
            X[i][1, 0] *= 2     # Time reversal needs to be adjusted to compensate for this
        elif i == len(interfaces)-1 and X[i][-1, -2] == 0:
            X[i][-2, -1] *= 2
        X_tr = (X[i]+X[i].T)/2.0
        print(np.array2string(X_tr, precision=4, suppress_small=True))
        
        # Warnings for significant differences in time-reversal symmetry
        if len(idx_tr) > 0:
            print("\n[WARNING] Time-reversal symmetry violations:")
            for idx in idx_tr:
                print(f"  Path {idx[0]} ↔ {idx[1]}: relative diff={tr_diff[idx[0],idx[1]]:.4f}, "
                    f"forward weight={X[i][idx[0]][idx[1]]:.2f}, "
                    f"backward weight={X[i][idx[1]][idx[0]]:.2f}")
        
        # 5. Unweighted data with time reversal symmetry
        print("\n3b. Unweighted data with time reversal")
        C_tr = (C[i]+C[i].T)/2.0
        print(np.array2string(C_tr, precision=4, suppress_small=True))

    # Combined statistics across all ensembles
    print("\n" + "="*40)
    print("\nALL ENSEMBLES COMBINED")
    print("-"*20)
    
    # Calculate combined weight matrix
    W = np.zeros_like(X[1])
    for j in range(len(interfaces)):
        for k in range(len(interfaces)):
            W[j][k] = np.sum([X[i][j][k] for i in range(n_int)])
    
    # Combined weights without time-reversal symmetry
    print("\n4. Weights of all ensembles combined (sum), no TR")
    print(np.array2string(W, precision=4, suppress_small=True))
    
    # Combined weights with time-reversal symmetry
    print("\n5. Weights of all ensembles combined (sum), with TR")
    W_tr = (W+W.T)/2.
    if W[1,0] == 0:
        W_tr[0,1] *= 1 
        W_tr[1,0] *= 1
    print(np.array2string(W_tr, precision=4, suppress_small=True))

    return C, X, W


def memory_analysis(w_path, tr=False):
    """
    Analyze memory effects in transition paths by calculating conditional crossing probabilities.
    
    This function quantifies how previous interface crossings influence future crossing probabilities,
    revealing the presence of memory effects in the system. Memory effects indicate non-Markovian
    behavior, where the history of a path affects its future evolution.
    
    The analysis works by calculating:
    1. For each ensemble separately: The probability q_k that a path starting at interface i
       and reaching k-1 (or k+1) will reach k, conditional on its starting point
    2. Across all ensembles: The combined probability q_tot with the same structure
    
    Parameters
    ----------
    w_path : dict
        Dictionary containing weighted path counts between interfaces.
    tr : bool, optional
        If True, enforces time-reversal symmetry in the weights before analysis.
        Default is False.
    
    Returns
    -------
    tuple
        Tuple containing memory effect matrices:
        - q_k: 4D array [2, n_ensembles-1, n_interfaces, n_interfaces] where:
          * q_k[0] contains conditional crossing probabilities for each ensemble
          * q_k[1] contains the number of samples for each calculation
        - q_tot: 3D array [2, n_interfaces, n_interfaces] where:
          * q_tot[0] contains combined conditional crossing probabilities
          * q_tot[1] contains the total number of samples
        
    Notes
    -----
    This analysis is crucial for:
    - Validating the Markovian assumption in TIS models
    - Identifying regions with significant memory effects
    - Determining whether more sophisticated models beyond iSTAR are needed
    
    The function creates visualizations of memory effects and prints detailed
    statistics for both forward (L→R) and backward (R→L) transitions.
    
    In a purely diffusive (memory-less) system, all conditional probabilities
    would be 0.5, regardless of starting point.
    """
    n_int = list(w_path.values())[0].shape[0]
    q_k = np.zeros([2, n_int-1, n_int, n_int])
    for ens in range(1, n_int):
        if tr:
            w_path[ens] += w_path[ens].T
        for i in range(w_path[ens].shape[0]):
            for k in range(w_path[ens].shape[0]):
                counts = np.zeros(2)
                if i == k:
                    if i == 0:
                        q_k[0][ens-1][i][k] = 1
                        continue
                    else:
                        continue
                elif i == 0 and k == 1 and ens == 1:
                    q_k[0][ens-1][i][k] = (np.sum(w_path[ens][i][k:])) / (np.sum(w_path[ens][i][k-1:]))
                    q_k[1][ens-1][i][k] = np.sum(w_path[ens][i][k-1:])
                    continue
                elif i < k:
                    if i <= ens <= k:
                        counts += [np.sum(w_path[ens][i][k:]), np.sum(w_path[ens][i][k-1:])]
                elif i > k:
                    if k+2 <= ens <= i+1:
                        counts += [np.sum(w_path[ens][i][:k+1]), np.sum(w_path[ens][i][:k+2])]

                q_k[0][ens-1][i][k] = counts[0] / counts[1] if counts[1] > 0 else np.nan
                q_k[1][ens-1][i][k] = counts[1]
        
        print(20*'-')
        print(f"ENSEMBLE [{ens-1}*] | ID {ens}")
        print(20*'-')
        print("==== L -> R ====")
        for intf in range(ens, n_int):
            print(f"-> interface {intf-1} is reached, probability to reach interface {intf}:")
            for start in range(ens):
                print(f"    - START at interface {start}: {q_k[0][ens-1][start][intf]} | # weights: {q_k[1][ens-1][start][intf]}")
        print("==== R -> L ====")
        for intf in range(ens-1):
            print(f"-> interface {intf+1} is reached, probability to reach interface {intf}:")
            for start in range(ens-1, n_int):
                print(f"    - START at interface {start}: {q_k[0][ens-1][start][intf]} | # weights: {q_k[1][ens-1][start][intf]}")

    q_tot = np.ones([2, n_int, n_int])
    for i in range(n_int):
        for k in range(n_int):
            counts = np.zeros(2)
            if i == k:
                if i == 0:
                    q_tot[0][i][k] = 1
                    continue
                else:
                    q_tot[0][i][k] = 0
                    continue
            elif i == 0 and k == 1:
                q_tot[0][i][k] = (np.sum(w_path[i+1][i][k:])) / (np.sum(w_path[i+1][i][k-1:]))
                q_tot[1][i][k] = np.sum(w_path[i+1][i][k-1:])
                continue
            elif i < k:
                for pe_i in range(i+1, k+1):
                    if pe_i > n_int-1:
                        break
                    counts += [np.sum(w_path[pe_i][i][k:]), np.sum(w_path[pe_i][i][k-1:])]
            elif i > k:
                for pe_i in range(k+2, i+2):
                    if pe_i > n_int-1:
                        break
                    counts += [np.sum(w_path[pe_i][i][:k+1]), np.sum(w_path[pe_i][i][:k+2])]

            q_tot[0][i][k] = counts[0] / counts[1] if counts[1] > 0 else np.nan
            q_tot[1][i][k] = counts[1]
    print()
    print(20*'-')
    print(f"TOTAL - ALL ENSEMBLES")
    print(20*'-')
    print("==== L -> R ====")
    for intf in range(1, n_int):
        print(f"-> interface {intf-1} is reached, probability to reach interface {intf}:")
        for start in range(intf):
            print(f"    - START at interface {start}: {q_tot[0][start][intf]} | # weights: {q_tot[1][start][intf]}")
    print("==== R -> L ====")
    for intf in range(n_int-2):
        print(f"-> interface {intf+1} is reached, probability to reach interface {intf}:")
        for start in range(intf+1, n_int):
            print(f"    - START at interface {start}: {q_tot[0][start][intf]} | # weights: {q_tot[1][start][intf]}")
    print("\nIntermediate probabilities matrix Q:")
    print("Conditional crossing probabilities:")
    print(np.array2string(q_tot[0], precision=3, suppress_small=True))
    print("\nNumber of samples for each calculation:")
    print(np.array2string(q_tot[1], precision=0, suppress_small=True))

    return q_k, q_tot

def ploc_memory(pathensembles, interfaces, trr=True):
    """
    Calculate global crossing probabilities using multiple methods and compare their results.
    
    This function implements and compares three different approaches for calculating
    global crossing probabilities:
    1. Milestoning: Based on local transitions between adjacent interfaces
    2. REPPTIS: Using the replica exchange TIS formalism
    3. APPTIS: Using the iSTAR path ensemble network approach
    
    The comparison reveals how different theoretical frameworks estimate transition
    probabilities, highlighting possible model dependencies or memory effects.
    
    Parameters
    ----------
    pathensembles : list
        List of :py:class:`.PathEnsemble` objects from TIS simulations.
    interfaces : list
        List of interface positions defining the order parameter space.
    trr : bool, optional
        If True, enforces time-reversal symmetry in the APPTIS calculation.
        Default is True.
    
    Returns
    -------
    dict
        Dictionary containing global crossing probabilities calculated with different methods:
        - "mlst": Milestoning probabilities
        - "apptis": APPTIS probabilities using iSTAR
        - "repptis": REPPTIS probabilities
        
    Notes
    -----
    This function also generates a logarithmic plot comparing the three methods, showing:
    - How estimates of crossing probabilities vary between methods
    - Whether the system exhibits significant memory effects (differences between methods)
    - Which method might be most appropriate for the system under study
    
    Significant differences between methods usually indicate memory effects or
    sampling issues that require careful interpretation.
    """
    plocs = {}
    plocs["mlst"] = [1.,]
    plocs["apptis"] = [1.,]
    repptisploc = []

    for i, pe in enumerate(pathensembles):
        # REPPTIS p_loc
        repptisploc.append(get_local_probs(pe, tr=False))

        # Milestoning p_loc
        if i == 1:
            plocs["mlst"].append(repptisploc[i]["LMR"]*plocs["mlst"][-1])
        elif i > 1:
            pmin = [repptisploc[r]["2L"] for r in range(1,len(repptisploc))]
            pplus = [repptisploc[r]["2R"] for r in range(1,len(repptisploc))]
            Milst = construct_M_milestoning(pmin, pplus, len(interfaces[:i+1]))
            z1, z2, y1, y2 = global_pcross_msm(Milst)
            plocs["mlst"].append(y1[0][0])

        # APPTIS p_loc
        if i < len(pathensembles)-1:
            wi = compute_weight_matrices(pathensembles[:i+2], interfaces[:i+2], len(interfaces), tr=trr)
            pi = get_transition_probs_weights(wi)
            Mi = construct_M_istar(pi, max(4, 2*len(interfaces[:i+2])), len(interfaces[:i+2]))
            z1, z2, y1, y2 = global_pcross_msm_star(Mi)
            plocs["apptis"].append(y1[0][0])

    _, _, plocs["repptis"] = get_global_probs_from_dict(repptisploc)

    print("\n=== Global Crossing Probability Analysis ===")
    print("\nMilestoning p_loc:")
    print(np.array2string(np.array(plocs["mlst"]), precision=4, suppress_small=True))
    
    print("\nREPPTIS p_loc:")
    print(np.array2string(np.array(plocs["repptis"]), precision=4, suppress_small=True))
    
    print("\nAPPTIS p_loc:")
    print(np.array2string(np.array(plocs["apptis"]), precision=4, suppress_small=True))

    # Make a figure of the global crossing probabilities
    plt.rcParams['text.usetex'] = True
    fig, ax = plt.subplots()
    ax.set_yscale("log")
    ax.errorbar([i for i in range(len(interfaces))], plocs["apptis"], fmt="-o", c = "b", ecolor="r", capsize=6, label="APPTIS")
    ax.errorbar([i for i in range(len(interfaces))], plocs["repptis"], fmt="-o", c = "orange", ecolor="r", capsize=6., label="REPPTIS")
    ax.errorbar([i for i in range(len(interfaces))], plocs["mlst"], fmt="-o", c = "r", ecolor="r", capsize=6., label="Milestoning")
    ax.set_xlabel(r"Interface index")
    ax.set_ylabel(r"$P_A(\lambda_i|\lambda_A)$")
    ax.set_xticks(np.arange(len(interfaces)))
    fig.tight_layout()
    fig.legend()
    fig.show()

    return plocs

def plot_memory_analysis_old(q_tot, p, interfaces=None):
    """
    Generate comprehensive visualizations for memory effect analysis in TIS simulations.
    
    This function creates three organized figures that analyze memory effects from multiple perspectives:
    1. Matrix heatmaps: memory effect matrix, memory effect ratio, and memory asymmetry
    2. Forward/backward probability plots with corresponding memory retention bar charts
    3. Additional memory effect visualizations including decay profiles and relaxation times
    
    Parameters
    ----------
    q_tot : numpy.ndarray
        A matrix with shape [2, n_interfaces, n_interfaces] where:
        - q_tot[0][i][k]: conditional crossing probabilities
        - q_tot[1][i][k]: sample counts for each calculation
    p : numpy.ndarray
        Transition probability matrix between interfaces.
    interfaces : list, optional
        The interface positions for axis labeling. If None, uses sequential indices.
        
    Returns
    -------
    tuple
        A tuple containing three matplotlib.figure.Figure objects:
        - fig1: Matrix heatmaps (memory effect matrix, ratio, asymmetry)
        - fig2: Forward/backward probability plots with memory retention bar charts
        - fig3: Memory decay profiles and additional visualizations
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    import matplotlib.gridspec as gridspec
    import matplotlib.colors as colors
    import seaborn as sns
    from scipy.optimize import curve_fit
    
    # Set seaborn style for better aesthetics
    sns.set(style="whitegrid", context="paper", font_scale=1.1)
    
    # Extract the probability matrix and weights matrix from q_tot
    q_probs = q_tot[0]
    q_weights = q_tot[1]
    n_interfaces = q_probs.shape[0]
    
    if interfaces is None:
        interfaces = list(range(n_interfaces))
    
    # Function to generate high-contrast colors for plots
    def generate_high_contrast_colors(n):
        if n <= 1:
            return ["#1f77b4"]  # Default blue for single item
        
        if n <= 10:
            # Use seaborn's color_palette for better colors
            return sns.color_palette("viridis", n)
        else:
            # For more interfaces, use viridis with adjusted spacing
            cmap1 = plt.cm.get_cmap('viridis')
            
            # Get colors with deliberate spacing for better contrast
            colors_list = []
            for i in range(n):
                # Distribute colors with slight variations in spacing
                # This avoids adjacent indices having too similar colors
                pos = (i / max(1, n-1)) * 0.85 + 0.1  # Scale to range 0.1-0.95
                
                # Introduce small oscillations in color position for adjacent indices
                if i % 2 == 1:
                    pos = min(0.95, pos + 0.05)
                    
                colors_list.append(colors.to_hex(cmap1(pos)))
                
            return colors_list
    
    # Exponential decay function for fitting memory effects
    def exp_decay(x, a, tau, c):
        return a * np.exp(-x / tau) + c
    
    # ================ Figure 1: Matrix Heatmaps ================
    fig1 = plt.figure(figsize=(18, 7))
    gs1 = gridspec.GridSpec(1, 3, width_ratios=[1.2, 1, 1])
    
    # Create custom colormap for memory effect heatmap
    cmap_memory = LinearSegmentedColormap.from_list('memory_effect', 
                                                  [(0, 'blue'), (0.5, 'white'), (1, 'red')], N=256)
    
    # Plot 1.1: Memory Effect Matrix (q_probs)
    ax1 = fig1.add_subplot(gs1[0])
    masked_data = np.ma.masked_invalid(q_probs)  # Mask NaN values
    im1 = ax1.imshow(masked_data, cmap=cmap_memory, vmin=0, vmax=1, 
                    interpolation='none', aspect='auto')
    
    # Add colorbar
    cbar1 = fig1.colorbar(im1, ax=ax1, label='Probability')
    
    # Add reference line at 0.5
    cbar1.ax.axhline(y=0.5, color='black', linestyle='--', linewidth=1)
    cbar1.ax.text(1.5, 0.5, '0.5 (diffusive)', va='center', ha='left', fontsize=9)
    
    # Add annotations for probability values and sample sizes
    for i in range(n_interfaces):
        for k in range(n_interfaces):
            if not np.isnan(q_probs[i, k]) and not np.ma.is_masked(masked_data[i, k]):
                weight = q_weights[i, k]
                text = f"{q_probs[i, k]:.2f}\n(n={int(weight)})" if weight > 0 else "N/A"
                color = 'black' if 0.2 < q_probs[i, k] < 0.8 else 'white'
                ax1.text(k, i, text, ha='center', va='center', color=color, fontsize=8)
    
    # Set ticks and labels
    ax1.set_xticks(np.arange(n_interfaces))
    ax1.set_yticks(np.arange(n_interfaces))
    ax1.set_xticklabels([f"{i}" for i in range(n_interfaces)])
    ax1.set_yticklabels([f"{i}" for i in range(n_interfaces)])
    ax1.set_xlabel('Target Interface k')
    ax1.set_ylabel('Starting Interface i')
    ax1.set_title('Memory Effect Matrix: q(i,k)', fontsize=12)
    
    # Plot 1.2: Memory Effect Ratio
    ax2 = fig1.add_subplot(gs1[1])
    
    # Calculate memory effect ratio: P/(1-P) compared to diffusive 0.5/(1-0.5)=1
    memory_ratio = np.zeros_like(q_probs)
    memory_ratio.fill(np.nan)
    
    for i in range(n_interfaces):
        for k in range(n_interfaces):
            if not np.isnan(q_probs[i, k]) and q_probs[i, k] != 0 and q_probs[i, k] != 1:
                memory_ratio[i, k] = (q_probs[i, k] / (1 - q_probs[i, k])) / 1.0  # Normalize by diffusive ratio of 1
    
    # Plot heatmap with logarithmic scale
    im2 = ax2.imshow(memory_ratio, cmap='RdBu_r', norm=colors.LogNorm(vmin=0.1, vmax=10))
    
    # Add colorbar
    cbar2 = fig1.colorbar(im2, ax=ax2, label='Probability Ratio P/(1-P) [log scale]')
    
    # Add annotations for ratio values
    for i in range(n_interfaces):
        for k in range(n_interfaces):
            if not np.isnan(memory_ratio[i, k]) and q_weights[i, k] > 5:
                text_color = 'black'
                if memory_ratio[i, k] > 5 or memory_ratio[i, k] < 0.2:
                    text_color = 'white'
                ax2.text(k, i, f"{memory_ratio[i, k]:.2f}", ha='center', va='center', 
                       color=text_color, fontsize=8)
    
    ax2.set_xlabel('Target Interface k')
    ax2.set_ylabel('Starting Interface i')
    ax2.set_title('Memory Effect Ratio: Deviation from Diffusive Behavior', fontsize=12)
    ax2.set_xticks(range(n_interfaces))
    ax2.set_yticks(range(n_interfaces))
    ax2.set_xticklabels([f"{i}" for i in range(n_interfaces)])
    ax2.set_yticklabels([f"{i}" for i in range(n_interfaces)])
    
    # Plot 1.3: Memory Asymmetry
    ax3 = fig1.add_subplot(gs1[2])
    
    # Calculate memory asymmetry for pairs of interfaces (i, j) using the p matrix
    memory_asymmetry = np.zeros_like(p)
    memory_asymmetry.fill(np.nan)
    
    for i in range(n_interfaces):
        for j in range(n_interfaces):
            if i != j:
                # Asymmetry is the difference between forward and backward transition probabilities
                memory_asymmetry[i, j] = p[i, j] - p[j, i]
    
    # Plot heatmap
    im3 = ax3.imshow(memory_asymmetry, cmap='RdBu', vmin=-0.5, vmax=0.5)
    
    # Add colorbar
    cbar3 = fig1.colorbar(im3, ax=ax3, label='Probability Asymmetry (i→j vs j→i)')
    
    # Add annotations
    for i in range(n_interfaces):
        for j in range(n_interfaces):
            if not np.isnan(memory_asymmetry[i, j]):
                text_color = 'black'
                if abs(memory_asymmetry[i, j]) > 0.3:
                    text_color = 'white'
                ax3.text(j, i, f"{memory_asymmetry[i, j]:.2f}", ha='center', va='center', 
                       color=text_color, fontsize=8)
    
    ax3.set_xlabel('Target Interface j')
    ax3.set_ylabel('Starting Interface i')
    ax3.set_title('Memory Asymmetry: Forward vs. Backward Transitions', fontsize=12)
    ax3.set_xticks(range(n_interfaces))
    ax3.set_yticks(range(n_interfaces))
    ax3.set_xticklabels([f"{i}" for i in range(n_interfaces)])
    ax3.set_yticklabels([f"{i}" for i in range(n_interfaces)])
    
    # Add explanatory text
    desc_text = """
    Memory Effect Matrix: Shows conditional crossing probabilities q(i,k).
    • For i<k: probability that a path starting at i and reaching k-1 will reach k
    • For i>k: probability that a path starting at i and reaching k+1 will reach k
    
    In a purely diffusive process, all values would be 0.5.
    Values > 0.5 (red) indicate bias toward crossing, < 0.5 (blue) indicate bias toward returning.
    """
    fig1.text(0.02, 0.02, desc_text, fontsize=10, wrap=True)
    
    plt.tight_layout(rect=[0, 0.07, 1, 0.95])
    fig1.suptitle('TIS Memory Effect Analysis - Matrix Representations', fontsize=14)
    
    # ================ Figure 2: Forward/Backward Probs + Memory Retention ================
    fig2 = plt.figure(figsize=(18, 12))
    gs2 = gridspec.GridSpec(2, 2, height_ratios=[1, 0.8])
    
    # Create colors for targets
    forward_targets = [k for k in range(1, n_interfaces)]
    forward_colors = generate_high_contrast_colors(len(forward_targets))
    
    backward_targets = [k for k in range(n_interfaces-1)]
    backward_colors = generate_high_contrast_colors(len(backward_targets)) 
    
    # Plot 2.1: Forward Transition Probabilities (L→R)
    ax4 = fig2.add_subplot(gs2[0, 0])
    
    # For each target interface k, plot q(i,k) for all starting interfaces i<k
    for idx, k in enumerate(forward_targets):
        target_data = []
        starting_interfaces = []
        
        # For interface 0, include adjacent transition to interface 1
        if k == 1:
            if not np.isnan(q_probs[0, k]) and q_weights[0, k] > 5:
                target_data.append(q_probs[0, k])
                starting_interfaces.append(0)
        
        # For all targets, include transitions from any starting point
        for i in range(k):
            # For interface 0, include all transitions (even adjacent)
            if i == 0:
                if not np.isnan(q_probs[i, k]) and q_weights[i, k] > 5:
                    target_data.append(q_probs[i, k])
                    starting_interfaces.append(i)
            # For other interfaces, skip adjacent transitions (i+1 = k)
            elif i < k-1:  # Only include non-adjacent transitions for i>0
                if not np.isnan(q_probs[i, k]) and q_weights[i, k] > 5:
                    target_data.append(q_probs[i, k])
                    starting_interfaces.append(i)
        
        if target_data:
            ax4.plot(starting_interfaces, target_data, 'o-', 
                   label=f'Target: {k}', linewidth=2, markersize=8,
                   color=forward_colors[idx])
    
    # Add reference line at 0.5
    ax4.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7)
    
    # Configure the forward plot
    ax4.set_xlabel('Starting Interface i')
    ax4.set_ylabel('Probability q(i,k)')
    ax4.set_title('Forward Transition Probabilities (L→R)', fontsize=12)
    ax4.set_ylim(0, 1.05)
    ax4.set_xlim(-0.5, n_interfaces-1.5)
    ax4.set_xticks(range(n_interfaces))
    ax4.legend(title='Target Interface k', loc='best')
    
    # Plot 2.2: Backward Transition Probabilities (R→L)
    ax5 = fig2.add_subplot(gs2[0, 1])
    
    # For each target interface k, plot q(i,k) for all starting interfaces i>k
    for idx, k in enumerate(backward_targets):
        target_data = []
        starting_interfaces = []
        for i in range(k+2, n_interfaces):  # Skip adjacent (i = k+1)
            if not np.isnan(q_probs[i, k]) and q_weights[i, k] > 5:
                target_data.append(q_probs[i, k])
                starting_interfaces.append(i)
        
        if target_data:
            ax5.plot(starting_interfaces, target_data, 'o-', 
                   label=f'Target: {k}', linewidth=2, markersize=8,
                   color=backward_colors[idx])
    
    # Add reference line at 0.5
    ax5.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7)
    
    # Configure the backward plot
    ax5.set_xlabel('Starting Interface i')
    ax5.set_ylabel('Probability q(i,k)')
    ax5.set_title('Backward Transition Probabilities (R→L)', fontsize=12)
    ax5.set_ylim(0, 1.05)
    ax5.set_xlim(0.5, n_interfaces-0.5)
    ax5.set_xticks(range(n_interfaces))
    ax5.legend(title='Target Interface k', loc='best')
    
    # Plot 2.3: Forward Memory Retention
    ax6 = fig2.add_subplot(gs2[1, 0])
    
    # Calculate memory retention using relative standard deviation
    memory_retention = np.zeros(n_interfaces)
    
    for k in range(1, n_interfaces):
        # Get all probabilities for reaching k from different starting points
        # Exclude i=k (q(i,i)=0) and i=k-1 (q(i,i+1)=1) for i>0
        probs = []
        for i in range(k-1):
            if i == 0 or i < k-1:  # Include adjacent transitions for i=0
                if not np.isnan(q_probs[i, k]) and q_weights[i, k] > 5:
                    probs.append(q_probs[i, k])
        
        if len(probs) > 1:
            # Calculate relative standard deviation (RSD = std/mean * 100%)
            mean_prob = np.mean(probs)
            if mean_prob > 0:  # Avoid division by zero
                std_prob = np.std(probs)
                memory_retention[k] = (std_prob / mean_prob) * 100  # As percentage
            else:
                memory_retention[k] = np.nan
    
    # Plot the memory retention (RSD)
    valid_k = [k for k in range(1, n_interfaces) if not np.isnan(memory_retention[k])]
    valid_retention = [memory_retention[k] for k in valid_k]
    valid_colors = [forward_colors[k-1] for k in valid_k]  # Match colors to targets in forward plot
    
    if valid_k:
        bars6 = ax6.bar(valid_k, valid_retention, color=valid_colors, alpha=0.7)
        
        # Add annotations for memory retention values
        for i, k in enumerate(valid_k):
            ax6.text(k, valid_retention[i] + 1, f"{valid_retention[i]:.1f}%", ha='center', fontsize=9)
        
        ax6.set_xlabel('Target Interface k')
        ax6.set_ylabel('Relative Standard Deviation (%)')
        ax6.set_title('Forward Memory Retention: Variability in Forward Transitions', fontsize=12)
        ax6.set_xticks(range(1, n_interfaces))
        ax6.set_ylim(0, max(valid_retention) * 1.2 if valid_retention else 10)
    else:
        ax6.text(0.5, 0.5, "Insufficient data for forward memory retention analysis", 
               ha='center', va='center', transform=ax6.transAxes)
    
    # Plot 2.4: Backward Memory Retention
    ax7 = fig2.add_subplot(gs2[1, 1])
    
    # Calculate backward memory retention
    backward_memory_retention = np.zeros(n_interfaces)
    
    for k in range(n_interfaces-1):
        # Get all probabilities for reaching k from different starting points i>k+1
        probs = [q_probs[i, k] for i in range(k+2, n_interfaces) if not np.isnan(q_probs[i, k]) and q_weights[i, k] > 5]
        if len(probs) > 1:
            mean_prob = np.mean(probs)
            if mean_prob > 0:  # Avoid division by zero
                std_prob = np.std(probs)
                backward_memory_retention[k] = (std_prob / mean_prob) * 100  # As percentage
            else:
                backward_memory_retention[k] = np.nan
    
    # Plot the backward memory retention (RSD)
    valid_k = [k for k in range(n_interfaces-1) if not np.isnan(backward_memory_retention[k])]
    valid_retention = [backward_memory_retention[k] for k in valid_k]
    valid_colors = [backward_colors[k] for k in valid_k]  # Match colors to targets in backward plot
    
    if valid_k:
        bars7 = ax7.bar(valid_k, valid_retention, color=valid_colors, alpha=0.7)
        
        # Add annotations for memory retention values
        for i, k in enumerate(valid_k):
            ax7.text(k, valid_retention[i] + 1, f"{valid_retention[i]:.1f}%", ha='center', fontsize=9)
        
        ax7.set_xlabel('Target Interface k')
        ax7.set_ylabel('Relative Standard Deviation (%)')
        ax7.set_title('Backward Memory Retention: Variability in Backward Transitions', fontsize=12)
        ax7.set_xticks(range(n_interfaces-1))
        ax7.set_ylim(0, max(valid_retention) * 1.2 if valid_retention else 10)
    else:
        ax7.text(0.5, 0.5, "Insufficient data for backward memory retention analysis", 
               ha='center', va='center', transform=ax7.transAxes)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig2.suptitle('TIS Memory Effect Analysis - Transition Probabilities and Memory Retention', fontsize=14)
    
    # ================ Figure 3: Memory Decay Profiles and Additional Visualizations ================
    fig3 = plt.figure(figsize=(18, 14))
    gs3 = gridspec.GridSpec(2, 2)
    
    # Plot 3.1: Memory Decay with Distance (Forward transitions)
    ax8 = fig3.add_subplot(gs3[0, 0])
    
    # Calculate memory effect as deviation from 0.5
    memory_data = np.zeros((n_interfaces-1, n_interfaces-1))
    memory_data.fill(np.nan)
    
    for start_i in range(n_interfaces-1):
        for target_k in range(start_i+1, n_interfaces):
            # Calculate distance between interfaces
            distance = target_k - start_i
            # Memory effect is deviation from 0.5
            if not np.isnan(q_probs[start_i, target_k]) and q_weights[start_i, target_k] > 5:
                memory_data[start_i, distance-1] = abs(q_probs[start_i, target_k] - 0.5)
    
    # Plot heatmap
    sns.heatmap(memory_data, cmap='viridis', ax=ax8, 
                cbar_kws={'label': '|Probability - 0.5| (Memory Effect Strength)'})
    
    ax8.set_xlabel('Interface Distance (k - i)')
    ax8.set_ylabel('Starting Interface i')
    ax8.set_title('Memory Effect Decay with Distance (Forward L→R)', fontsize=12)
    ax8.set_xticks(np.arange(n_interfaces-1) + 0.5)
    ax8.set_xticklabels(np.arange(1, n_interfaces))
    ax8.set_yticks(np.arange(n_interfaces-1) + 0.5)
    ax8.set_yticklabels(np.arange(n_interfaces-1))
    
    # Plot 3.2: Memory Decay with Distance (Backward transitions)
    ax9 = fig3.add_subplot(gs3[0, 1])
    
    # Calculate memory effect for backward transitions
    backward_memory_data = np.zeros((n_interfaces-1, n_interfaces-1))
    backward_memory_data.fill(np.nan)
    
    for start_i in range(1, n_interfaces):
        for target_k in range(start_i):
            # Calculate distance between interfaces
            distance = start_i - target_k
            # Memory effect is deviation from 0.5
            if not np.isnan(q_probs[start_i, target_k]) and q_weights[start_i, target_k] > 5:
                backward_memory_data[start_i-1, distance-1] = abs(q_probs[start_i, target_k] - 0.5)
    
    # Plot heatmap
    sns.heatmap(backward_memory_data, cmap='viridis', ax=ax9, 
                cbar_kws={'label': '|Probability - 0.5| (Memory Effect Strength)'})
    
    ax9.set_xlabel('Interface Distance (i - k)')
    ax9.set_ylabel('Starting Interface i')
    ax9.set_title('Memory Effect Decay with Distance (Backward R→L)', fontsize=12)
    ax9.set_xticks(np.arange(n_interfaces-1) + 0.5)
    ax9.set_xticklabels(np.arange(1, n_interfaces))
    ax9.set_yticks(np.arange(n_interfaces-1) + 0.5)
    ax9.set_yticklabels(np.arange(1, n_interfaces))
    
    # Plot 3.3: Memory Effect by Jump Distance
    ax10 = fig3.add_subplot(gs3[1, 0])
    
    max_distance = n_interfaces - 1
    distances = range(1, max_distance + 1)
    
    # Create colors for jump distances
    jump_colors = generate_high_contrast_colors(max_distance)
    
    for idx, dist in enumerate(distances):
        probs = []
        starting_positions = []
        for i in range(n_interfaces - dist):
            k = i + dist
            if not np.isnan(q_probs[i, k]) and q_weights[i, k] > 5:  # Minimum sample threshold
                probs.append(q_probs[i, k])
                starting_positions.append(i)
                
        if probs:
            ax10.plot(starting_positions, probs, 'o-', 
                    label=f'Δλ = {dist}', linewidth=2, markersize=8,
                    color=jump_colors[idx])
    
    # Add reference line at 0.5
    ax10.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7)
    
    # Configure the jump distance plot
    ax10.set_xlabel('Starting Interface i')
    ax10.set_ylabel('Probability q(i, i+dist)')
    ax10.set_title('Memory Effect by Jump Distance (L→R)', fontsize=12)
    ax10.set_ylim(0, 1.05)
    ax10.set_xlim(-0.5, n_interfaces-1.5)
    ax10.set_xticks(range(n_interfaces))
    ax10.legend(title='Jump Distance Δλ', loc='best')
    
    # Plot 3.4: Deviations from Diffusive Behavior in Transitions
    ax11 = fig3.add_subplot(gs3[1, 1])

    # Create array for adjacency transitions (i to i+1 and i to i-1) using p matrix
    adjacent_forward = np.zeros(n_interfaces - 1)
    adjacent_backward = np.zeros(n_interfaces - 1)

    for i in range(n_interfaces - 1):
        if i < p.shape[0] and i+1 < p.shape[1]:
            adjacent_forward[i] = p[i, i+1] - 0.5
        
    for i in range(1, n_interfaces):
        if i < p.shape[0] and i-1 < p.shape[1]:
            adjacent_backward[i-1] = p[i, i-1] - 0.5

    # Calculate y_limit for plot
    y_limit = max(np.nanmax(np.abs(adjacent_forward)), np.nanmax(np.abs(adjacent_backward))) * 1.2

    # Use the same colors from forward/backward transitions plots for consistency
    forward_base_color = sns.color_palette()[0]   # Base color for forward transitions
    backward_base_color = sns.color_palette()[1]  # Base color for backward transitions

    # Create bars with consistent colors that match the legend
    # For forward transitions: separate into positive and negative values
    pos_mask_fwd = adjacent_forward >= 0
    neg_mask_fwd = adjacent_forward < 0
    
    # Positive forward bars (higher alpha)
    ax11.bar(np.arange(n_interfaces-1)[pos_mask_fwd] - 0.2, adjacent_forward[pos_mask_fwd], 0.4, 
                    color=forward_base_color, edgecolor='black', linewidth=0.5, alpha=0.85,
                    label='Forward (i → i+1)')
    
    # Negative forward bars (lower alpha)
    ax11.bar(np.arange(n_interfaces-1)[neg_mask_fwd] - 0.2, adjacent_forward[neg_mask_fwd], 0.4, 
                    color=forward_base_color, edgecolor='black', linewidth=0.5, alpha=0.4)
    
    # For backward transitions: separate into positive and negative values
    pos_mask_bwd = adjacent_backward >= 0
    neg_mask_bwd = adjacent_backward < 0
    
    # Positive backward bars (higher alpha)
    ax11.bar(np.arange(n_interfaces-1)[pos_mask_bwd] + 0.2, adjacent_backward[pos_mask_bwd], 0.4, 
                    color=backward_base_color, edgecolor='black', linewidth=0.5, alpha=0.85,
                    label='Backward (i+1 → i)')
    
    # Negative backward bars (lower alpha)
    ax11.bar(np.arange(n_interfaces-1)[neg_mask_bwd] + 0.2, adjacent_backward[neg_mask_bwd], 0.4, 
                    color=backward_base_color, edgecolor='black', linewidth=0.5, alpha=0.4)

    # Add zero line
    ax11.axhline(y=0, color='k', linestyle='-', alpha=0.5)

    # Add shaded regions to indicate bias direction
    ax11.axhspan(0, y_limit, alpha=0.1, color='green', label='Bias toward crossing')
    ax11.axhspan(-y_limit, 0, alpha=0.1, color='red', label='Bias toward returning')

    # Add annotations with improved visibility
    for i, v in enumerate(adjacent_forward):
        if not np.isnan(v):
            offset = 0.02 if v >= 0 else -0.08
            ax11.text(i - 0.2, v + offset, f"{v:.2f}", ha='center', fontsize=9, 
                    fontweight='bold', color='black')

    for i, v in enumerate(adjacent_backward):
        if not np.isnan(v):
            offset = 0.02 if v >= 0 else -0.08
            ax11.text(i + 0.2, v + offset, f"{v:.2f}", ha='center', fontsize=9, 
                    fontweight='bold', color='black')

    # Add explanatory note about bias
    ax11.text(0.02, 0.02, 
            "Positive values: Bias toward crossing (>0.5)\nNegative values: Bias toward returning (<0.5)",
            transform=ax11.transAxes, fontsize=10, va='bottom', ha='left',
            bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))

    # Set better y limits to show all bars clearly
    max_abs_val = max(np.nanmax(np.abs(adjacent_forward)), np.nanmax(np.abs(adjacent_backward)))
    y_limit = max(0.5, max_abs_val * 1.3)
    ax11.set_ylim(-y_limit, y_limit)

    ax11.set_xlabel('Interface Pair (i, i+1)')
    ax11.set_ylabel('Deviation from Diffusive (P - 0.5)')
    ax11.set_title('Memory Effect in Adjacent Transitions', fontsize=12)
    ax11.set_xticks(np.arange(n_interfaces-1))
    ax11.set_xticklabels([f"{i},{i+1}" for i in range(n_interfaces - 1)])
    ax11.legend(loc='upper right')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig3.suptitle('TIS Memory Effect Analysis - Decay Profiles and Additional Metrics', fontsize=14)
    
    return fig1, fig2, fig3

def plot_memory_analysis(q_tot, p, interfaces=None):
    """
    Generate comprehensive visualizations for memory effect analysis in TIS simulations
    with support for non-equidistant interfaces.
    
    Parameters:
    -----------
    q_tot : numpy.ndarray
        A matrix with shape [2, n_interfaces, n_interfaces] where:
        - q_tot[0][i][k]: conditional crossing probabilities
        - q_tot[1][i][k]: sample counts for each calculation
    p : numpy.ndarray
        Transition probability matrix between interfaces
    interfaces : list, optional
        The interface positions for axis labeling. If None, uses sequential indices.
        
    Returns:
    -------
    tuple
        A tuple containing three matplotlib.figure.Figure objects:
        - fig1: Matrix heatmaps (memory effect matrix, ratio, asymmetry)
        - fig2: Forward/backward probability plots with memory retention bar charts
        - fig3: Memory decay profiles and additional visualizations
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    import matplotlib.gridspec as gridspec
    import matplotlib.colors as colors
    import seaborn as sns
    from scipy.optimize import curve_fit
    from matplotlib.widgets import Button
    # Add legend
    from matplotlib.lines import Line2D
    from matplotlib.lines import Line2D
    
    # Extract the probability matrix and weights matrix from q_tot
    q_probs = q_tot[0]
    q_weights = q_tot[1]
    n_interfaces = q_probs.shape[0]
    
    if interfaces is None:
        interfaces = list(range(n_interfaces))
        is_equidistant = True
    else:
        # Check if interfaces are equidistant
        if len(interfaces) > 2:
            diffs = np.diff(interfaces)
            is_equidistant = np.allclose(diffs, diffs[0], rtol=0.05)
        else:
            is_equidistant = True
    
    # Calculate diffusive reference probabilities based on interface spacing
    diff_ref = calculate_diffusive_reference_spacing(interfaces)
    
    # Function to generate high-contrast colors for plots
    def generate_high_contrast_colors(n):
        if n <= 1:
            return ["#1f77b4"]  # Default blue for single item
        
        if n <= 10:
            # Viridis with enhanced spacing for better contrast
            viridis_cmap = plt.cm.get_cmap('viridis')
            return [colors.to_hex(viridis_cmap(i/(n-1) if n > 1 else 0.5)) for i in range(n)]
        else:
            # For more interfaces, use viridis with adjusted spacing
            cmap1 = plt.cm.get_cmap('viridis')
            
            # Get colors with deliberate spacing for better contrast
            colors_list = []
            for i in range(n):
                # Distribute colors with slight variations in spacing
                # This avoids adjacent indices having too similar colors
                pos = (i / max(1, n-1)) * 0.85 + 0.1  # Scale to range 0.1-0.95
                
                # Introduce small oscillations in color position for adjacent indices
                if i % 2 == 1:
                    pos = min(0.95, pos + 0.05)
                    
                colors_list.append(colors.to_hex(cmap1(pos)))
                
            return colors_list
    
    # Exponential decay function for fitting memory effects
    def exp_decay(x, a, tau, c):
        return a * np.exp(-x / tau) + c
    
    # ================ Figure 1: Matrix Heatmaps ================
    fig1 = plt.figure(figsize=(18, 7))
    gs1 = gridspec.GridSpec(1, 3, width_ratios=[1.2, 1, 1])
    
    # Create custom colormap for memory effect heatmap
    cmap_memory = LinearSegmentedColormap.from_list('memory_effect', 
                                                  [(0, 'blue'), (0.5, 'white'), (1, 'red')], N=256)
    
    # Plot 1.1: Memory Effect Matrix (q_probs)
    ax1 = fig1.add_subplot(gs1[0])
    
    # Calculate memory effect as deviation from diffusive reference
    memory_effect = np.zeros_like(q_probs)
    memory_effect.fill(np.nan)
    
    for i in range(n_interfaces):
        for j in range(n_interfaces):
            if not np.isnan(q_probs[i, j]) and not np.isnan(diff_ref[i, j]):
                memory_effect[i, j] = q_probs[i, j] - diff_ref[i, j]
    
    # Create diverging colormap centered at 0
    max_effect = np.nanmax(np.abs(memory_effect))
    
    masked_data = np.ma.masked_invalid(memory_effect)  # Mask NaN values
    im1 = ax1.imshow(masked_data, cmap=cmap_memory, vmin=-max_effect, vmax=max_effect, 
                    interpolation='none', aspect='auto')
    
    # Add colorbar
    cbar1 = fig1.colorbar(im1, ax=ax1, label='Memory Effect (q - q_diff)')
    
    # Add reference line at 0
    cbar1.ax.axhline(y=0.0, color='black', linestyle='--', linewidth=1)
    cbar1.ax.text(1.5, 0.0, '0 (diffusive)', va='center', ha='left', fontsize=9)
    
    # Add annotations with more compact formatting
    for i in range(n_interfaces):
        for j in range(n_interfaces):
            if not np.isnan(memory_effect[i, j]) and not np.ma.is_masked(masked_data[i, j]):
                weight = q_weights[i, j]
                # More compact format: actual/diff
                text = f"{q_probs[i, j]:.2f}/{diff_ref[i, j]:.2f}" if weight > 0 else "N/A"
                # Only show count if it's significant
                if weight > 10:
                    text += f"\n{int(weight)}"
                color = 'black' if abs(memory_effect[i, j]) < 0.3 else 'white'
                ax1.text(j, i, text, ha='center', va='center', color=color, fontsize=7)
    
    # Set ticks and labels using index values
    ax1.set_xticks(np.arange(n_interfaces))
    ax1.set_yticks(np.arange(n_interfaces))
    ax1.set_xticklabels([f"{i}" for i in range(n_interfaces)])
    ax1.set_yticklabels([f"{i}" for i in range(n_interfaces)])
    ax1.set_xlabel('Target Interface k')
    ax1.set_ylabel('Starting Interface i')
    ax1.set_title('Memory Effect Matrix: q(i,k) - q_diff(i,k)', fontsize=12)
    
    # Plot 1.2: Memory Effect Ratio
    ax2 = fig1.add_subplot(gs1[1])
    
    # Calculate memory effect ratio: ratio of actual prob to diffusive prob
    memory_ratio = np.zeros_like(q_probs)
    memory_ratio.fill(np.nan)
    
    for i in range(n_interfaces):
        for j in range(n_interfaces):
            if (not np.isnan(q_probs[i, j]) and not np.isnan(diff_ref[i, j]) and
                diff_ref[i, j] > 0 and diff_ref[i, j] < 1):
                memory_ratio[i, j] = q_probs[i, j] / diff_ref[i, j]
    
    # Plot heatmap with logarithmic scale
    im2 = ax2.imshow(memory_ratio, cmap='RdBu_r', norm=colors.LogNorm(vmin=0.1, vmax=10))
    
    # Add colorbar
    cbar2 = fig1.colorbar(im2, ax=ax2, label='Probability Ratio q/q_diff [log scale]')
    
    # Add annotations for ratio values - more compact
    for i in range(n_interfaces):
        for j in range(n_interfaces):
            if not np.isnan(memory_ratio[i, j]) and q_weights[i, j] > 5:
                text_color = 'black'
                if memory_ratio[i, j] > 5 or memory_ratio[i, j] < 0.2:
                    text_color = 'white'
                ax2.text(j, i, f"{memory_ratio[i, j]:.1f}", ha='center', va='center', 
                       color=text_color, fontsize=7)
    
    ax2.set_xlabel('Target Interface k')
    ax2.set_ylabel('Starting Interface i')
    ax2.set_title('Memory Effect Ratio: Deviation from Diffusive Behavior', fontsize=12)
    ax2.set_xticks(range(n_interfaces))
    ax2.set_yticks(range(n_interfaces))
    ax2.set_xticklabels([f"{i}" for i in range(n_interfaces)])
    ax2.set_yticklabels([f"{i}" for i in range(n_interfaces)])
    
    # Plot 1.3: Memory Asymmetry
    ax3 = fig1.add_subplot(gs1[2])
    
    # Calculate memory asymmetry for pairs of interfaces (i, j) using the p matrix
    memory_asymmetry = np.zeros_like(p)
    memory_asymmetry.fill(np.nan)
    
    for i in range(n_interfaces):
        for j in range(n_interfaces):
            if i != j:
                # Asymmetry is the difference between forward and backward transition probabilities
                memory_asymmetry[i, j] = p[i, j] - p[j, i]
    
    # Plot heatmap
    im3 = ax3.imshow(memory_asymmetry, cmap='RdBu', vmin=-0.5, vmax=0.5)
    
    # Add colorbar
    cbar3 = fig1.colorbar(im3, ax=ax3, label='Probability Asymmetry (i→j vs j→i)')
    
    # Add annotations - more compact
    for i in range(n_interfaces):
        for j in range(n_interfaces):
            if not np.isnan(memory_asymmetry[i, j]):
                text_color = 'black'
                if abs(memory_asymmetry[i, j]) > 0.3:
                    text_color = 'white'
                ax3.text(j, i, f"{memory_asymmetry[i, j]:.2f}", ha='center', va='center', 
                       color=text_color, fontsize=7)
    
    ax3.set_xlabel('Target Interface j')
    ax3.set_ylabel('Starting Interface i')
    ax3.set_title('Memory Asymmetry: Forward vs. Backward Transitions', fontsize=12)
    ax3.set_xticks(range(n_interfaces))
    ax3.set_yticks(range(n_interfaces))
    ax3.set_xticklabels([f"{i}" for i in range(n_interfaces)])
    ax3.set_yticklabels([f"{i}" for i in range(n_interfaces)])
    
    # Add explanatory text that includes info about non-equidistant interfaces
    if is_equidistant:
        desc_text = """
        Memory Effect Matrix: Shows deviations from diffusive behavior (q - 0.5).
        In a purely diffusive process, all values would be 0.
        Values > 0 (red) indicate bias toward crossing, < 0 (blue) indicate bias toward returning.
        """
    else:
        desc_text = """
        Memory Effect Matrix: Shows deviations from diffusive behavior (q - q_diff).
        Due to non-equidistant interfaces, the diffusive reference varies for each transition.
        Values > 0 (red) indicate bias toward crossing, < 0 (blue) indicate bias toward returning.
        Format: actual/diffusive (sample count when >10)
        """
    fig1.text(0.02, 0.02, desc_text, fontsize=10, wrap=True)
    
    plt.tight_layout(rect=[0, 0.07, 1, 0.95])
    fig1.suptitle('TIS Memory Effect Analysis - Matrix Representations' + 
                (' (Non-equidistant Interfaces)' if not is_equidistant else ''), fontsize=14)
    
    # ================ Figure 2: Forward/Backward Probs + Memory Retention ================
    fig2 = plt.figure(figsize=(18, 12))
    gs2 = gridspec.GridSpec(2, 2, height_ratios=[1, 0.8])
    
    # Create colors for targets
    forward_targets = [k for k in range(1, n_interfaces)]
    forward_colors = generate_high_contrast_colors(len(forward_targets))
    
    backward_targets = [k for k in range(n_interfaces-1)]
    backward_colors = generate_high_contrast_colors(len(backward_targets)) 
    
    # Plot 2.1: Forward Transition Probabilities (L→R)
    ax4 = fig2.add_subplot(gs2[0, 0])

    # For each target interface k, plot q(i,k) for all starting interfaces i<k
    for idx, k in enumerate(forward_targets):
        target_data = []
        starting_positions = []
        ref_probs = []
        valid_indices = []
        
        for i in range(k):
            # Include adjacent transitions only for interface 0->1
            if (i < k-1 or (i == 0 and k == 1)) and not np.isnan(q_probs[i, k]) and q_weights[i, k] > 5:
                target_data.append(q_probs[i, k])
                starting_positions.append(interfaces[i])
                valid_indices.append(i)
                ref_probs.append(diff_ref[i, k])
        
        if target_data:
            # Plot actual probabilities with physical positions on x-axis
            ax4.plot(starting_positions, target_data, 'o-', 
                    label=f'Target: {k}', linewidth=2, markersize=8,
                    color=forward_colors[idx])
            
            # Plot diffusive reference as dashed lines
            ax4.plot(starting_positions, ref_probs, '--', 
                    color=forward_colors[idx], alpha=0.5)
    
    # Configure the forward plot
    ax4.set_xlabel('Interface Position (λ)')
    ax4.set_ylabel('Probability q(i,k)')
    ax4.set_title('Forward Transition Probabilities (L→R)', fontsize=12)
    ax4.set_ylim(0, 1.05)
    sns.despine(ax=ax4)
    
    # Create better x-axis ticks using interface indices as labels but keeping physical distances
    ax4.set_xlim(min(interfaces) - 0.1, interfaces[n_interfaces-2] + 0.1)
    # Set the physical positions of interfaces on the x-axis
    ax4.set_xticks(interfaces)
    # But label them with their indices
    ax4.set_xticklabels([f"{i}" for i in range(n_interfaces)])
    
    # Add explanatory text about the dashed lines
    ref_text = """
    Dashed lines: Diffusive reference probabilities
    • Based on free energy differences between interfaces
    • Calculated using detailed balance principle
    """
    ax4.text(0.02, 0.02, ref_text, transform=ax4.transAxes, fontsize=9, 
             bbox=dict(facecolor='white', alpha=0.8))
    
    # Add a legend with reasonable size
    ax4.legend(title='Target Interface', loc='best', fontsize=9)
    
    # Plot 2.2: Backward Transition Probabilities (R→L)
    ax5 = fig2.add_subplot(gs2[0, 1])
    
    # For each target interface k, plot q(i,k) for all starting interfaces i>k
    for idx, k in enumerate(backward_targets):
        target_data = []
        starting_positions = []
        ref_probs = []
        valid_indices = []
        
        for i in range(k+1, n_interfaces):
            # Exclude adjacent transitions (i.e., exclude i=k+1)
            if i > k+1 and not np.isnan(q_probs[i, k]) and q_weights[i, k] > 5:
                target_data.append(q_probs[i, k])
                starting_positions.append(interfaces[i])
                valid_indices.append(i)
                ref_probs.append(diff_ref[i, k])
        
        if target_data:
            # Plot actual probabilities with physical positions on x-axis
            ax5.plot(starting_positions, target_data, 'o-', 
                    label=f'Target: {k}', linewidth=2, markersize=8,
                    color=backward_colors[idx])
            
            # Plot diffusive reference as dashed lines
            ax5.plot(starting_positions, ref_probs, '--', 
                    color=backward_colors[idx], alpha=0.5)
    
    # Configure the backward plot
    ax5.set_xlabel('Interface Position (λ)')
    ax5.set_ylabel('Probability q(i,k)')
    ax5.set_title('Backward Transition Probabilities (R→L)', fontsize=12)
    ax5.set_ylim(0, 1.05)
    sns.despine(ax=ax5)
    
    # Create better x-axis ticks using interface indices as labels but keeping physical distances
    ax5.set_xlim(interfaces[1] - 0.1, max(interfaces) + 0.1)
    # Set the physical positions of interfaces on the x-axis
    ax5.set_xticks(interfaces)
    # But label them with their indices
    ax5.set_xticklabels([f"{i}" for i in range(n_interfaces)])
    
    # Add explanatory text about the dashed lines
    ax5.text(0.02, 0.02, ref_text, transform=ax5.transAxes, fontsize=9,
             bbox=dict(facecolor='white', alpha=0.8))
    
    # Add a legend with reasonable size
    ax5.legend(title='Target Interface', loc='best', fontsize=9)
    
    # Plot 2.3: Forward Memory Retention
    ax6 = fig2.add_subplot(gs2[1, 0])

    # Calculate memory retention using simplified approach
    memory_index = calculate_memory_effect_index(q_probs, q_weights)

    # Prepare data for forward plot
    valid_k = [k for k in range(1, n_interfaces) if not np.isnan(memory_index['forward_variation'][k])]
    valid_variation = [memory_index['forward_variation'][k] for k in valid_k]
    valid_positions = [interfaces[k] for k in valid_k]
    valid_colors = [forward_colors[k-1] for k in valid_k]
    valid_counts = [memory_index['forward_sample_sizes'][k] for k in valid_k]

    # Calculate mean difference with diffusive reference for each target interface
    forward_mean_diff = np.zeros(n_interfaces)
    forward_mean_diff.fill(np.nan)
    
    for k in range(1, n_interfaces):
        diffs = []
        for i in range(k-1):  # Skip adjacent interface (i=k-1)
                # Only consider non-adjacent transitions with enough samples
                if not np.isnan(q_probs[i, k]) and q_weights[i, k] > 5:
                    diffs.append(abs(q_probs[i, k] - diff_ref[i, k]))
        
        if diffs:
                forward_mean_diff[k] = np.mean(diffs) * 100  # Convert to percentage
    
    # Extract valid mean differences to plot
    valid_mean_diff = [forward_mean_diff[k] if not np.isnan(forward_mean_diff[k]) else 0 for k in valid_k]

    if valid_k:
        # Create a twin axis for the memory retention plot
        ax6_twin = ax6.twinx()
        
        # Create bar plot for variation
        bars = ax6.bar(valid_positions, valid_variation, color=valid_colors, alpha=0.7, 
                            width=np.mean(np.diff(interfaces))*0.7)  # Use average interface spacing for width
        
        # Add line plot for mean differences
        line = ax6_twin.plot(valid_positions, valid_mean_diff, 'o--', color='red', 
                                    linewidth=2, markersize=8, label='Mean |Δq|')
        
        # Add annotations showing variation and sample size
        for pos, var, count in zip(valid_positions, valid_variation, valid_counts):
                ax6.text(pos, var + 0.5, f"SD: {var:.1f}%\nn={count}", ha='center', fontsize=9)
        
        # Configure main plot
        ax6.set_xticks(interfaces)
        ax6.set_xticklabels([f"{i}" for i in range(n_interfaces)])
        ax6.set_xlabel('Target Interface k')
        ax6.set_ylabel('Memory Effect (Std. Dev. %)', color='C0')
        ax6.tick_params(axis='y', labelcolor='C0')
        ax6.set_title('Forward Memory Retention: Variation in Crossing Probabilities', fontsize=12)
        
        # Configure twin axis
        ax6_twin.set_ylabel('Mean |Δq| (%)', color='red')
        ax6_twin.tick_params(axis='y', labelcolor='red')
        
        # Set reasonable y-limits
        max_y = max(10.0, max(valid_variation) * 1.2) if valid_variation else 10.0
        ax6.set_ylim(0, max_y)
        
        max_y_twin = max(10.0, max(valid_mean_diff) * 1.2) if valid_mean_diff else 10.0
        ax6_twin.set_ylim(0, max_y_twin)
        
        # Set x limits based on the valid data points rather than all interfaces
        if len(valid_positions) > 0:
                padding = np.mean(np.diff(interfaces)) if len(interfaces) > 1 else 0.5
                ax6.set_xlim(min(valid_positions) - padding/2, max(valid_positions) + padding/2)
        
        # Create a combined legend
        custom_lines = [
                Line2D([0], [0], color='black', lw=0, marker='s', markersize=10, markerfacecolor='C0', alpha=0.7),
                Line2D([0], [0], color='red', lw=2, marker='o', markersize=6)
        ]
        ax6.legend(custom_lines, ['Std. Dev. (%)', 'Mean |Δq| (%)'], loc='upper left')
        
        # Add explanatory text
        mem_text = """
        Memory Metrics:
        • Std. Dev.: Variation in transition probabilities from different starting points 
        • Mean |Δq|: Average deviation from diffusive reference (excluding adjacent transitions)
        • Higher values indicate stronger memory effects
        """
        ax6.text(0.02, 0.95, mem_text, transform=ax6.transAxes, fontsize=9,
                    bbox=dict(facecolor='white', alpha=0.8), va='top')
    else:
        ax6.text(0.5, 0.5, "Insufficient data for forward memory retention analysis", 
                ha='center', va='center', transform=ax6.transAxes)
        ax6.set_xticks(interfaces)
        ax6.set_xticklabels([f"{i}" for i in range(n_interfaces)])

    # Plot 2.4: Backward Memory Retention
    ax7 = fig2.add_subplot(gs2[1, 1])

    # Prepare data for backward plot
    valid_k = [k for k in range(n_interfaces-1) if not np.isnan(memory_index['backward_variation'][k])]
    valid_variation = [memory_index['backward_variation'][k] for k in valid_k]
    valid_positions = [interfaces[k] for k in valid_k]
    valid_colors = [backward_colors[k] for k in valid_k]
    valid_counts = [memory_index['backward_sample_sizes'][k] for k in valid_k]
    
    # Calculate mean difference with diffusive reference for each target interface
    backward_mean_diff = np.zeros(n_interfaces)
    backward_mean_diff.fill(np.nan)
    
    for k in range(n_interfaces-1):
        diffs = []
        for i in range(k+2, n_interfaces):  # Skip adjacent interface (i=k+1)
                # Only consider non-adjacent transitions with enough samples
                if not np.isnan(q_probs[i, k]) and q_weights[i, k] > 5:
                    diffs.append(abs(q_probs[i, k] - diff_ref[i, k]))
        
        if diffs:
                backward_mean_diff[k] = np.mean(diffs) * 100  # Convert to percentage
    
    # Extract valid mean differences to plot
    valid_mean_diff = [backward_mean_diff[k] if not np.isnan(backward_mean_diff[k]) else 0 for k in valid_k]

    if valid_k:
        # Create a twin axis for the memory retention plot
        ax7_twin = ax7.twinx()
        
        # Create bar plot using interface physical positions
        bars = ax7.bar(valid_positions, valid_variation, color=valid_colors, alpha=0.7,
                            width=np.mean(np.diff(interfaces))*0.7)  # Use average interface spacing for width
        
        # Add line plot for mean differences
        line = ax7_twin.plot(valid_positions, valid_mean_diff, 'o--', color='red', 
                                    linewidth=2, markersize=8, label='Mean |Δq|')
        
        # Add annotations showing variation and sample size
        for pos, var, count in zip(valid_positions, valid_variation, valid_counts):
                ax7.text(pos, var + 0.5, f"SD: {var:.1f}%\nn={count}", ha='center', fontsize=9)
        
        # Configure plot
        ax7.set_xticks(interfaces)
        ax7.set_xticklabels([f"{i}" for i in range(n_interfaces)])
        ax7.set_xlabel('Target Interface k')
        ax7.set_ylabel('Memory Effect (Std. Dev. %)', color='C0')
        ax7.tick_params(axis='y', labelcolor='C0')
        ax7.set_title('Backward Memory Retention: Variation in Crossing Probabilities', fontsize=12)
        
        # Configure twin axis
        ax7_twin.set_ylabel('Mean |Δq| (%)', color='red')
        ax7_twin.tick_params(axis='y', labelcolor='red')
        
        # Set reasonable y-limits
        max_y = max(10.0, max(valid_variation) * 1.2) if valid_variation else 10.0
        ax7.set_ylim(0, max_y)
        
        max_y_twin = max(10.0, max(valid_mean_diff) * 1.2) if valid_mean_diff else 10.0
        ax7_twin.set_ylim(0, max_y_twin)
        
        # Set x limits based on the valid data points rather than all interfaces
        if len(valid_positions) > 0:
                padding = np.mean(np.diff(interfaces)) if len(interfaces) > 1 else 0.5
                ax7.set_xlim(min(valid_positions) - padding/2, max(valid_positions) + padding/2)
        
        # Create a combined legend
        custom_lines = [
                Line2D([0], [0], color='black', lw=0, marker='s', markersize=10, markerfacecolor='C0', alpha=0.7),
                Line2D([0], [0], color='red', lw=2, marker='o', markersize=6)
        ]
        ax7.legend(custom_lines, ['Std. Dev. (%)', 'Mean |Δq| (%)'], loc='upper left')
        
        # Add explanatory text - same as for forward plot
        ax7.text(0.02, 0.95, mem_text, transform=ax7.transAxes, fontsize=9,
                    bbox=dict(facecolor='white', alpha=0.8), va='top')
    else:
        ax7.text(0.5, 0.5, "Insufficient data for backward memory retention analysis", 
                ha='center', va='center', transform=ax7.transAxes)
        ax7.set_xticks(interfaces)
        ax7.set_xticklabels([f"{i}" for i in range(n_interfaces)])
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig2.suptitle('TIS Memory Effect Analysis - Transition Probabilities and Memory Retention', fontsize=14)
                                                       
    # ================ Figure 3: Memory Decay Profiles and Additional Visualizations ================
    fig3 = plt.figure(figsize=(18, 14))
    gs3 = gridspec.GridSpec(2, 2)
    
    # Plot 3.1: Memory Decay with Distance (Forward transitions)
    ax8 = fig3.add_subplot(gs3[0, 0])
    
    # Calculate memory effect as deviation from diffusive reference
    memory_data = np.zeros((n_interfaces-1, n_interfaces-1))
    memory_data.fill(np.nan)
    
    for start_i in range(n_interfaces-1):
        for target_k in range(start_i+1, n_interfaces):
            # Calculate distance between interfaces (in index space)
            distance = target_k - start_i
            # Memory effect is deviation from diffusive reference
            if not np.isnan(q_probs[start_i, target_k]) and q_weights[start_i, target_k] > 5:
                memory_data[start_i, distance-1] = abs(q_probs[start_i, target_k] - diff_ref[start_i, target_k])
    
    # Plot heatmap
    sns.heatmap(memory_data, cmap='viridis', ax=ax8, 
                cbar_kws={'label': '|Probability - Diffusive Reference|'})
    
    ax8.set_xlabel('Interface Distance (k - i)')
    ax8.set_ylabel('Starting Interface i')
    ax8.set_title('Memory Effect Decay with Distance (Forward L→R)', fontsize=12)
    ax8.set_xticks(np.arange(n_interfaces-1) + 0.5)
    ax8.set_xticklabels(np.arange(1, n_interfaces))
    ax8.set_yticks(np.arange(n_interfaces-1) + 0.5)
    ax8.set_yticklabels([f"{i}" for i in range(n_interfaces-1)])
    
    # Plot 3.2: Memory Decay with Physical Distance
    ax9 = fig3.add_subplot(gs3[0, 1])
    
    # Plot 3.3: Memory Decay with Distance (Backward transitions)
    ax10 = fig3.add_subplot(gs3[1, 0])
    
    # Calculate memory effect as deviation from diffusive reference
    backward_memory_data = np.zeros((n_interfaces-1, n_interfaces-1))
    backward_memory_data.fill(np.nan)
    
    for start_i in range(1, n_interfaces):
        for target_k in range(start_i):
            # Calculate distance between interfaces
            distance = start_i - target_k
            # Memory effect is deviation from diffusive reference
            if not np.isnan(q_probs[start_i, target_k]) and q_weights[start_i, target_k] > 5:
                backward_memory_data[start_i-1, distance-1] = abs(q_probs[start_i, target_k] - diff_ref[start_i, target_k])
    
    # Plot heatmap
    sns.heatmap(backward_memory_data, cmap='viridis', ax=ax10, 
                cbar_kws={'label': '|Probability - Diffusive Reference|'})
    
    ax10.set_xlabel('Interface Distance (i - k)')
    ax10.set_ylabel('Starting Interface i')
    ax10.set_title('Memory Effect Decay with Distance (Backward R→L)', fontsize=12)
    ax10.set_xticks(np.arange(n_interfaces-1) + 0.5)
    ax10.set_xticklabels(np.arange(1, n_interfaces))
    ax10.set_yticks(np.arange(n_interfaces-1) + 0.5)
    ax10.set_yticklabels([f"{i+1}" for i in range(n_interfaces-1)])
    
    # Plot 3.4: Deviations from Diffusive Behavior in Transitions
    ax11 = fig3.add_subplot(gs3[1, 1])

    plot_destination_bias(p, interfaces, ax_forward=ax11, ax_backward=ax9)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig3.suptitle('TIS Memory Effect Analysis - Decay Profiles and Additional Metrics', fontsize=14)
    
    return fig1, fig2, fig3

def plot_destination_bias(p, interfaces=None, ax_forward=None, ax_backward=None):
    """
    Create separate visualizations showing the average destination interface for forward 
    and backward transitions.
    
    Parameters:
    -----------
    p : numpy.ndarray
        Transition probability matrix where p[i,j] is the probability of
        transitioning from interface i to interface j
    interfaces : list or array, optional
        The positions of the interfaces along the reaction coordinate.
        If None, uses sequential indices.
    ax_forward : matplotlib.Axes, optional
        Axes to plot forward transitions on. If None, creates a new figure and axes.
    ax_backward : matplotlib.Axes, optional
        Axes to plot backward transitions on. If None, creates a new figure and axes.
        
    Returns:
    --------
    tuple
        (ax_forward, ax_backward): The axes containing the forward and backward plots
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    n_interfaces = p.shape[0]
    
    if interfaces is None:
        interfaces = list(range(n_interfaces))
        is_equidistant = True
    else:
        # Check if interfaces are equidistant
        if len(interfaces) > 2:
            diffs = np.diff(interfaces)
            is_equidistant = np.allclose(diffs, diffs[0], rtol=0.05)
        else:
            is_equidistant = True
    
    # Create new figures if axes not provided
    if ax_forward is None:
        _, ax_forward = plt.subplots(figsize=(10, 6))
    if ax_backward is None:
        _, ax_backward = plt.subplots(figsize=(10, 6))
    
    # Calculate forward average destinations (i→j where i<j)
    forward_destinations = np.zeros(n_interfaces)
    forward_valid = np.zeros(n_interfaces, dtype=bool)
    
    for i in range(n_interfaces-1):  # Skip last interface (no forward transitions)
        # Extract forward transitions (j > i)
        forward_probs = p[i, i+1:]
        forward_targets = np.arange(i+1, n_interfaces)
        
        # Calculate weighted average if there are transitions
        if np.sum(forward_probs) > 0:
            forward_destinations[i] = np.sum(forward_targets * forward_probs) / np.sum(forward_probs)
            forward_valid[i] = True
        else:
            forward_destinations[i] = np.nan  # No valid forward transitions
    
    # Calculate backward average destinations (i→j where i>j)
    backward_destinations = np.zeros(n_interfaces)
    backward_valid = np.zeros(n_interfaces, dtype=bool)
    
    for i in range(1, n_interfaces):  # Skip first interface (no backward transitions)
        # Extract backward transitions (j < i)
        backward_probs = p[i, :i]
        backward_targets = np.arange(i)
        
        # Calculate weighted average if there are transitions
        if np.sum(backward_probs) > 0:
            backward_destinations[i] = np.sum(backward_targets * backward_probs) / np.sum(backward_probs)
            backward_valid[i] = True
        else:
            backward_destinations[i] = np.nan  # No valid backward transitions
    
    # Calculate expected destinations
    # For forward transitions: expected = i + expected_jump
    # For backward transitions: expected = i - expected_jump
    # Calculate expected destinations
    forward_expected = np.zeros_like(forward_destinations)
    backward_expected = np.zeros_like(backward_destinations)
    
    for i in range(n_interfaces):
        if forward_valid[i]:
            # For forward transitions from i, calculate expected destination
            # For a diffusive system, this would be the probability-weighted average
            # of all possible destinations j where j > i
            probs = np.zeros(n_interfaces)
            for j in range(i+1, n_interfaces):
                # In a diffusive system, probability decreases with distance
                # We use a simple geometric series: p(j) = p(i+1) * r^(j-i-1)
                # where r is a decay factor (e.g., 0.5)
                dist_factor = 0.5 ** (j - i - 1)
                probs[j] = dist_factor
            
            # Normalize probabilities
            if np.sum(probs) > 0:
                probs = probs / np.sum(probs)
                # Calculate expected destination
                forward_expected[i] = np.sum(np.arange(n_interfaces) * probs)
        
        if backward_valid[i]:
            # For backward transitions from i, calculate expected destination
            # For a diffusive system, this would be the probability-weighted average
            # of all possible destinations j where j < i
            probs = np.zeros(n_interfaces)
            for j in range(i):
                # Similar decay factor for backward transitions
                dist_factor = 0.5 ** (i - j - 1)
                probs[j] = dist_factor
            
            # Normalize probabilities
            if np.sum(probs) > 0:
                probs = probs / np.sum(probs)
                # Calculate expected destination
                backward_expected[i] = np.sum(np.arange(n_interfaces) * probs)
    
    # Calculate bias
    forward_bias = forward_destinations - forward_expected
    backward_bias = backward_destinations - backward_expected
    
    # Get x values for plotting
    x_values = np.arange(n_interfaces) if is_equidistant else np.array(interfaces)
    
    # Plot forward transitions
    plot_directional_bias(ax_forward, x_values, forward_destinations, forward_expected, 
                       forward_bias, forward_valid, 'Forward Transitions (i→j where i<j)', 'higher',
                       interfaces, is_equidistant)
    
    # Plot backward transitions
    plot_directional_bias(ax_backward, x_values, backward_destinations, backward_expected, 
                       backward_bias, backward_valid, 'Backward Transitions (i→j where i>j)', 'lower',
                       interfaces, is_equidistant)
    
    return ax_forward, ax_backward

def plot_directional_bias(ax, x_values, avg_destinations, expected_destinations, bias, valid_mask,
                       title, direction, interfaces, is_equidistant):
    """Helper function to create a directional bias plot on the given axes"""
    import numpy as np
    
    n_interfaces = len(x_values)
    
    # Plot expected destination
    ax.plot(x_values[valid_mask], expected_destinations[valid_mask], '--', color='gray', 
           label='Expected destination')
    
    # Plot actual average destination
    ax.plot(x_values[valid_mask], avg_destinations[valid_mask], 'o-', color='blue', linewidth=2,
           markersize=8, label='Actual average')
    
    # Add annotations showing the bias
    for i, valid in enumerate(valid_mask):
        if valid:
            bias_text = f"{bias[i]:.2f}"
            text_color = 'red' if bias[i] > 0.2 else ('blue' if bias[i] < -0.2 else 'black')
            ax.annotate(bias_text, 
                      xy=(x_values[i], avg_destinations[i]), 
                      xytext=(0, 10 if bias[i] > 0 else -15),
                      textcoords='offset points',
                      ha='center', va='center',
                      color=text_color, fontsize=9,
                      bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8))

    # Add shaded region showing significant bias
    valid_expected = expected_destinations[valid_mask]
    
    # Configure the plot
    ax.set_xlabel('Starting Interface i')
    ax.set_ylabel('Average Destination Interface')
    ax.set_title(f'Average Destination Analysis: {title}', fontsize=12)
    
    # Set appropriate axis limits
    valid_y = avg_destinations[valid_mask]
    if len(valid_y) > 0:
        y_min, y_max = np.nanmin(valid_y), np.nanmax(valid_y)
        expected_min, expected_max = np.nanmin(valid_expected), np.nanmax(valid_expected)
        
        y_min = min(y_min, expected_min)
        y_max = max(y_max, expected_max)
        
        y_range = y_max - y_min
        buffer = 0.5 if is_equidistant else (interfaces[1] - interfaces[0]) / 2
        ax.set_xlim(np.min(x_values) - buffer, np.max(x_values) + buffer)
        ax.set_ylim(y_min - y_range*0.1, y_max + y_range*0.1)
    
    # Add gridlines for easier reading
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Create a better legend
    handles, labels = ax.get_legend_handles_labels()
    keep_indices = [0, 1, 3]  # Just keep the main curves and diffusive range
    ax.legend([handles[i] for i in keep_indices if i < len(handles)], 
             [labels[i] for i in keep_indices if i < len(labels)],
             loc='best', fontsize=10)
    
    # Add explanatory text
    if direction == 'higher':
        info_text = """
        Average destination for forward transitions (i→j where i<j):
        • Numbers indicate deviation from expected destination
        • Positive values (red): Paths go further than expected
        • Negative values (blue): Paths go less far than expected
        """
    else:  # 'lower'
        info_text = """
        Average destination for backward transitions (i→j where i>j):
        • Numbers indicate deviation from expected destination
        • Positive values (red): Paths go less far back than expected
        • Negative values (blue): Paths go further back than expected
        """
    
    ax.text(0.02, 0.02, info_text, transform=ax.transAxes, fontsize=9,
          bbox=dict(facecolor='white', alpha=0.9, boxstyle='round'), va='bottom')

def calculate_memory_effect_index(q_probs, q_weights, min_samples=5):
    """
    Calculate a simplified memory effect index based solely on the variation in conditional 
    crossing probabilities without normalizing or weighting by sample size.
    
    Parameters:
    -----------
    q_probs : numpy.ndarray
        Matrix of conditional crossing probabilities where q_probs[i,k] is the probability
        that a path starting at interface i and reaching k-1 (for i<k) or k+1 (for i>k)
        will reach interface k.
    q_weights : numpy.ndarray
        Matrix of sample counts for each q value.
    min_samples : int, optional
        Minimum number of samples required to consider a q value valid.
        
    Returns:
    --------
    memory_index : dict
        Dictionary containing:
        - 'forward_variation': Standard deviation of forward probabilities for each target
        - 'backward_variation': Standard deviation of backward probabilities for each target
        - 'forward_sample_sizes': Number of samples used for forward calculations
        - 'backward_sample_sizes': Number of samples used for backward calculations
    """
    n_interfaces = q_probs.shape[0]
    
    # Initialize result arrays
    forward_variation = np.zeros(n_interfaces)
    forward_variation.fill(np.nan)
    forward_sample_sizes = np.zeros(n_interfaces, dtype=int)
    
    backward_variation = np.zeros(n_interfaces)
    backward_variation.fill(np.nan)
    backward_sample_sizes = np.zeros(n_interfaces, dtype=int)
    
    # Calculate forward memory effect index (for targets k > 0)
    for k in range(1, n_interfaces):
        # Collect q values for paths reaching target k from different starting interfaces
        q_values = []
        weights = []
        
        for i in range(k-1):  # Skip adjacent interface (i=k-1)
            if not np.isnan(q_probs[i, k]) and q_weights[i, k] >= min_samples:
                q_values.append(q_probs[i, k])
                weights.append(q_weights[i, k])
        
        if len(q_values) >= 2:  # Need at least 2 starting points for meaningful variation
            q_values = np.array(q_values)
            weights = np.array(weights)
            
            # Calculate total sample size
            total_samples = np.sum(weights)
            forward_sample_sizes[k] = total_samples
            
            # Simply use standard deviation as the memory effect index (as a percentage)
            forward_variation[k] = np.std(q_values/np.mean(q_values)) * 100  # Convert to percentage
    
    # Calculate backward memory effect index (for targets k < n_interfaces-1)
    for k in range(n_interfaces - 1):
        # Collect q values for paths reaching target k from different starting interfaces
        q_values = []
        weights = []
        
        for i in range(k+2, n_interfaces):  # Skip adjacent interface (i=k+1)
            if not np.isnan(q_probs[i, k]) and q_weights[i, k] >= min_samples:
                q_values.append(q_probs[i, k])
                weights.append(q_weights[i, k])
        
        if len(q_values) >= 2:  # Need at least 2 starting points for meaningful variation
            q_values = np.array(q_values)
            weights = np.array(weights)
            
            # Calculate total sample size
            total_samples = np.sum(weights)
            backward_sample_sizes[k] = total_samples
            
            # Simply use standard deviation as the memory effect index (as a percentage)
            backward_variation[k] = np.std(q_values/np.mean(q_values)) * 100  # Convert to percentage
    
    # Package results into a dictionary
    memory_index = {
        'forward_variation': forward_variation,
        'backward_variation': backward_variation,
        'forward_sample_sizes': forward_sample_sizes,
        'backward_sample_sizes': backward_sample_sizes
    }
    
    return memory_index

def calculate_diffusive_reference(interfaces, q_matrix, q_weights=None, min_samples=5):
    """
    Calculate diffusive reference probabilities for non-equidistant interfaces.
    
    Accounts for the fact that in TIS, conditional crossing probabilities for
    directly adjacent interfaces (q(i,i+1) and q(i,i-1)) are always 1.0 due to
    the nature of path sampling. Instead uses transition probabilities or non-adjacent
    conditional probabilities to estimate free energy differences.
    
    Parameters:
    -----------
    interfaces : list or array
        The positions of the interfaces along the reaction coordinate
    q_matrix : numpy.ndarray
        Matrix of conditional crossing probabilities
    q_weights : numpy.ndarray, optional
        Matrix of sample counts for each q_matrix value
    min_samples : int, optional
        Minimum number of samples required to consider a q value valid
        
    Returns:
    --------
    diff_ref : numpy.ndarray
        A matrix of diffusive reference probabilities for each i,k pair
    """
    n_interfaces = len(interfaces)
    diff_ref = np.zeros((n_interfaces, n_interfaces))
    diff_ref.fill(np.nan)
    
    # Estimate free energy differences between adjacent interfaces
    # We can't use q(i,i+1) directly because it's always 1.0 in TIS
    delta_G = estimate_free_energy_differences(interfaces, q_matrix, q_weights, min_samples)
    
    # For diagonal elements (self-transitions), probability is 0
    for i in range(n_interfaces):
        diff_ref[i, i] = 0.0
    
    # Fill in reference probabilities for adjacent interfaces
    for i in range(n_interfaces-1):
        if not np.isnan(delta_G[i, i+1]):
            # Forward probability i→i+1
            diff_ref[i, i+1] = 1.0 / (1.0 + np.exp(delta_G[i, i+1]))
            # Backward probability i+1→i
            diff_ref[i+1, i] = 1.0 / (1.0 + np.exp(delta_G[i+1, i]))
        else:
            # Fallback to 0.5 if we can't estimate from data
            diff_ref[i, i+1] = 0.5
            diff_ref[i+1, i] = 0.5
    
    # Calculate non-adjacent transition reference probabilities
    for i in range(n_interfaces):
        for k in range(n_interfaces):
            if abs(i - k) >= 2:  # Non-adjacent interfaces
                if i < k:  # Forward transitions (i → k)
                    # Calculate cumulative free energy difference from i to k
                    cum_delta_G = 0.0
                    valid_path = True
                    
                    for j in range(i, k):
                        if np.isnan(delta_G[j, j+1]):
                            valid_path = False
                            break
                        cum_delta_G += delta_G[j, j+1]
                    
                    if valid_path:
                        # Diffusive reference probability based on cumulative free energy difference
                        diff_ref[i, k] = 1.0 / (1.0 + np.exp(cum_delta_G))
                    else:
                        # Check if we have direct measurement
                        valid_q = (not np.isnan(q_matrix[i, k]) and 
                                  (q_weights is None or q_weights[i, k] >= min_samples))
                        if valid_q:
                            # Compare direct measurement to product of intermediate steps
                            product_ref = 1.0
                            for j in range(i, k):
                                if np.isnan(diff_ref[j, j+1]):
                                    product_ref = 0.5 ** (k-i)  # Fallback
                                    break
                                product_ref *= diff_ref[j, j+1]
                            diff_ref[i, k] = product_ref
                        else:
                            diff_ref[i, k] = 0.5 ** (k-i)  # Fallback to 0.5^distance
                        
                elif i > k:  # Backward transitions (i → k)
                    # Calculate cumulative free energy difference from i to k
                    cum_delta_G = 0.0
                    valid_path = True
                    
                    for j in range(i, k, -1):
                        if np.isnan(delta_G[j, j-1]):
                            valid_path = False
                            break
                        cum_delta_G += delta_G[j, j-1]
                    
                    if valid_path:
                        # Diffusive reference probability based on cumulative free energy difference
                        diff_ref[i, k] = 1.0 / (1.0 + np.exp(cum_delta_G))
                    else:
                        # Check if we have direct measurement
                        valid_q = (not np.isnan(q_matrix[i, k]) and 
                                  (q_weights is None or q_weights[i, k] >= min_samples))
                        if valid_q:
                            # Compare direct measurement to product of intermediate steps
                            product_ref = 1.0
                            for j in range(i, k, -1):
                                if np.isnan(diff_ref[j, j-1]):
                                    product_ref = 0.5 ** (i-k)  # Fallback
                                    break
                                product_ref *= diff_ref[j, j-1]
                            diff_ref[i, k] = product_ref
                        else:
                            diff_ref[i, k] = 0.5 ** (i-k)  # Fallback to 0.5^distance
    
    return diff_ref

def calculate_diffusive_reference_spacing(interfaces):
    """
    Calculate diffusive reference probabilities based on local interface spacing.
    
    This function provides a reference that accounts for the geometric effects
    of non-equidistant interface placement. The conditional probability q(i,k)
    always refers to the probability of a trajectory that has reached interface k-1
    (for forward, i<k) or k+1 (for backward, i>k) to reach interface k.
    
    For forward transitions (i<k), the probability depends on the distances between
    interfaces k-2, k-1 and k. For backward transitions (i>k), it depends on
    interfaces k, k+1 and k+2.
    
    Parameters:
    -----------
    interfaces : list or array
        The positions of the interfaces along the reaction coordinate
        
    Returns:
    --------
    diff_ref : numpy.ndarray
        A matrix of diffusive reference probabilities for each i,k pair
    """
    n_interfaces = len(interfaces)
    diff_ref = np.zeros((n_interfaces, n_interfaces))
    diff_ref.fill(np.nan)
    
    # For diagonal elements (self-transitions), probability is 0
    for i in range(n_interfaces):
        diff_ref[i, i] = 0.0
    
    # Calculate reference values for forward transitions (moving right)
    forward_probs = np.zeros(n_interfaces)  # Probability to go from k-1 to k
    forward_probs.fill(np.nan)
    
    # Calculate reference values for backward transitions (moving left)
    backward_probs = np.zeros(n_interfaces)  # Probability to go from k+1 to k
    backward_probs.fill(np.nan)
    
    if isinstance(interfaces[0], (int, float)):
        # For forward transitions q(i,k) with i<k:
        # When at interface k-1, compare distance from k-2 to k-1 vs k-1 to k
        for k in range(2, n_interfaces):
            # Distance from k-2 to k-1
            d_prev = interfaces[k-1] - interfaces[k-2]
            # Distance from k-1 to k
            d_next = interfaces[k] - interfaces[k-1]
            
            if d_prev > 0 and d_next > 0:
                # Probability to continue in forward direction is inversely 
                # proportional to the ratio of distances
                forward_probs[k] = d_prev / (d_prev + d_next)
        
        # For backward transitions q(i,k) with i>k:
        # When at interface k+1, compare distance from k to k+1 vs k+1 to k+2
        for k in range(n_interfaces-3):
            # Distance from k to k+1
            d_next = interfaces[k+1] - interfaces[k]
            # Distance from k+1 to k+2
            d_prev = interfaces[k+2] - interfaces[k+1]
            
            if d_prev > 0 and d_next > 0:
                # Probability to go back is inversely proportional to the
                # ratio of distances
                backward_probs[k] = d_prev / (d_prev + d_next)
    
    # Fill in default values (0.5) for edge cases and missing values
    forward_probs[np.isnan(forward_probs)] = 0.5
    backward_probs[np.isnan(backward_probs)] = 0.5
    
    # Handle special case of first forward probability
    forward_probs[1] = 0.5  # No distance k-2 available, use 0.5
    
    # Handle special case of last backward probabilities
    backward_probs[n_interfaces-2] = 0.5  # No distance k+2 available, use 0.5
    backward_probs[n_interfaces-1] = 0.5  # Edge case
    
    # Now populate the diff_ref matrix
    for i in range(n_interfaces):
        for k in range(n_interfaces):
            if i < k:  # Forward transition (i → k)
                # q(i,k) is the probability that a path starting at i
                # and having reached k-1 will reach k
                diff_ref[i, k] = forward_probs[k]
            elif i > k:  # Backward transition (i → k)
                # q(i,k) is the probability that a path starting at i
                # and having reached k+1 will reach k
                diff_ref[i, k] = backward_probs[k]
    
    return diff_ref

def estimate_free_energy_differences(interfaces, q_matrix, q_weights=None, min_samples=5):
    """
    Estimate free energy differences between interfaces from conditional crossing probabilities,
    taking into account physical distances between interfaces.
    
    Parameters:
    -----------
    interfaces : list or array
        The positions of the interfaces along the reaction coordinate
    q_matrix : numpy.ndarray
        Matrix of conditional crossing probabilities
    q_weights : numpy.ndarray, optional
        Matrix of sample counts for each q_matrix value
    min_samples : int, optional
        Minimum number of samples required to consider a q value valid
        
    Returns:
    --------
    delta_G : numpy.ndarray
        Matrix of free energy differences between interfaces
    """
    n_interfaces = len(interfaces)
    delta_G = np.zeros((n_interfaces, n_interfaces))
    delta_G.fill(np.nan)
    
    # Check if interfaces have physical values
    has_physical_distances = isinstance(interfaces[0], (int, float)) and len(interfaces) > 1
    
    # Calculate interface spacings if available
    if has_physical_distances:
        spacings = np.diff(interfaces)
        mean_spacing = np.mean(spacings)
        relative_spacings = spacings / mean_spacing if mean_spacing > 0 else np.ones_like(spacings)
    
    # First try to estimate delta_G from non-adjacent q-values with distance=2
    for i in range(n_interfaces-2):
        # Skip cases where i=0 or i+2=n_interfaces-1 (boundary states)
        if i == 0 or i+2 >= n_interfaces-1:
            continue
            
        # Forward transition i -> i+2
        valid_forward = (not np.isnan(q_matrix[i, i+2]) and 
                        (q_weights is None or q_weights[i, i+2] >= min_samples) and
                        q_matrix[i, i+2] > 0 and q_matrix[i, i+2] < 1)
                        
        # Backward transition i+2 -> i
        valid_backward = (not np.isnan(q_matrix[i+2, i]) and 
                         (q_weights is None or q_weights[i+2, i] >= min_samples) and
                         q_matrix[i+2, i] > 0 and q_matrix[i+2, i] < 1)
        
        if valid_forward and valid_backward:
            # For a diffusive system, the ratio of these probabilities gives information
            # about the free energy difference between i and i+1
            ratio = q_matrix[i, i+2] / q_matrix[i+2, i]
            
            # Total free energy difference between i and i+2
            dG_total = -np.log(ratio)
            
            if has_physical_distances:
                # Distribute free energy proportionally to physical distances
                dist_i_to_i1 = spacings[i]
                dist_i1_to_i2 = spacings[i+1]
                total_dist = dist_i_to_i1 + dist_i1_to_i2
                
                if total_dist > 0:
                    # Distribute the free energy change proportionally to distances
                    dG_i_to_i1 = dG_total * (dist_i_to_i1 / total_dist)
                    dG_i1_to_i2 = dG_total * (dist_i1_to_i2 / total_dist)
                else:
                    # Equal distribution if distances are invalid
                    dG_i_to_i1 = dG_total / 2
                    dG_i1_to_i2 = dG_total / 2
            else:
                # Without physical distance, assume equal distribution
                dG_i_to_i1 = dG_total / 2
                dG_i1_to_i2 = dG_total / 2
                
            # Assign these values
            delta_G[i, i+1] = dG_i_to_i1
            delta_G[i+1, i] = -dG_i_to_i1
            delta_G[i+1, i+2] = dG_i1_to_i2
            delta_G[i+2, i+1] = -dG_i1_to_i2
    
    # Next, try to estimate from adjacent probabilities in diffusive reference
    # This is more similar to calculate_diffusive_reference_spacing approach
    if has_physical_distances:
        # Similar to diffusive_reference_spacing logic
        for i in range(n_interfaces-1):
            if np.isnan(delta_G[i, i+1]):
                # For adjacent interfaces with missing free energy estimates
                # Use ratio of adjacent interface distances
                
                # Forward direction logic - similar to diffusive_reference_spacing
                if i > 0 and i+1 < n_interfaces-1:  # Have previous and next interfaces
                    # Distance from i-1 to i
                    d_prev = interfaces[i] - interfaces[i-1]
                    # Distance from i to i+1
                    d_next = interfaces[i+1] - interfaces[i]
                    
                    if d_prev > 0 and d_next > 0:
                        # Calculate forward probability based on distance ratios
                        p_forward = d_prev / (d_prev + d_next)
                        # Convert probability to free energy
                        delta_G[i, i+1] = -np.log(p_forward / (1 - p_forward))
                        delta_G[i+1, i] = -delta_G[i, i+1]
                        continue
                
                # Backward direction logic - similar to diffusive_reference_spacing
                if i > 0 and i+2 < n_interfaces:  # Can use backward reference
                    # Distance from i to i+1
                    d_curr = interfaces[i+1] - interfaces[i]
                    # Distance from i+1 to i+2
                    d_next = interfaces[i+2] - interfaces[i+1]
                    
                    if d_curr > 0 and d_next > 0:
                        # Calculate backward probability based on distance ratios
                        p_backward = d_next / (d_next + d_curr)
                        # Convert probability to free energy
                        delta_G[i+1, i] = -np.log(p_backward / (1 - p_backward))
                        delta_G[i, i+1] = -delta_G[i+1, i]
                        continue
    
    # Fill in any remaining gaps with estimates from physical distance
    default_barrier = 0.01  # Small non-zero default barrier (in kT units)
    
    for i in range(n_interfaces-1):
        if np.isnan(delta_G[i, i+1]):
            if has_physical_distances:
                # Scale default barrier by relative interface spacing
                scale = relative_spacings[i]
                delta_G[i, i+1] = default_barrier * scale
                delta_G[i+1, i] = -delta_G[i, i+1]
            else:
                # No physical information, use default barrier
                delta_G[i, i+1] = default_barrier
                delta_G[i+1, i] = -default_barrier
    
    return delta_G

def plot_free_energy_landscape(interfaces, q_matrix, q_weights=None, min_samples=5):
    """
    Plot the free energy differences between interfaces.
    
    This function visualizes the free energy landscape used in the diffusive reference
    calculation, showing both the free energy differences between adjacent interfaces
    and the cumulative free energy profile.
    
    Parameters:
    -----------
    interfaces : list or array
        The positions of the interfaces along the reaction coordinate
    q_matrix : numpy.ndarray
        Matrix of conditional crossing probabilities
    q_weights : numpy.ndarray, optional
        Matrix of sample counts for each q_matrix value
    min_samples : int, optional
        Minimum number of samples required to consider a q value valid
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure containing the free energy landscape plots
    """
    from matplotlib.patches import Patch
    # Estimate free energy differences
    delta_G = estimate_free_energy_differences(interfaces, q_matrix, q_weights, min_samples)
    
    # Calculate cumulative free energy profile (setting first interface as reference point)
    cumulative_G = np.zeros(len(interfaces))
    for i in range(1, len(interfaces)):
        # Add up all the delta_G values from the first interface
        valid_path = True
        for j in range(i):
            if np.isnan(delta_G[j, j+1]):
                valid_path = False
                break
            cumulative_G[i] += delta_G[j, j+1]
        
        if not valid_path:
            cumulative_G[i] = np.nan
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Free energy differences between adjacent interfaces
    adjacent_dG = np.array([delta_G[i, i+1] for i in range(len(interfaces)-1)])
    bar_positions = np.arange(len(interfaces)-1)
    bar_width = 0.7
    
    # Create bars with color indicating uphill/downhill
    colors = ['red' if dg > 0 else 'green' if dg < 0 else 'gray' for dg in adjacent_dG]
    bars = ax1.bar(bar_positions, adjacent_dG, width=bar_width, color=colors, alpha=0.7)
    
    # Add value annotations on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        if not np.isnan(height):
            y_pos = 0.3 if height < 0 else height + 0.1
            va = 'bottom' if height >= 0 else 'top'
            ax1.text(bar.get_x() + bar.get_width()/2, y_pos, f"{height:.2f}", 
                    ha='center', va=va, fontsize=9)
    
    # Configure the first subplot
    ax1.set_xlabel('Interface Pair Index')
    ax1.set_ylabel('Free Energy Difference ΔG (kT)')
    ax1.set_title('Free Energy Differences Between Adjacent Interfaces', fontsize=12)
    ax1.set_xticks(bar_positions)
    ax1.set_xticklabels([f"{i}→{i+1}" for i in range(len(interfaces)-1)])
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax1.grid(axis='y', alpha=0.3)
    
    legend_elements = [
        Patch(facecolor='red', alpha=0.7, label='Uphill (Barrier)'),
        Patch(facecolor='green', alpha=0.7, label='Downhill (Favorable)')
    ]
    ax1.legend(handles=legend_elements, loc='best')
    
    # Plot 2: Cumulative free energy profile
    ax2.plot(interfaces, cumulative_G, 'o-', linewidth=2, color='blue')
    
    # Add marker points with annotations
    for i, (pos, g) in enumerate(zip(interfaces, cumulative_G)):
        if not np.isnan(g):
            ax2.plot(pos, g, 'o', markersize=8, color='blue')
            ax2.text(pos, g + 0.1, f"{g:.2f}", ha='center', va='bottom', fontsize=9)
    
    # Configure the second subplot
    ax2.set_xlabel('Interface Position λ')
    ax2.set_ylabel('Cumulative Free Energy G(λ) - G(λ₀) (kT)')
    ax2.set_title('Free Energy Profile Along Interface Coordinate', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Add informational text
    info_text = """
    Free energy differences estimated from TIS data:
    • Used to calculate diffusive reference probabilities
    • Red bars indicate barriers (ΔG > 0)
    • Green bars indicate favorable transitions (ΔG < 0)
    
    The cumulative profile shows the estimated free energy
    landscape along the reaction coordinate.
    """
    fig.text(0.02, 0.02, info_text, fontsize=10, wrap=True)
    
    plt.tight_layout(rect=[0, 0.07, 1, 0.95])
    fig.suptitle('Estimated Free Energy Landscape for Diffusive Reference', fontsize=14)
    
    return fig

def pcca_analysis(M, n_clusters):
    """
    Perform PCCA+ analysis on the Markov State Model transition matrix.

    Parameters
    ----------
    M : np.ndarray
        Transition matrix representing the Markov state model.
    n_clusters : int
        Number of metastable states (clusters) to identify.

    Returns
    -------
    pcca : dpt.markov.PCCA
        PCCA+ object containing the clustering results.
    """
    # Ensure the transition matrix is a valid stochastic matrix
    assert np.allclose(M.sum(axis=1), 1), "Rows of the transition matrix must sum to 1."

    # Create a MarkovStateModel object
    msm = dpt.markov.msm.MarkovStateModel(M)

    # Perform PCCA+ analysis
    pcca = dpt.markov.pcca(M, n_clusters)

    # Print the membership matrix
    print("PCCA+ Membership Matrix:")
    print(pcca.memberships)

    # Print the metastable states
    print("Metastable States:")
    for i, state in enumerate(pcca.sets):
        print(f"State {i+1}: {state}")

    return pcca
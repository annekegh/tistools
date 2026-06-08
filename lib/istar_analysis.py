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
    M[2, 1] = P[0, 0]      # Transition from state 2 to state 0
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
        represents the weighted count of paths that start at ensemble i, with a turn from interface j,
        and terminate with a turn at interface k.
    
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
    
    return p, q

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

def compute_weight_matrices(pes, interfaces, n_int=None, tr=False, weights=None, correct_ha=False, norm=False):
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
            ha = pe.shootlinks if correct_ha else None
            w = get_weights_staple(i, pe.flags, pe.generation, pe.lmrs, n_int, ACCFLAGS, REJFLAGS, ha_factors=ha, verbose=True)
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
                    elif j == len(interfaces)-1 and k == len(interfaces)-2:
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
    if tr:
        for i in range(len(pes)):
            if i == 2 and X[i][0, 1] == 0:     # In [1*] all LML paths are classified as 1 -> 0 (for now).
                X[i][1, 0] *= 2     # Time reversal needs to be adjusted to compensate for this
            elif i == 2 and X[i][1, 0] == 0:
                X[i][0, 1] *= 2
            elif i == len(interfaces)-1 and X[i][-1, -2] == 0:
                X[i][-2, -1] *= 2
            X[i] = (X[i] + X[i].T) / 2.0   # Properly symmetrize the matrix
        # this is not right
        if norm:
            X[i] /= np.sum(X[i]) if np.sum(X[i]) > 0 else np.zeros_like(X[i])

    return X

def compute_weight_matrix(pe, pe_id, interfaces, tr=False, weights=None, correct_ha=False):
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
        ha = pe.shootlinks if correct_ha else None
        w = get_weights_staple(pe_id, pe.flags, pe.generation, pe.lmrs, len(interfaces), ACCFLAGS, REJFLAGS, ha_factors=ha, verbose=False)
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


def get_weights_staple(pe_i, flags, gen, ptypes, n_pes, ACCFLAGS, REJFLAGS, ha_factors=None, verbose=True):
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
    print(f"{len(ha_factors) if ha_factors is not None else None} ha_factors for {pe_i}th ensemble")
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
    print(ha_factors)
    assert flags[0] == 'ACC'
    for i, fgp in enumerate(zip(flags, gen, ptypes)):
        flag, gg, ptype = fgp
        if flag in ACCFLAGS:
        # or (pe_i == 1 and ptype in ["RMR", "R**"]):
            # if pe_i == 1 and ptype in ["RMR", "R**"] and flag == "ILL":
            #     print(f"Resetting acc_w for trajectory {i} in ensemble {pe_i} with type {ptype} and generation {gg, flag}")
            #     acc_w = 0
            # Store previous trajectory with accumulated weight
            weights[acc_index] = prev_ha * acc_w
            tot_w += prev_ha * acc_w
            # Info for new trajectory
            acc_index = i
            acc_w = 1
            accepted += 1
            if ha_factors is not None:
                prev_ha = ha_factors[acc_index]
                # if ha_factors[acc_index] == 2:
                    # print(f"High acceptance factor of 2 for trajectory {acc_index} in ensemble {pe_i} with type {ptype} and generation {gg}")
            else:
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

def cprobs_repptis_istar(pes, interfaces, n_int=None):
    if n_int is None:
        n_int = len(interfaces)
    
    X = compute_weight_matrices(pes, interfaces, len(interfaces), tr=True)
    print("\n\n")

    plocrepptis = {}
    plocistar = {}
    for i, pe in enumerate(pes):
        print(f"Ensemble {i} ([{max(0,i-1)}{'+-' if i > 0 else '-'}]):")
        plocrepptis[i] = get_local_probs(pe, tr=False)
        print("\n")

        plocistar[i] = {}
        if i == 0:
            w = get_weights_staple(i, pe.flags, pe.generation, pe.lmrs, len(interfaces), ACCFLAGS, REJFLAGS, ha_factors=pe.weights,verbose = False)
            masks= get_lmr_masks(pe)
            accmask = get_flag_mask(pe, "ACC")
            loadmask = get_generation_mask(pe, "ld")
            plocistar[i]["LML"] = [np.sum(select_with_masks(w, [masks["LML"], accmask, ~loadmask]))/(np.sum(select_with_masks(w, [masks["LML"], accmask, ~loadmask]))+np.sum(select_with_masks(w, [masks["LMR"], accmask, ~loadmask]))), np.sum(select_with_masks(w, [masks["LML"], accmask, ~loadmask]))+np.sum(select_with_masks(w, [masks["LMR"], accmask, ~loadmask]))]
            plocistar[i]["LMR"] = [np.sum(select_with_masks(w, [masks["LMR"], accmask, ~loadmask]))/(np.sum(select_with_masks(w, [masks["LML"], accmask, ~loadmask]))+np.sum(select_with_masks(w, [masks["LMR"], accmask, ~loadmask]))), np.sum(select_with_masks(w, [masks["LML"], accmask, ~loadmask]))+np.sum(select_with_masks(w, [masks["LMR"], accmask, ~loadmask]))]
            plocistar[i]["RMR"] = [np.sum(select_with_masks(w, [masks["RMR"], accmask, ~loadmask]))/(np.sum(select_with_masks(w, [masks["RML"], accmask, ~loadmask]))+np.sum(select_with_masks(w, [masks["RMR"], accmask, ~loadmask]))), np.sum(select_with_masks(w, [masks["RMR"], accmask, ~loadmask]))+np.sum(select_with_masks(w, [masks["RML"], accmask, ~loadmask]))]
            plocistar[i]["RML"] = [1 - plocistar[i]["RMR"][0], np.sum(select_with_masks(w, [masks["RMR"], accmask, ~loadmask]))+np.sum(select_with_masks(w, [masks["RML"], accmask, ~loadmask]))]
            print(f"REPPTIS pLMR approx: {plocistar[i]['LML'][0]}\nREPPTIS pLML approx: {plocistar[i]['LML'][0]} with # weights = {plocistar[i]['LMR'][1]}\n")
            print(f"REPPTIS pRML approx: {plocistar[i]['RML'][0]}\nREPPTIS pRMR approx: {plocistar[i]['RMR'][0]} with # weights = {plocistar[i]['RML'][1]}")
            continue
        # REPPTIS estimates with [i*] weights - slightly different numerically because of different weighting
        plocistar[i]["LML"] = [np.sum(X[i][:max(1,i-1), i-1])/np.sum(X[i][:max(1,i-1), i-1:]), np.sum(X[i][:max(1,i-1), i-1:])]
        plocistar[i]["LMR"] = [1-plocistar[i]["LML"][0], np.sum(X[i][:max(1,i-1), i-1:]), ]
        plocistar[i]["RMR"] = [np.sum(X[i][i:, max(1, i-1)])/np.sum(X[i][i:, :i]), np.sum(X[i][i:, :i])]
        plocistar[i]["RML"] = [1-plocistar[i]["RMR"][0], np.sum(X[i][i:, :i])]

        plocistar[i]["startLMLR"] = np.zeros([3,max(1, i-1)])
        plocistar[i]["startLMLR"] = np.array([np.array([np.sum(X[i][j, i-1])/np.sum(X[i][j, i-1:]), np.sum(X[i][j, i:])/np.sum(X[i][j, i-1:]), np.sum(X[i][j, i-1:])]) for j in range(max(1, i-1))])

        plocistar[i]["startRMRL"] = np.zeros([3, len(interfaces)-i+1])
        plocistar[i]["startRMRL"] = np.array([np.array([np.sum(X[i][j, i-1])/np.sum(X[i][j, :i]), np.sum(X[i][j, :i-1])/np.sum(X[i][j, :i]), np.sum(X[i][j, :i])]) for j in range(i, len(interfaces))])

        plocistar[i]["endLMLR"] = np.zeros([3,len(interfaces)-i+1])
        plocistar[i]["endLMLR"] = np.array([np.array([np.sum(X[i][:max(1,i-1), i-1])/(np.sum(X[i][:max(1,i-1), j])+np.sum(X[i][:max(1,i-1), i-1])), np.sum(X[i][:max(1,i-1), j])/(np.sum(X[i][:max(1,i-1), j])+np.sum(X[i][:max(1,i-1), i-1])), (np.sum(X[i][:max(1,i-1), j])+np.sum(X[i][:max(1,i-1), i-1]))]) for j in range(i, len(interfaces))])
        plocistar[i]["endRMRL"] = np.zeros([3, max(1, i-1)])
        plocistar[i]["endRMRL"] = np.array([np.array([np.sum(X[i][i:, i-1])/(np.sum(X[i][i:, j])+np.sum(X[i][i:, i-1])), np.sum(X[i][i:, j])/(np.sum(X[i][i:, j])+np.sum(X[i][i:, i-1])), (np.sum(X[i][i:, j])+np.sum(X[i][i:, i-1]))]) for j in range(max(0,i-1))])

        plocistar[i]["full"] = np.zeros([len(interfaces), len(interfaces)])
        for j in range(len(interfaces)):
            for k in range(len(interfaces)):
                if j < k:
                    plocistar[i]["full"][j][k] = (X[i][j][k]) / np.sum(X[i][j][j:])
                elif k < j:
                    plocistar[i]["full"][j][k] = (X[i][j][k]) / np.sum(X[i][j][:j])
                else:
                    if j == 0:
                        plocistar[i]["full"][j][k] = X[i][j][k] / np.sum(X[i][j][j:])
                    else:
                        plocistar[i]["full"][j][k] = 0

        # pprint(plocistar)
        print(f"Ensemble {i} ([{i-1}*]):")
        print("    --- LMR/LML ---")
        print(f"REPPTIS pLMR approx: {plocistar[i]['LMR'][0]} (vs. REPPTIS = {plocrepptis[i]['LMR']}) with # weights = {plocistar[i]['LMR'][1]}")
        print(f"REPPTIS pLML fw approx: {plocistar[i]['LML'][0]} (vs. REPPTIS = {plocrepptis[i]['LML']}) with # weights = {plocistar[i]['LML'][1]}")
        print("    START TURNS:")
        for st, p in enumerate(plocistar[i]["startLMLR"]):
            print(f"     -> Conditional pLMR/pLML with start turn at interface {st}: pLML={p[0]} <-> pL{st}MR {p[1]} with sum {np.sum(p[:-1])} \n    with # weights {p[-1]}")
        print("    END TURNS:")
        for st, p in enumerate(plocistar[i]["endLMLR"]):
            print(f"     -> Conditional pLMR/pLML with LMR end turn at interface {st+i}: pLML={p[0]} <-> pLMR{st+i}={p[1]} with sum {np.sum(p[:-1])} \n    with # weights {p[-1]}")
        print("    --- RML/RMR ---")
        print(f"REPPTIS pRML approx: {plocistar[i]['RML'][0]} (vs. REPPTIS = {plocrepptis[i]['RML']}) with # weights = {plocistar[i]['RML'][1]}")
        print(f"REPPTIS pRMR bw approx: {plocistar[i]['RMR'][0]} with # weights = {plocistar[i]['RMR'][1]}")
        print("    START TURNS:")
        for st, p in enumerate(plocistar[i]["startRMRL"]):
            print(f"     -> Conditional pRML/pRMR with start turn at interface {st}: pRMR={p[0]} <-> pR{st}ML {p[1]} with sum {np.sum(p[:-1])} \n    with # weights {p[-1]}")
        print("    END TURNS:")
        for st, p in enumerate(plocistar[i]["endLMLR"]):
            print(f"     -> Conditional pRML/pRMR with RML end turn at interface {st}: pRMR={p[0]} <-> pRML{st}={p[1]} with sum {np.sum(p[:-1])} \nwith # weights {p[-1]}")
        print(f"Full conditional turn probability matrix for ensemble {i}:")
        print(np.array_str(plocistar[i]["full"], precision=5, suppress_small=True))
        print("\n")

    return plocrepptis, plocistar

def display_data(pes, interfaces, n_int=None, weights=None, correct_ha=False):
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
    X_norm = {}
    X_tr = {}
    X_tr_norm = {}
    C = {}
    C_md = {}  # Number of paths where new MD steps are performed (shooting/WF)
    ploc_repptis = {}
    ploc_istar = {}
    if n_int is None:
        n_int = len(pes)
    for i, pe in enumerate(pes):
        print(10*'-')
        star_or_minus = "*" if i > 0 else "-"
        print(f"ENSEMBLE [{i-1 if i>0 else 0}{star_or_minus}] | ID {i}")
        print(10*'-')
        # Get the lmr masks, weights, ACCmask, and loadmask of the paths
        masks[i] = get_lmr_masks(pe)
        if weights is None:
            ha = pe.shootlinks if correct_ha else None
            w = get_weights_staple(i, pe.flags, pe.generation, pe.lmrs, n_int, ACCFLAGS, REJFLAGS, ha_factors=ha,verbose=True)
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
        X_tr[i] = np.zeros([len(interfaces), len(interfaces)])
        X_tr_norm[i] = np.zeros([len(interfaces), len(interfaces)])
        C[i] = np.zeros([len(interfaces), len(interfaces)])
        C_md[i] = np.zeros([len(interfaces), len(interfaces)])
        X_val = compute_weight_matrix(pe, i, interfaces, tr=False, weights=weights)
        X_val_tr = compute_weight_matrix(pe, i, interfaces, tr=True, weights=weights)

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
                            dir_mask = np.full_like(pe.dirs, True)  # For now no distinction yet for [1*] paths: classify all paths as 1 -> 0. Later: dir_mask = np.logical_or(pe.dirs == 1, masks[i]["LML"])
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
                    elif j == len(interfaces)-1 and k == len(interfaces)-2:
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
        
        print("\n2b. Normalized weighted data")
        print(f"X_norm[{i}] = ")
        X_norm[i] = X[i] / np.sum(X[i]) if np.sum(X[i]) != 0 else np.zeros_like(X[i])
        print(np.array2string(X_norm[i], precision=4, suppress_small=True))

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
        X_copy = X[i].copy()
        if i == 2 and X[i][0, 1] == 0:     # In [1*] all LML paths are classified as 1 -> 0 (for now).
            X_copy[1, 0] *= 2     # Time reversal needs to be adjusted to compensate for this
        elif i == 2 and X[i][1, 0] == 0:
            X_copy[0, 1] *= 2
        elif i == len(interfaces)-1 and X[i][-1, -2] == 0:
            X_copy[-2, -1] *= 2
        X_tr[i] = (X_copy+X_copy.T)/2.0
        print(np.array2string(X_tr[i], precision=4, suppress_small=True))
        
        # Warnings for significant differences in time-reversal symmetry
        if len(idx_tr) > 0:
            print("\n[WARNING] Time-reversal symmetry violations:")
            for idx in idx_tr:
                print(f"  Path {idx[0]} ↔ {idx[1]}: relative diff={tr_diff[idx[0],idx[1]]:.4f}, "
                    f"forward weight={X[i][idx[0]][idx[1]]:.2f}, "
                    f"backward weight={X[i][idx[1]][idx[0]]:.2f}")
        
        # Warning for interfaces (columns) with unusually low sampling
        total_sum = np.sum(X_tr[i])
        col_sums = np.sum(X_tr[i], axis=0)
        n_cols = X_tr[i].shape[1]
        threshold = total_sum / (5 * n_cols)  # Threshold for warning: 20% of average
        
        low_cols = np.where(col_sums < threshold)[0]
        if len(low_cols) > 0:
            print("\n[WARNING] Interfaces with insufficient sampling:")
            for col in low_cols:
                print(f"  Interface {col}: weight sum={col_sums[col]:.2f}, "
                    f"only {(col_sums[col]/total_sum)*100:.1f}% of total weight "
                    f"(threshold: {threshold:.2f})")

        print("3b. Normalized data with time reversal")
        print(f"X_tr_norm[{i}] = ")
        X_tr_norm[i] = X_tr[i] / np.sum(X_tr[i]) if np.sum(X_tr[i]) != 0 else np.zeros_like(X_tr[i])
        print(np.array2string(X_tr_norm[i], precision=4, suppress_small=True))

        # 5. Unweighted data with time reversal symmetry
        print("\n3c. Unweighted data with time reversal")
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
    
    W_norm = np.zeros_like(W)
    for j in range(len(interfaces)):
        for k in range(len(interfaces)):
            W_norm[j][k] = np.sum(X_norm[i][j][k] for i in range(n_int))
    
    W_tr_norm = np.zeros_like(W)
    for j in range(len(interfaces)):
        for k in range(len(interfaces)):
            W_tr_norm[j][k] = np.sum(X_tr_norm[i][j][k] for i in range(n_int))

    # Combined weights without time-reversal symmetry
    print("\n4. Weights of all ensembles combined (sum), no TR")
    print(np.array2string(W, precision=4, suppress_small=True))
    
    # Combined weights with time-reversal symmetry
    print("\n5. Weights of all ensembles combined (sum), with TR")
    W_tr = (W+W.T)/2.
    if W[1,0] == 0:
        W_tr[0,1] *= 1 
        W_tr[1,0] *= W_tr[0,1]
    print(np.array2string(W_tr, precision=4, suppress_small=True))

    return C, X_tr, X_norm, X_tr_norm, X, W_tr, W_norm, W_tr_norm, W

def ploc_repptis_from_staples(pes, interfaces, n_int=None, staple_weights=None):
    """
    Calculate and compare REPPTIS local crossing probabilities using staple weights.
    
    This function extracts local crossing probabilities from path ensembles using both
    traditional REPPTIS calculations and alternative calculations based on staple-weighted
    path counts. It provides a comprehensive comparison between these approaches.
    
    Parameters
    ----------
    pes : list
        List of PathEnsemble objects to analyze
    interfaces : list
        List of interface positions
    n_int : int, optional
        Number of interfaces to consider. If None, uses the length of pes.
    staple_weights : dict, optional
        Pre-computed staple weights. If None, weights are calculated within the function.
        
    Returns
    -------
    tuple
        (ploc_repptis, ploc_istar) - Dictionaries containing local probability estimates
        from standard REPPTIS and iSTAR (staple-weighted) approaches.
    """
    masks = {}
    ploc_repptis = {}
    ploc_istar = {}
    
    # Compute weight matrices for all path ensembles
    X = compute_weight_matrices(pes, interfaces, n_int=n_int, weights=staple_weights, tr=False)
    
    if n_int is None:
        n_int = len(pes)

    # Process each path ensemble
    for i, pe in enumerate(pes):
        print("\n" + "="*50)
        print(f"ENSEMBLE [{i-1 if i>0 else 0}{'*' if i>0 else '-'}] | ID {i}")
        print("="*50)
        
        # Get masks, weights, and other path statistics
        masks[i] = get_lmr_masks(pe)
        if staple_weights is None:
            w = get_weights_staple(i, pe.flags, pe.generation, pe.lmrs, n_int, ACCFLAGS, REJFLAGS, ha_factors=pe.weights,verbose=False)
        else:
            w = staple_weights[i]
            
        accmask = get_flag_mask(pe, "ACC")
        loadmask = get_generation_mask(pe, "ld")
        md_mask = np.logical_or(pe.generation == "sh", pe.generation == "wf")
        
        if i == 1: 
            md_mask = np.logical_or(md_mask, pe.generation == "s-")
        elif i == 0: 
            md_mask = np.logical_or(md_mask, pe.generation == "s+")
            
        # Log ensemble statistics
        msg = f"Ensemble {pe.name[-3:]} summary:\n"
        msg += f"• Total paths: {len(w)}\n"
        msg += f"• Total ensemble weight: {np.sum(w):.2f}\n"
        msg += f"• Accepted paths: {np.sum(accmask)}\n"
        msg += f"• Loaded paths: {np.sum(loadmask)}"
        logger.debug(msg)

        # Calculate standard REPPTIS local probabilities
        ploc_repptis[i] = get_local_probs(pe, tr=False)
        ploc_istar[i] = {}
        
        # Special handling for first ensemble
        if i == 0:
            w = get_weights_staple(i, pe.flags, pe.generation, pe.lmrs, len(interfaces), ACCFLAGS, REJFLAGS, ha_factors=pe.weights,verbose=False)
            masks = get_lmr_masks(pe)
            accmask = get_flag_mask(pe, "ACC")
            loadmask = get_generation_mask(pe, "ld")
            
            # Calculate LML/LMR probabilities
            lml_weight = np.sum(select_with_masks(w, [masks["LML"], accmask, ~loadmask]))
            lmr_weight = np.sum(select_with_masks(w, [masks["LMR"], accmask, ~loadmask]))
            total_lm_weight = lml_weight + lmr_weight
            
            ploc_istar[i]["LML"] = [lml_weight/total_lm_weight if total_lm_weight > 0 else 0, total_lm_weight]
            ploc_istar[i]["LMR"] = [lmr_weight/total_lm_weight if total_lm_weight > 0 else 0, total_lm_weight]
            
            # Calculate RML/RMR probabilities
            rmr_weight = np.sum(select_with_masks(w, [masks["RMR"], accmask, ~loadmask]))
            rml_weight = np.sum(select_with_masks(w, [masks["RML"], accmask, ~loadmask]))
            total_rm_weight = rmr_weight + rml_weight
            
            ploc_istar[i]["RMR"] = [rmr_weight/total_rm_weight if total_rm_weight > 0 else 0, total_rm_weight]
            ploc_istar[i]["RML"] = [1 - ploc_istar[i]["RMR"][0], total_rm_weight]
            
            print("\n📊 ENSEMBLE STATISTICS:")
            print(f"  LMR/LML Statistics:")
            print(f"  • pLMR from iSTAR: {ploc_istar[i]['LMR'][0]:.4f} (weight: {ploc_istar[i]['LMR'][1]:.1f})")
            print(f"  • pLML from iSTAR: {ploc_istar[i]['LML'][0]:.4f} (weight: {ploc_istar[i]['LML'][1]:.1f})")
            print(f"\n  RML/RMR Statistics:")
            print(f"  • pRML from iSTAR: {ploc_istar[i]['RML'][0]:.4f} (weight: {ploc_istar[i]['RML'][1]:.1f})")
            print(f"  • pRMR from iSTAR: {ploc_istar[i]['RMR'][0]:.4f} (weight: {ploc_istar[i]['RMR'][1]:.1f})")
            continue
            
        # Calculate iSTAR probabilities for other ensembles
        lml_sum = np.sum(X[i][:max(1,i-1), i-1])
        lm_total = np.sum(X[i][:max(1,i-1), i-1:])
        ploc_istar[i]["LML"] = [lml_sum/lm_total if lm_total > 0 else 0, lm_total]
        ploc_istar[i]["LMR"] = [1-ploc_istar[i]["LML"][0], lm_total]
        
        rmr_sum = np.sum(X[i][i:, i-1])
        rm_total = np.sum(X[i][i:, :i])
        ploc_istar[i]["RMR"] = [rmr_sum/rm_total if rm_total > 0 else 0, rm_total]
        ploc_istar[i]["RML"] = [1-ploc_istar[i]["RMR"][0], rm_total]
        
        # Print individual ensemble results with comparison
        print("\n📊 INDIVIDUAL ENSEMBLE ANALYSIS:")
        print("  LMR/LML Transition Probabilities:")
        print(f"  • pLMR: {ploc_istar[i]['LMR'][0]:.4f} (iSTAR) vs {ploc_repptis[i]['LMR']:.4f} (REPPTIS)")
        print(f"    - Total weight: {ploc_istar[i]['LMR'][1]:.1f}")
        print(f"  • pLML: {ploc_istar[i]['LML'][0]:.4f} (iSTAR) vs {ploc_repptis[i]['LML']:.4f} (REPPTIS)")
        print(f"    - Total weight: {ploc_istar[i]['LML'][1]:.1f}")
        
        print("\n  RML/RMR Transition Probabilities:")
        print(f"  • pRML: {ploc_istar[i]['RML'][0]:.4f} (iSTAR) vs {ploc_repptis[i]['RML']:.4f} (REPPTIS)")
        print(f"    - Total weight: {ploc_istar[i]['RML'][1]:.1f}")
        print(f"  • pRMR: {ploc_istar[i]['RMR'][0]:.4f} (iSTAR) vs {ploc_repptis[i]['RMR']:.4f} (REPPTIS)")
        print(f"    - Total weight: {ploc_istar[i]['RMR'][1]:.1f}")
        
        # Combined results from all ensembles
        print("\n🔄 ALL ENSEMBLES COMBINED:")
        print(f"  REPPTIS local probabilities from [{i-1}*]-weighted paths:")
        
        # Calculate LML/LMR probabilities from all ensembles
        lml_counts = np.zeros(2)
        for ens in range(1, i+1):
            lml_counts[0] += np.sum(X[ens][:max(1,i-1), i-1])
            lml_counts[1] += np.sum(X[ens][:max(1,i-1), i-1:])
            
        ploc_istar[i]["LMLtot"] = [lml_counts[0]/lml_counts[1] if lml_counts[1] > 0 else np.nan, lml_counts[1]]
        ploc_istar[i]["LMRtot"] = [1 - ploc_istar[i]["LMLtot"][0] if not np.isnan(ploc_istar[i]["LMLtot"][0]) else np.nan, lml_counts[1]]
        
        # Calculate RML/RMR probabilities from all ensembles
        rmr_counts = np.zeros(2)
        for ens in range(i, n_int):
            rmr_counts[0] += np.sum(X[ens][i:, i-1])
            rmr_counts[1] += np.sum(X[ens][i:, :i])
            
        ploc_istar[i]["RMRtot"] = [rmr_counts[0]/rmr_counts[1] if rmr_counts[1] > 0 else np.nan, rmr_counts[1]]
        ploc_istar[i]["RMLtot"] = [1 - ploc_istar[i]["RMRtot"][0] if not np.isnan(ploc_istar[i]["RMRtot"][0]) else np.nan, rmr_counts[1]]
        
        # Print aggregated results
        print("\n  LMR/LML Transition Probabilities (Combined):")
        print(f"  • pLMR: {ploc_istar[i]['LMRtot'][0]:.4f} (iSTAR) vs {ploc_repptis[i]['LMR']:.4f} (REPPTIS)")
        print(f"    - Total weight: {ploc_istar[i]['LMRtot'][1]:.1f}")
        print(f"  • pLML: {ploc_istar[i]['LMLtot'][0]:.4f} (iSTAR) vs {ploc_repptis[i]['LML']:.4f} (REPPTIS)")
        print(f"    - Total weight: {ploc_istar[i]['LMLtot'][1]:.1f}")
        
        print("\n  RML/RMR Transition Probabilities (Combined):")
        print(f"  • pRML: {ploc_istar[i]['RMLtot'][0]:.4f} (iSTAR) vs {ploc_repptis[i]['RML']:.4f} (REPPTIS)")
        print(f"    - Total weight: {ploc_istar[i]['RMLtot'][1]:.1f}")
        print(f"  • pRMR: {ploc_istar[i]['RMRtot'][0]:.4f} (iSTAR) vs {ploc_repptis[i]['RMR']:.4f} (REPPTIS)")
        print(f"    - Total weight: {ploc_istar[i]['RMRtot'][1]:.1f}")

    return ploc_repptis, ploc_istar


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

def ploc_memory(pathensembles, interfaces, trr=False):
    """
    Calculate global crossing probabilities using multiple methods and compare their results.
    
    This function implements and compares three different approaches for calculating
    global crossing probabilities:
    1. Milestoning: Based on local transitions between adjacent interfaces
    2. REPPTIS: Using the replica exchange TIS formalism
    3. APPTIS: Using the iSTAR path ensemble network approach (with and without HA correction)
    
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
        - "apptis": APPTIS probabilities using iSTAR without HA correction
        - "repptis": REPPTIS probabilities
        
    Notes
    -----
    This function also generates a logarithmic plot comparing the four methods, showing:
    - How estimates of crossing probabilities vary between methods
    - Whether the system exhibits significant memory effects (differences between methods)
    - Which method might be most appropriate for the system under study
    
    Significant differences between methods usually indicate memory effects or
    sampling issues that require careful interpretation.
    """
    plocs = {}
    plocs["mlst"] = [1.,]
    plocs["apptis"] = [1.,]
    plocs["apptis_ha"] = [1.,]
    plocs["apptis_ha, norm"] = [1.,]
    repptisploc = []

    for i, pe in enumerate(pathensembles):
        # REPPTIS p_loc
        repptisploc.append(get_local_probs(pe, tr=trr))

        # Milestoning p_loc
        if i == 1:
            plocs["mlst"].append(repptisploc[i]["LMR"]*plocs["mlst"][-1])
        elif i > 1:
            pmin = [repptisploc[r]["2L"] for r in range(1,len(repptisploc))]
            pplus = [repptisploc[r]["2R"] for r in range(1,len(repptisploc))]
            Milst = construct_M_milestoning(pmin, pplus, len(interfaces[:i+1]))
            z1, z2, y1, y2 = global_pcross_msm(Milst)
            plocs["mlst"].append(y1[0][0])

        # APPTIS p_loc (without HA correction)
        if i < len(pathensembles)-1:
            wi = compute_weight_matrices(pathensembles[:i+2], interfaces[:i+2], len(interfaces), tr=trr, correct_ha=False, norm=False)
            pi, _ = get_transition_probs_weights(wi)
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
    # ax.errorbar([i for i in range(len(interfaces))], plocs["apptis_ha"], fmt="-s", c = "cyan", ecolor="r", capsize=6, label="APPTIS (HA corrected)")
    ax.errorbar([i for i in range(len(interfaces))], plocs["repptis"], fmt="-o", c = "orange", ecolor="r", capsize=6., label="REPPTIS")
    ax.errorbar([i for i in range(len(interfaces))], plocs["mlst"], fmt="-o", c = "r", ecolor="r", capsize=6., label="Milestoning")
    ax.set_xlabel(r"Interface index")
    ax.set_ylabel(r"$P_A(\lambda_i|\lambda_A)$")
    ax.set_xticks(np.arange(len(interfaces)))
    fig.tight_layout()
    fig.legend()
    fig.show()

    return plocs

def generate_state_labels(n_interfaces):
    """
    Generate descriptive state labels for interface states.
    
    Parameters:
    -----------
    n_interfaces : int
        Number of interfaces in the system
    
    Returns:
    --------
    list
        List of descriptive labels for each state
    """
    state_labels = []
    middle = n_interfaces 
    
    for i in range(2*n_interfaces):
        if i == 0:
            state_labels.append("[0$^-$]")
        elif i == 1:
            state_labels.append("[0←]")
        elif i == 2:
            state_labels.append("[0→]")
        elif i <= middle:
            state_labels.append(f"[{i-2}$\\subset$]")
        elif middle < i < 2*n_interfaces - 1:
            state_labels.append(f"[{i-middle}$\\supset$]")
        else:
            state_labels.append(f"[{i-middle}]")
    
    return state_labels

def calculate_memory_effect_index(q_probs, q_weights, q_errors=None, min_samples=5):
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

    if q_errors is not None:
        if isinstance(q_errors, str):
            # q_errors is a filepath string, try to load it.
            try:
                # Assuming q_errors is a path to a text file loadable by np.loadtxt
                loaded_q_errors = read_block_errors(q_errors, q_probs.shape)
                if loaded_q_errors.shape != q_probs.shape:
                    raise RuntimeWarning(
                        f"Shape of q_errors loaded from file '{q_errors}' ({loaded_q_errors.shape}) "
                        f"does not match q_probs shape ({q_probs.shape})."
                    )
                # Update the local q_errors variable to the loaded numpy array.
                # Note: The rest of this function does not currently use q_errors.
                q_errors = loaded_q_errors
            except FileNotFoundError:
                q_errors = None  # Reset to None if file not found.
                raise RuntimeWarning(f"q_errors file not found: {q_errors}")
            except Exception as e:
                q_errors = None 
                raise RuntimeWarning(f"Error loading q_errors from file '{q_errors}': {e}")
        elif not (isinstance(q_errors, np.ndarray) and q_errors.shape == q_probs.shape):
            # q_errors is provided, is not a string, but is not a valid ndarray of the correct shape.
            q_errors = None  # Reset to None to avoid further issues.
            raise RuntimeWarning(
                f"If provided, q_errors must be a NumPy array with shape {q_probs.shape} "
                f"or a path to a loadable text file. "
                f"Got type {type(q_errors)} with shape {getattr(q_errors, 'shape', 'N/A')}."
            )
    # If q_errors was None, it remains None.
    # If it was a valid np.ndarray, it remains as is.
    # If it was a string path, it is now a loaded np.ndarray (or an error was raised).
    # The current function calculate_memory_effect_index does not use q_errors beyond this point.

    
    # Initialize result arrays
    forward_variation = np.zeros(n_interfaces)
    forward_variation.fill(np.nan)
    forward_variation_error = np.zeros(n_interfaces)
    forward_variation_error.fill(np.nan)              
    forward_sample_sizes = np.zeros(n_interfaces, dtype=int)
    
    backward_variation = np.zeros(n_interfaces)
    backward_variation.fill(np.nan)
    backward_variation_error = np.zeros(n_interfaces)  
    backward_variation_error.fill(np.nan)              
    backward_sample_sizes = np.zeros(n_interfaces, dtype=int)
    
    # Calculate forward memory effect index (for targets k > 0)
    for k in range(1, n_interfaces):
        q_values = []
        weights = []
        q_errors_k = []

        for i in range(max(1, k-1)):  # Skip adjacent interface (i=k-1), so range is 0 to k-2
            if not np.isnan(q_probs[i, k]) and q_weights[i, k] >= min_samples:
                q_values.append(q_probs[i, k])
                weights.append(q_weights[i, k])
                if q_errors is not None and not np.isnan(q_errors[i, k]):
                    q_errors_k.append(q_errors[i, k])
                else:
                    # Fallback to binomial error if block error not available
                    binomial_error = np.sqrt(q_probs[i, k] * (1 - q_probs[i, k]) / q_weights[i, k])
                    q_errors_k.append(binomial_error)
        
        if len(q_values) >= 2:
            q_values = np.array(q_values)
            weights = np.array(weights)
            q_errors_arr = np.array(q_errors_k)
            
            total_samples = np.sum(weights)
            forward_sample_sizes[k] = total_samples
            
            # Calculate standard deviation
            std_dev = np.std(q_values)
            forward_variation[k] = std_dev * 100  # Convert to percentage
            
            # Calculate standard error of the standard deviation using error propagation
            n = len(q_values)
            mean_q = np.mean(q_values)
            
            # For standard deviation σ = sqrt(Σ(q_i - μ)²/(n-1)), 
            # the error propagation gives: σ_σ ≈ sqrt(Σ((q_i - μ)/σ * σ_q_i)²) / sqrt(2(n-1))
            if std_dev > 0 and not np.any(np.isnan(q_errors_arr)):
                # Partial derivatives of std with respect to each q_i
                # ∂σ/∂q_i = (q_i - μ) / ((n-1) * σ)
                partial_derivs = (q_values - mean_q) / ((n - 1) * std_dev)
                
                # Error propagation: σ_σ² = Σ(∂σ/∂q_i)² * σ_q_i²
                std_error_squared = np.sum((partial_derivs * q_errors_arr) ** 2)
                std_error = np.sqrt(std_error_squared)
                
                forward_variation_error[k] = std_error * 100  # Convert to percentage
            else:
                forward_variation_error[k] = np.nan
        else:
            forward_variation[k] = np.nan
            forward_variation_error[k] = np.nan
            forward_sample_sizes[k] = 0
    
    # Calculate backward memory effect index (for targets k < n_interfaces-1)
    for k in range(n_interfaces - 1):
        q_values = []
        weights = []
        q_errors_k = []
        
        for i in range(k+2, n_interfaces):  # Skip adjacent interface (i=k+1)
            if not np.isnan(q_probs[i, k]) and q_weights[i, k] >= min_samples:
                q_values.append(q_probs[i, k])
                weights.append(q_weights[i, k])
                if q_errors is not None and not np.isnan(q_errors[i, k]):
                    q_errors_k.append(q_errors[i, k])
                else:
                    # Fallback to binomial error if block error not available
                    binomial_error = np.sqrt(q_probs[i, k] * (1 - q_probs[i, k]) / q_weights[i, k])
                    q_errors_k.append(binomial_error)
        
        if len(q_values) >= 2:
            q_values = np.array(q_values)
            weights = np.array(weights)
            q_errors_arr = np.array(q_errors_k)

            total_samples = np.sum(weights)
            backward_sample_sizes[k] = total_samples
            
            # Calculate standard deviation
            std_dev = np.std(q_values)
            backward_variation[k] = std_dev * 100  # Convert to percentage
            
            # Calculate standard error of the standard deviation
            n = len(q_values)
            mean_q = np.mean(q_values)
            
            if std_dev > 0 and not np.any(np.isnan(q_errors_arr)):
                partial_derivs = (q_values - mean_q) / ((n - 1) * std_dev)
                std_error_squared = np.sum((partial_derivs * q_errors_arr) ** 2)
                std_error = np.sqrt(std_error_squared)
                
                backward_variation_error[k] = std_error * 100  # Convert to percentage
            else:
                backward_variation_error[k] = np.nan
        else:
            backward_variation[k] = np.nan
            backward_variation_error[k] = np.nan
            backward_sample_sizes[k] = 0
            
    memory_index = {
        'forward_variation': forward_variation,
        'forward_variation_error': forward_variation_error, 
        'backward_variation': backward_variation,
        'backward_variation_error': backward_variation_error, 
        'forward_sample_sizes': forward_sample_sizes,
        'backward_sample_sizes': backward_sample_sizes
    }
    
    return memory_index

def calculate_diffusive_reference(interfaces, q_matrix, q_weights=None, min_samples=5, account_for_distances=True):
    """
    Calculate diffusive reference probabilities for non-equidistant interfaces,
    using the same approach as analyze_momentum_vs_free_energy.
    
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
    account_for_distances : bool, optional
        Whether to account for physical distances between interfaces (default: True)
        
    Returns:
    --------
    diffusive_q : numpy.ndarray
        A matrix of diffusive reference probabilities for each i,k pair
    """
    n_interfaces = len(interfaces)
    diffusive_q = np.zeros((n_interfaces, n_interfaces))
    diffusive_q.fill(np.nan)
    
    # Step 1: Estimate free energy differences between interfaces
    delta_G = estimate_free_energy_differences(interfaces, q_matrix, q_weights, min_samples, account_for_distances)
    
    # Check if interfaces have physical values we can use for geometric corrections
    has_physical_distances = isinstance(interfaces[0], (int, float)) and len(interfaces) > 1 and account_for_distances
    
    # For diagonal elements (self-transitions), probability is 0
    for i in range(n_interfaces):
        diffusive_q[i, i] = 0.0
    
    # Create reference probabilities for forward transitions (i<k)
    for k in range(1, n_interfaces):
        for i in range(k):
            if i == k-1:
                # Adjacent interfaces always have probability 1.0
                diffusive_q[i, k] = 1.0
            else:
                # Non-adjacent interfaces use the diffusive reference with geometric correction
                ref_prob = 0.5  # Default value
                
                # Use Boltzmann factor for the free energy difference between k-1 and k
                if not np.isnan(delta_G[k-1, k]):
                    # Apply geometric correction if we have physical interface positions
                    if has_physical_distances and k > 1:
                        # Calculate distance-based geometric expected probability
                        dist_km1_to_k = interfaces[k] - interfaces[k-1]
                        dist_km2_to_km1 = interfaces[k-1] - interfaces[k-2] if k > 1 else dist_km1_to_k
                        geo_q = dist_km2_to_km1 / (dist_km1_to_k + dist_km2_to_km1) if (dist_km1_to_k + dist_km2_to_km1) > 0 else 0.5
                        
                        # Combine geometric factor with free energy difference
                        # For forward transitions: P = 1/(1 + exp($\Delta$G))
                        ref_prob = 1.0 / (1.0 + np.exp(delta_G[k-1, k]) * (1-geo_q)/geo_q)
                    else:
                        # Without geometry, just use free energy
                        ref_prob = 1.0 / (1.0 + np.exp(delta_G[k-1, k]))
                
                diffusive_q[i, k] = ref_prob
    
    # Create reference probabilities for backward transitions (i>k)
    for k in range(n_interfaces-1):
        for i in range(k+1, n_interfaces):
            if i == k+1:
                # Adjacent interfaces always have probability 1.0
                diffusive_q[i, k] = 1.0
            else:
                # Non-adjacent interfaces use the diffusive reference with geometric correction
                ref_prob = 0.5  # Default value
                
                # Use Boltzmann factor for the free energy difference between k and k+1
                if not np.isnan(delta_G[k, k+1]):
                    # Apply geometric correction if we have physical interface positions
                    if has_physical_distances and k+2 < n_interfaces:
                        # Calculate distance-based geometric expected probability
                        dist_k_to_kp1 = interfaces[k+1] - interfaces[k]
                        dist_kp1_to_kp2 = interfaces[k+2] - interfaces[k+1] if k+2 < n_interfaces else dist_k_to_kp1
                        geo_q = dist_kp1_to_kp2 / (dist_k_to_kp1 + dist_kp1_to_kp2) if (dist_k_to_kp1 + dist_kp1_to_kp2) > 0 else 0.5
                        
                        # Combine geometric factor with free energy difference
                        # For backward transitions: P = 1/(1 + exp(-$\Delta$G))
                        ref_prob = 1.0 / (1.0 + np.exp(-delta_G[k, k+1]) * (1-geo_q)/geo_q)
                    else:
                        # Without geometry, just use free energy
                        ref_prob = 1.0 / (1.0 + np.exp(-delta_G[k, k+1]))
                
                diffusive_q[i, k] = ref_prob
    
    # Handle edge cases and verify
    for i in range(n_interfaces):
        for k in range(n_interfaces):
            # Ensure probabilities are in valid range
            if not np.isnan(diffusive_q[i, k]):
                # Constrain to [0,1]
                diffusive_q[i, k] = max(0.0, min(1.0, diffusive_q[i, k]))
    
    # Print some statistics about the calculated reference probabilities
    print("\nDiffusive reference probability summary:")
    print(f"Mean forward reference probability: {np.nanmean([diffusive_q[i,j] for i in range(n_interfaces) for j in range(i+1,n_interfaces)]):.4f}")
    print(f"Mean backward reference probability: {np.nanmean([diffusive_q[i,j] for i in range(n_interfaces) for j in range(i)]):.4f}")
    print(f"Accounted for physical distances: {has_physical_distances}")
    
    return diffusive_q

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

def analyze_momentum_vs_free_energy(interfaces, q_matrix, q_weights=None, min_samples=5, momentum_threshold=0.2):
    """
    Distinguish between free energy effects and momentum effects in turn-based path networks
    by comparing observed transition probabilities with diffusive predictions based on estimated free energies.
    
    Parameters
    ----------
    interfaces : list or array
        The positions of the interfaces along the reaction coordinate
    q_matrix : numpy.ndarray
        Matrix of conditional crossing probabilities
    q_weights : numpy.ndarray, optional
        Matrix of sample counts for each q_matrix value
    min_samples : int, optional
        Minimum number of samples required to consider a q value valid
    momentum_threshold : float, optional
        Threshold for determining significant momentum effects (relative deviation)
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'free_energy_differences': Free energy differences between interfaces
        - 'diffusive_probabilities': Predicted probabilities based on free energy model
        - 'momentum_effects': Deviations between observed and diffusive probabilities
        - 'momentum_significance': Whether the deviations are significant
        - 'classification': Classification of each interface as free-energy-dominated or momentum-dominated
        - 'overall_classification': Overall assessment of the system
    """
    n_interfaces = len(interfaces)
    
    # Step 1: Estimate free energy differences between interfaces
    delta_G = estimate_free_energy_differences(interfaces, q_matrix, q_weights, min_samples, account_for_distances=True)
    
    # Step 2: Calculate diffusive reference probabilities based on free energy
    diffusive_q = np.zeros_like(q_matrix)
    diffusive_q.fill(np.nan)
    
    # For diagonal elements (self-transitions), probability is 0
    for i in range(n_interfaces):
        diffusive_q[i, i] = 0.0
    
    # Check if interfaces have physical values we can use for geometric corrections
    has_physical_distances = isinstance(interfaces[0], (int, float)) and len(interfaces) > 1
    
    # Create reference probabilities for forward transitions (i<k)
    for k in range(1, n_interfaces):
        for i in range(k):
            if i == k-1:
                # Adjacent interfaces always have probability 1.0
                diffusive_q[i, k] = 1.0
            else:
                # Non-adjacent interfaces use the diffusive reference with geometric correction
                ref_prob = 0.5  # Default value
                
                # Use Boltzmann factor for the free energy difference between k-1 and k
                if not np.isnan(delta_G[k-1, k]):
                    # Apply geometric correction if we have physical interface positions
                    if has_physical_distances and k > 1:
                        # Calculate distance-based geometric expected probability
                        dist_km1_to_k = interfaces[k] - interfaces[k-1]
                        dist_km2_to_km1 = interfaces[k-1] - interfaces[k-2] if k > 1 else dist_km1_to_k
                        geo_q = dist_km2_to_km1 / (dist_km1_to_k + dist_km2_to_km1) if (dist_km1_to_k + dist_km2_to_km1) > 0 else 0.5
                        
                        # Combine geometric factor with free energy difference
                        # For forward transitions: P = 1/(1 + exp($\Delta$G))
                        ref_prob = 1.0 / (1.0 + np.exp(delta_G[k-1, k]) * (1-geo_q)/geo_q)
                    else:
                        # Without geometry, just use free energy
                        ref_prob = 1.0 / (1.0 + np.exp(delta_G[k-1, k]))
                
                diffusive_q[i, k] = ref_prob
    
    # Create reference probabilities for backward transitions (i>k)
    for k in range(n_interfaces-1):
        for i in range(k+1, n_interfaces):
            if i == k+1:
                # Adjacent interfaces always have probability 1.0
                diffusive_q[i, k] = 1.0
            else:
                # Non-adjacent interfaces use the diffusive reference with geometric correction
                ref_prob = 0.5  # Default value
                
                # Use Boltzmann factor for the free energy difference between k and k+1
                if not np.isnan(delta_G[k, k+1]):
                    # Apply geometric correction if we have physical interface positions
                    if has_physical_distances and k+2 < n_interfaces:
                        # Calculate distance-based geometric expected probability
                        dist_k_to_kp1 = interfaces[k+1] - interfaces[k]
                        dist_kp1_to_kp2 = interfaces[k+2] - interfaces[k+1] if k+2 < n_interfaces else dist_k_to_kp1
                        geo_q = dist_kp1_to_kp2 / (dist_k_to_kp1 + dist_kp1_to_kp2) if (dist_k_to_kp1 + dist_kp1_to_kp2) > 0 else 0.5
                        
                        # Combine geometric factor with free energy difference
                        # For backward transitions: P = 1/(1 + exp(-$\Delta$G))
                        ref_prob = 1.0 / (1.0 + np.exp(delta_G[k+1, k]) * (1-geo_q)/geo_q)
                    else:
                        # Without geometry, just use free energy
                        ref_prob = 1.0 / (1.0 + np.exp(delta_G[k+1, k]))
                
                diffusive_q[i, k] = ref_prob
    
    # Step 3: Calculate momentum effects as deviations from diffusive predictions
    momentum_effects = np.zeros_like(q_matrix)
    momentum_effects.fill(np.nan)
    
    # Calculate relative deviations
    for i in range(n_interfaces):
        for k in range(n_interfaces):
            if i != k and not np.isnan(q_matrix[i, k]) and not np.isnan(diffusive_q[i, k]):
                # Only consider values with enough samples
                if q_weights is None or q_weights[i, k] >= min_samples:
                    # Calculate relative deviation as a percentage
                    momentum_effects[i, k] = (q_matrix[i, k] - diffusive_q[i, k])
    
    # Step 4: Determine which deviations are significant
    momentum_significance = np.zeros_like(q_matrix, dtype=bool)
    
    for i in range(n_interfaces):
        for k in range(n_interfaces):
            if not np.isnan(momentum_effects[i, k]):
                momentum_significance[i, k] = abs(momentum_effects[i, k]) > momentum_threshold
    
    # Step 5: Classify each interface pair
    pair_classification = []
    
    for i in range(n_interfaces-1):
        # Look at transitions between interfaces i and i+1
        forward_effect = momentum_effects[i-1, i+1] if i > 0 and not np.isnan(momentum_effects[i-1, i+1]) else 0
        backward_effect = momentum_effects[i+2, i] if i+2 < n_interfaces and not np.isnan(momentum_effects[i+2, i]) else 0
        
        # Check if effects are significant
        forward_significant = momentum_significance[i-1, i+1] if i > 0 else False
        backward_significant = momentum_significance[i+2, i] if i+2 < n_interfaces else False
        
        # Assess the average strength of momentum effects for this interface pair
        avg_effect = (abs(forward_effect) + abs(backward_effect)) / 2 if (i > 0 and i+2 < n_interfaces) else \
                     (abs(forward_effect) if i > 0 else abs(backward_effect))
        
        # Check for directional bias in momentum effects
        if forward_significant or backward_significant:
            if abs(forward_effect + backward_effect) < 0.2 * (abs(forward_effect) + abs(backward_effect)):
                # Momentum effects in both directions that nearly cancel out
                pair_classification.append("symmetric_momentum")
            elif avg_effect > momentum_threshold * 2:
                # Strong momentum effects
                pair_classification.append("strong_momentum")
            else:
                # Moderate momentum effects
                pair_classification.append("momentum_dominated")
        else:
            # No significant momentum effects
            pair_classification.append("free_energy_dominated")
    
    # Step 6: Calculate overall system-wide metrics
    avg_abs_momentum = np.nanmean(np.abs(momentum_effects))
    sum_momentum = np.nansum(momentum_effects)
    
    # Calculate how strong and balanced the momentum effects are
    if np.isnan(avg_abs_momentum):
        overall_classification = "insufficient_data"
    elif avg_abs_momentum < momentum_threshold:
        overall_classification = "free_energy_dominated"
    elif abs(sum_momentum) < 0.2 * np.nansum(np.abs(momentum_effects)):
        overall_classification = "symmetric_momentum_dominated"
    else:
        overall_classification = "directional_momentum_dominated"
    
    # Analyze flat energy surface with high momenta (special case)
    avg_abs_free_energy = np.nanmean(np.abs(delta_G))
    avg_probabilities = np.nanmean([np.nanmean(q_matrix[i, :]) for i in range(n_interfaces)])
    
    if avg_abs_free_energy < 0.2 and avg_probabilities > 0.7:
        # Special case: flat energy with high transition probabilities
        # This indicates strong momentum effects on a flat landscape
        if overall_classification == "free_energy_dominated":
            overall_classification = "flat_high_momentum"
    
    return {
        'free_energy_differences': delta_G,
        'diffusive_probabilities': diffusive_q,
        'momentum_effects': momentum_effects,
        'momentum_significance': momentum_significance,
        'classification': pair_classification,
        'overall_classification': overall_classification,
        'avg_momentum_effect': avg_abs_momentum,
        'avg_free_energy': avg_abs_free_energy,
        'avg_probabilities': avg_probabilities
    }

def estimate_free_energy_differences(interfaces, q_matrix, q_weights=None, min_samples=5, account_for_distances=True):
    """
    Estimate free energy differences between interfaces from conditional crossing probabilities,
    taking into account physical distances between interfaces. Uses only the first valid forward
    and backward estimates (closest to the transition) to eliminate biases from long-term memory effects.
    
    Parameters:
    -----------
    interfaces : list or array
        The positions of the interfaces along the reaction coordinate
    q_matrix : numpy.ndarray
        Matrix of conditional crossing probabilities where q[i,k] is the probability
        that a path starting with a turn at interface i and reaching k-1 (for i<k) 
        or k+1 (for i>k) will make a turn at interface k
    q_weights : numpy.ndarray, optional
        Matrix of sample counts for each q_matrix value
    min_samples : int, optional
        Minimum number of samples required to consider a q value valid
    account_for_distances : bool, optional
        Whether to account for physical distances between interfaces (default: True).
        If False, interfaces are treated as having uniform spacing regardless of their actual values.
        
    Returns:
    --------
    delta_G : numpy.ndarray
        Matrix of free energy differences between interfaces
    """
    n_interfaces = len(interfaces)
    delta_G = np.zeros((n_interfaces, n_interfaces))
    delta_G.fill(np.nan)
    
    # Check if interfaces have physical values we can use
    has_physical_distances = isinstance(interfaces[0], (int, float)) and len(interfaces) > 1 and account_for_distances
    
    # Helper function to check if a q value is valid and usable
    def is_valid_q(i, k):
        return (not np.isnan(q_matrix[i, k]) and
                (q_weights is None or q_weights[i, k] >= min_samples) and
                abs(i-k) >= 2)
    
    # For each pair of adjacent interfaces, estimate the free energy difference
    for i in range(n_interfaces - 1):
        forward_estimate = None
        backward_estimate = None
        
        # Forward cascade: try q[i-1,i+1], then q[i-2,i+1], etc.
        # Only take the first valid estimate (closest to the transition)
        for start in range(i-1, -1, -1):  # Start from i-1 and go backwards to 0
            if is_valid_q(start, i+1):
                # First valid forward transition from 'start' to i+1 (going through i)
                q_fw = q_matrix[start, i+1]
                weight = q_weights[start, i+1] if q_weights is not None else 1.0
                
                # Adjust for physical distances
                if has_physical_distances and i > 0:
                    dist_i_to_ip1 = interfaces[i+1] - interfaces[i]
                    dist_im1_to_i = interfaces[i] - interfaces[i-1]
                    geo_q = dist_im1_to_i / (dist_i_to_ip1 + dist_im1_to_i) if (dist_i_to_ip1 + dist_im1_to_i) > 0 else 1.0
                    dG_obs = -np.log(q_fw / (1 - q_fw))
                    dG_geo = -np.log(geo_q / (1 - geo_q))
                    dG = dG_obs - dG_geo
                else:
                    dG = -np.log(q_fw / (1 - q_fw))
                
                forward_estimate = np.nan_to_num(dG, posinf=6.0, neginf=-6.0)
                print(rf"Forward $\Delta$G estimate for interfaces {i}-{i+1}: {dG:.4f} using q[{start},{i+1}]={q_fw:.4f}, weight={weight:.1f}")
                break  # Take only the first valid estimate
        
        # Backward cascade: try q[i+2,i], then q[i+3,i], etc.
        # Only take the first valid estimate (closest to the transition)
        for start in range(i+2, n_interfaces):
            if is_valid_q(start, i):
                # First valid backward transition from 'start' to i (going through i+1)
                q_bw = q_matrix[start, i]
                weight = q_weights[start, i] if q_weights is not None else 1.0
                
                # Adjust for physical distances
                if has_physical_distances and i+1 < n_interfaces-1:
                    dist_ip1_to_i = interfaces[i+1] - interfaces[i]
                    dist_ip2_to_ip1 = interfaces[i+2] - interfaces[i+1]
                    geo_q = dist_ip2_to_ip1 / (dist_ip2_to_ip1 + dist_ip1_to_i) if (dist_ip2_to_ip1 + dist_ip1_to_i) > 0 else 1.0
                    dG_obs = np.log(q_bw / (1 - q_bw))
                    dG_geo = np.log(geo_q / (1 - geo_q))
                    dG = dG_obs - dG_geo
                else:
                    dG = np.log(q_bw / (1 - q_bw))
                
                backward_estimate = np.nan_to_num(dG, posinf=6.0, neginf=-6.0)
                print(rf"Backward $\Delta$G estimate for interfaces {i}-{i+1}: {dG:.4f} using q[{start},{i}]={q_bw:.4f}, weight={weight:.1f}")
                break  # Take only the first valid estimate
        
        # Combine forward and backward estimates if available
        if forward_estimate is not None and backward_estimate is not None:
            # Take the average of the forward and backward estimates
            # This helps eliminate biases from the timestep
            combined_dG = (forward_estimate + backward_estimate) / 2.0
            
            # Report the difference to highlight any systematic bias
            bias = forward_estimate - backward_estimate
            print(rf"Combined $\Delta$G for interfaces {i}-{i+1}: {combined_dG:.4f} (forward-backward bias: {bias:.4f})")
            
            delta_G[i, i+1] = combined_dG
            delta_G[i+1, i] = -combined_dG
            
        elif forward_estimate is not None:
            # Only forward estimate available
            delta_G[i, i+1] = forward_estimate
            delta_G[i+1, i] = -forward_estimate
            print(f"Using only forward estimate for interfaces {i}-{i+1}: {forward_estimate:.4f}")
            
        elif backward_estimate is not None:
            # Only backward estimate available
            delta_G[i, i+1] = backward_estimate
            delta_G[i+1, i] = -backward_estimate
            print(f"Using only backward estimate for interfaces {i}-{i+1}: {backward_estimate:.4f}")
            
        else:
            print(f"No valid estimates found for interfaces {i}-{i+1}")
    
    return delta_G

##################################
# Network Analysis Tools
##################################

def analyze_network_connectivity(M, source_state=0, sink_state=-1, max_paths=10):
    """
    Analyze the connectivity of a transition network efficiently without enumerating all paths.
    
    Parameters
    ----------
    M : np.ndarray
        Transition matrix representing the Markov state model
    source_state : int, optional
        Index of the source state (default: 0)
    sink_state : int, optional
        Index of the sink state (default: -1)
    max_paths : int, optional
        Maximum number of paths to sample for visualization (default: 10)
        
    Returns
    -------
    dict
        Dictionary containing connectivity analysis results
    """
    from scipy import sparse
    from scipy.sparse.csgraph import connected_components, shortest_path
    import networkx as nx

    n_states = M.shape[0]
    sink_idx = n_states - 1 if sink_state == -1 else sink_state
    
    # Generate state labels for the plot
    state_labels = generate_state_labels(n_states//2)
    
    # Create a graph representation where edges exist if M[i,j] > 0
    graph = (M > 0).astype(int)
    
    # Find strongly connected components (nodes with paths in both directions)
    n_strong, strong_labels = connected_components(
        sparse.csr_matrix(graph), directed=True, connection='strong'
    )
    
    # Find weakly connected components (nodes connected ignoring direction)
    n_weak, weak_labels = connected_components(
        sparse.csr_matrix(graph), directed=True, connection='weak'
    )
    
    # Check path existence and get predecessor information
    try:
        path_lengths, predecessors = shortest_path(
            sparse.csr_matrix(graph), directed=True, 
            indices=source_state, return_predecessors=True
        )
        direct_path_exists = np.isfinite(path_lengths[sink_idx])
    except:
        direct_path_exists = False
        predecessors = None
        path_lengths = None
    
    # Create NetworkX graph for visualization
    G = nx.DiGraph()
    
    # Add edges with transition probabilities as weights
    for i in range(n_states):
        for j in range(n_states):
            if M[i, j] > 0:
                G.add_edge(i, j, weight=M[i, j])
    
    # Find critical nodes using edge betweenness centrality
    # This identifies bottleneck edges without enumerating all paths
    if direct_path_exists:
        # Calculate edge betweenness centrality only for edges on paths between source and sink
        # This is much more efficient than calculating for the entire graph
        subgraph_nodes = set()
        
        # Use a different algorithm to find a sample of paths for visualization
        # First use Yen's algorithm to find k-shortest paths
        try:
            sample_paths = []
            for i, path in enumerate(nx.shortest_simple_paths(G, source_state, sink_idx, weight='weight')):
                if i >= max_paths:
                    break
                sample_paths.append(path)
                subgraph_nodes.update(path)
        except nx.NetworkXNoPath:
            sample_paths = []
        
        # If we couldn't get paths with the above method, reconstruct at least one path using predecessors
        if not sample_paths and predecessors is not None:
            path = [sink_idx]
            current = sink_idx
            while current != source_state:
                if current < 0 or predecessors[current] < 0:
                    # No path found
                    path = []
                    break
                current = predecessors[current]
                path.append(current)
            path.reverse()
            if path:
                sample_paths.append(path)
                subgraph_nodes.update(path)
        
        # Find critical nodes using a different approach
        if len(subgraph_nodes) > 2:  # If we have nodes besides source and sink
            # Create a subgraph containing only the nodes on the sampled paths
            subgraph = G.subgraph(subgraph_nodes).copy()
            
            # See if source and sink are still connected if we remove each node
            critical_nodes = set()
            for node in subgraph_nodes:
                # Skip source and sink
                if node == source_state or node == sink_idx:
                    continue
                    
                # Remove node and check connectivity
                temp_graph = subgraph.copy()
                temp_graph.remove_node(node)
                if not nx.has_path(temp_graph, source_state, sink_idx):
                    critical_nodes.add(node)
        else:
            critical_nodes = set()
    else:
        sample_paths = []
        critical_nodes = set()
    
    # Create visualization
    plt.figure(figsize=(10, 8))
    
    # Use a more deterministic layout if possible
    try:
        pos = nx.kamada_kawai_layout(G)
    except:
        pos = nx.spring_layout(G, seed=42)  # Use seed for reproducibility
    
    # Node colors based on strongly connected component
    node_colors = [strong_labels[i] for i in range(len(pos.values()))]
    
    # Draw the basic graph with node labels using state_labels
    nx.draw(G, pos, with_labels=True, node_color=node_colors, 
           labels={int(i): state_labels[i] for i in pos.keys()},
           cmap=plt.cm.tab10, node_size=500, alpha=0.8)
    
    # Highlight source and sink
    nx.draw_networkx_nodes(G, pos, nodelist=[source_state], 
                          node_color='green', node_size=700)
    nx.draw_networkx_nodes(G, pos, nodelist=[sink_idx], 
                          node_color='red', node_size=700)
    
    # Highlight critical nodes
    if critical_nodes:
        nx.draw_networkx_nodes(G, pos, nodelist=list(critical_nodes), 
                              node_color='yellow', node_size=600)
    
    # If a direct path exists, highlight one of the paths
    if sample_paths:
        shortest = sample_paths[0]  # Just use the first path
        path_edges = list(zip(shortest[:-1], shortest[1:]))
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, 
                              edge_color='red', width=2)
    
    plt.title(f"Network Analysis: {n_strong} Strong Components, {n_weak} Weak Components")
    
    # Create legend patches with state_labels for source and sink
    legend_patches = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', 
                 markersize=10, label=f'Source State: {state_labels[source_state]}'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                 markersize=10, label=f'Sink State: {state_labels[sink_idx]}')
    ]
    if critical_nodes:
        legend_patches.append(
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='yellow', 
                     markersize=10, label='Critical Bridge States')
        )
    plt.legend(handles=legend_patches, loc='upper right')
    
    return {
        'n_strong_components': n_strong,
        'strong_component_labels': strong_labels,
        'n_weak_components': n_weak,
        'weak_component_labels': weak_labels,
        'direct_path_exists': direct_path_exists,
        'critical_nodes': critical_nodes,
        'sample_paths': sample_paths,
        'network_graph': G
    }

def find_network_bottlenecks(M, source_state=0, sink_state=-1):
    """
    Identify bottlenecks in the transition network where removing
    a small number of edges would disconnect source from sink.
    
    Parameters
    ----------
    M : np.ndarray
        Transition matrix representing the Markov state model
    source_state : int, optional
        Index of the source state (default: 0)
    sink_state : int, optional
        Index of the sink state (default: -1)
        
    Returns
    -------
    dict
        Dictionary containing bottleneck analysis results
    """
    import networkx as nx
    
    n_states = M.shape[0]
    sink_idx = n_states - 1 if sink_state == -1 else sink_state
    
    # Generate state labels for the plot
    state_labels = generate_state_labels(n_states//2)
    
    # Create weighted directed graph
    G = nx.DiGraph()
    
    # Add edges with transition probabilities as weights
    for i in range(n_states):
        for j in range(n_states):
            if M[i,j] > 0:
                G.add_edge(i, j, capacity=M[i,j])
    
    # Calculate minimum cut
    try:
        cut_value, partition = nx.minimum_cut(G, source_state, sink_idx)
        reachable, non_reachable = partition
        
        # Find the edges in the cut
        cut_edges = []
        for u in reachable:
            for v in non_reachable:
                if G.has_edge(u, v):
                    cut_edges.append((u, v))
        
        # Visualize the network with cut highlighted
        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(G)
        
        # Draw the basic graph with updated node labels
        nx.draw(G, pos, with_labels=True, node_color='lightblue', 
               labels={int(i): state_labels[int(i)] for i in pos.keys()},
               node_size=500, alpha=0.8)
        
        # Highlight source and sink
        nx.draw_networkx_nodes(G, pos, nodelist=[source_state], 
                              node_color='green', node_size=700)
        nx.draw_networkx_nodes(G, pos, nodelist=[sink_idx], 
                              node_color='red', node_size=700)
        
        # Highlight cut edges
        nx.draw_networkx_edges(G, pos, edgelist=cut_edges, 
                              edge_color='red', width=2, style='dashed')
        
        plt.title(f"Network Min-Cut Analysis: Cut Value = {cut_value:.4f}")
        
        # Create legend with state labels for source and sink
        legend_patches = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', 
                     markersize=10, label=f'Source State: {state_labels[source_state]}'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                     markersize=10, label=f'Sink State: {state_labels[sink_idx]}'),
            plt.Line2D([0], [0], color='red', linestyle='dashed', 
                     label='Min-Cut Edges')
        ]
        plt.legend(handles=legend_patches, loc='upper right')
        
        return {
            'cut_value': cut_value,
            'reachable_from_source': reachable,
            'non_reachable_from_source': non_reachable,
            'cut_edges': cut_edges
        }
    except nx.NetworkXError:
        print("No path exists from source to sink.")
        return None
    
def calculate_effective_transitions(M, source_state=0, sink_state=-1, max_steps=100):
    """
    Calculate effective transition probabilities from source to sink,
    factoring in the possibility of multi-step transitions.
    
    Parameters
    ----------
    M : np.ndarray
        Transition matrix representing the Markov state model
    source_state : int, optional
        Index of the source state (default: 0)
    sink_state : int, optional
        Index of the sink state (default: -1)
    max_steps : int, optional
        Maximum number of steps to consider (default: 100)
        
    Returns
    -------
    dict
        Dictionary containing effective transition probabilities and paths
    """    
    n_states = M.shape[0]
    sink_idx = n_states - 1 if sink_state == -1 else sink_state
    
    # Generate state labels for the plot
    state_labels = generate_state_labels(n_states)
    
    # Initialize probability matrix for different number of steps
    P = np.zeros((max_steps+1, n_states, n_states))
    P[0] = np.eye(n_states)  # Identity matrix for 0 steps
    P[1] = M  # 1-step transition is just M
    
    # Calculate multi-step transition matrices
    for i in range(2, max_steps+1):
        P[i] = np.dot(P[i-1], M)
    
    # Extract source-to-sink probabilities for each number of steps
    source_sink_probs = np.zeros(max_steps+1)
    for i in range(max_steps+1):
        source_sink_probs[i] = P[i][source_state, sink_idx]
    
    # Calculate cumulative probability (probability of reaching within n steps)
    # This accounts for the absorbing nature of the sink state
    cumulative_probs = np.zeros(max_steps+1)
    curr_prob = 0
    
    for i in range(1, max_steps+1):
        # Probability of reaching in exactly i steps without having reached earlier
        new_arrival_prob = source_sink_probs[i] - curr_prob
        if new_arrival_prob < 0:  # Numerical issues
            new_arrival_prob = 0
        curr_prob += new_arrival_prob
        cumulative_probs[i] = curr_prob
    
    # Plot the results
    plt.figure(figsize=(10, 6))
    
    plt.plot(range(max_steps+1), source_sink_probs, 'b.-', label='N-step probability')
    plt.plot(range(max_steps+1), cumulative_probs, 'r.-', label='Cumulative probability')
    
    plt.xlabel('Number of Steps')
    plt.ylabel('Probability')
    plt.title(f'Effective Transition Probability from {state_labels[source_state]} to {state_labels[sink_idx]}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    return {
        'n_step_probabilities': source_sink_probs,
        'cumulative_probabilities': cumulative_probs,
        'effective_probability': cumulative_probs[-1]
    }

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
    # Generate state labels for printing
    n_states = M.shape[0]
    state_labels = generate_state_labels(n_states)
    
    # Ensure the transition matrix is a valid stochastic matrix
    assert np.allclose(M.sum(axis=1), 1), "Rows of the transition matrix must sum to 1."

    # Create a MarkovStateModel object
    msm = dpt.markov.msm.MarkovStateModel(M)

    # Perform PCCA+ analysis
    pcca = dpt.markov.pcca(M, n_clusters)

    # Print the membership matrix
    print("PCCA+ Membership Matrix:")
    print(pcca.memberships)

    # Print the metastable states with state labels
    print("Metastable States:")
    for i, state_set in enumerate(pcca.sets):
        # Convert state indices to state labels
        labeled_states = [state_labels[idx] for idx in state_set]
        print(f"State {i+1}: {labeled_states} (indices: {state_set})")

    return pcca

###################################
# Older Memory Analysis Functions
###################################

def analyze_memory_vs_free_energy_effects(interfaces, q_matrix, q_weights=None, min_samples=5, memory_threshold=0.3):
    """
    Analyze whether observed turn-based transition probabilities between interfaces are best explained
    by free energy differences or by memory effects (history dependence).
    
    This function compares observed conditional crossing probabilities (q_i,k) against what would
    be expected from a purely diffusive model with free energy differences. Significant deviations
    indicate memory effects where future turns depend on the history of previous turns.
    
    Parameters
    ----------
    interfaces : list or array
        The positions of the interfaces along the reaction coordinate
    q_matrix : numpy.ndarray
        Matrix of conditional crossing probabilities where q[i,k] is the probability
        that a path starting with a turn at interface i and reaching k-1 (for i<k) or k+1 (for i>k)
        will make a turn at interface k
    q_weights : numpy.ndarray, optional
        Matrix of sample counts for each q_matrix value (for statistical confidence)
    min_samples : int, optional
        Minimum number of samples required to consider a q value valid
    memory_threshold : float, optional
        Threshold for determining significant memory effects (relative deviation)
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'free_energy_differences': Matrix of estimated free energy differences
        - 'predicted_q_matrix': Matrix of predicted q values from free energy model
        - 'memory_effects': Matrix quantifying the memory effect strength
        - 'memory_significance': Matrix indicating if memory effects are significant
        - 'interface_classification': Classification of each interface as 
          'free_energy_dominated', 'memory_dominated', or 'mixed'
        - 'overall_classification': Overall assessment of the system
        - 'turn_based_metrics': Additional metrics specific to turn-based analysis
          
    Notes
    -----
    The function accounts for the turn-based nature of q_i,k values by:
    1. Recognizing that q_i,k represents the probability of a trajectory making a turn at k
       after previously making a turn at i and reaching k-1 or k+1
    2. Differentiating between the forward (i<k) and backward (i>k) turn-based transitions
    3. Comparing with an appropriate diffusive reference model for turn-based dynamics
    
    Memory effects in turn-based dynamics often indicate the presence of hidden slow variables
    or complex kinetics that aren't captured by the chosen reaction coordinate.
    """    
    n_interfaces = len(interfaces)
    
    # Step 1: Estimate free energy differences between adjacent interfaces
    # This accounts for the turn-based nature of the system
    delta_G = estimate_free_energy_differences(interfaces, q_matrix, q_weights, min_samples)
    
    # Step 2: Predict conditional crossing probabilities from the free energy model
    # These are the expected q values for a memory-less diffusive process with the same free energy profile
    predicted_q = np.zeros_like(q_matrix)
    predicted_q.fill(np.nan)
    
    # For diagonal elements (self-transitions), probability is 0
    for i in range(n_interfaces):
        predicted_q[i, i] = 0.0
    
    # Calculate reference probabilities based solely on local free energy differences
    # For forward transitions (i<k):
    for k in range(1, n_interfaces):
        # The diffusive reference q(i,k) depends only on the free energy difference between k-1 and k
        # not on the starting interface i
        ref_prob = 0.5  # Default value
        
        if not np.isnan(delta_G[k-1, k]):
            # Use Boltzmann factor for the free energy difference
            ref_prob = 1.0 / (1.0 + np.exp(delta_G[k-1, k]))
        
        # Apply this reference to all starting interfaces i < k
        # But for adjacent interfaces (i=k-1), the probability is 1.0 by definition
        for i in range(k):
            if i == k-1:
                # Adjacent interfaces always have probability 1.0
                predicted_q[i, k] = 1.0
            else:
                # Non-adjacent interfaces use the diffusive reference
                predicted_q[i, k] = ref_prob
    
    # For backward transitions (i>k):
    for k in range(n_interfaces-1):
        # The diffusive reference q(i,k) depends only on the free energy difference between k+1 and k
        # not on the starting interface i
        ref_prob = 0.5  # Default value
        
        if not np.isnan(delta_G[k+1, k]):
            # Use Boltzmann factor for the free energy difference
            ref_prob = 1.0 / (1.0 + np.exp(delta_G[k+1, k]))
        
        # Apply this reference to all starting interfaces i > k
        # But for adjacent interfaces (i=k+1), the probability is 1.0 by definition
        for i in range(k+1, n_interfaces):
            if i == k+1:
                # Adjacent interfaces always have probability 1.0
                predicted_q[i, k] = 1.0
            else:
                # Non-adjacent interfaces use the diffusive reference
                predicted_q[i, k] = ref_prob
    
    # Step 3: Calculate memory effects as deviations from predicted values
    memory_effects = np.zeros_like(q_matrix)
    memory_effects.fill(np.nan)
    
    # Calculate absolute and relative memory effects
    for i in range(n_interfaces):
        for k in range(n_interfaces):
            if i != k and not np.isnan(q_matrix[i, k]) and not np.isnan(predicted_q[i, k]):
                # Only consider values with enough samples
                if q_weights is None or q_weights[i, k] >= min_samples:
                    # Calculate relative deviation
                    if predicted_q[i, k] > 0:
                        # Relative deviation as percentage
                        memory_effects[i, k] = (q_matrix[i, k] - predicted_q[i, k]) / predicted_q[i, k]
    
    # Step 4: Determine significance of memory effects
    memory_significance = np.zeros_like(q_matrix, dtype=bool)
    
    for i in range(n_interfaces):
        for k in range(n_interfaces):
            if not np.isnan(memory_effects[i, k]):
                memory_significance[i, k] = abs(memory_effects[i, k]) > memory_threshold
    
    # Step 5: Specialized analysis for turn-based dynamics
    turn_persistence = analyze_turn_persistence(q_matrix, q_weights, min_samples)
    
    # Step 6: Classify each interface by dominant effect type
    interface_classification = {}
    
    for i in range(n_interfaces):
        # Count significant memory effects for transitions starting at this interface
        memory_count = np.sum(memory_significance[i, :])
        total_transitions = np.sum(~np.isnan(memory_effects[i, :]))
        
        if total_transitions == 0:
            interface_classification[i] = "insufficient_data"
        elif memory_count / total_transitions > 0.5:
            interface_classification[i] = "memory_dominated"
        elif memory_count / total_transitions < 0.2:
            interface_classification[i] = "free_energy_dominated"
        else:
            interface_classification[i] = "mixed"
    
    # Step 7: Classify the overall system
    total_significant_memory = np.sum(memory_significance)
    total_valid_transitions = np.sum(~np.isnan(memory_effects))
    
    if total_valid_transitions == 0:
        overall_classification = "insufficient_data"
    elif total_significant_memory / total_valid_transitions > 0.5:
        overall_classification = "memory_dominated"
    elif total_significant_memory / total_valid_transitions < 0.2:
        overall_classification = "free_energy_dominated"
    else:
        overall_classification = "mixed"
    
    # Step 8: Calculate directional memory effects (forward vs backward)
    forward_memory = 0
    forward_count = 0
    backward_memory = 0
    backward_count = 0
    
    for i in range(n_interfaces):
        for k in range(n_interfaces):
            if i < k and not np.isnan(memory_effects[i, k]):  # Forward
                forward_memory += abs(memory_effects[i, k])
                forward_count += 1
            elif i > k and not np.isnan(memory_effects[i, k]):  # Backward
                backward_memory += abs(memory_effects[i, k])
                backward_count += 1
    
    # Average memory effects in each direction
    avg_forward_memory = forward_memory / forward_count if forward_count > 0 else 0
    avg_backward_memory = backward_memory / backward_count if backward_count > 0 else 0
    
    # Return comprehensive analysis results
    return {
        'free_energy_differences': delta_G,
        'predicted_q_matrix': predicted_q,
        'memory_effects': memory_effects,
        'memory_significance': memory_significance,
        'interface_classification': interface_classification,
        'overall_classification': overall_classification,
        'avg_forward_memory': avg_forward_memory,
        'avg_backward_memory': avg_backward_memory,
        'forward_vs_backward': 'forward_dominated' if avg_forward_memory > avg_backward_memory * 1.5 else 
                               'backward_dominated' if avg_backward_memory > avg_forward_memory * 1.5 else
                               'balanced',
        'turn_based_metrics': turn_persistence
    }

def analyze_turn_persistence(q_matrix, q_weights=None, min_samples=5):
    """
    Analyze turn persistence patterns in the system to detect memory effects specific to turn-based dynamics.
    
    Parameters
    ----------
    q_matrix : numpy.ndarray
        Matrix of conditional crossing probabilities
    q_weights : numpy.ndarray, optional
        Matrix of sample counts for each q_matrix value
    min_samples : int, optional
        Minimum number of samples required to consider a q value valid
    
    Returns
    -------
    dict
        Dictionary containing turn persistence metrics
    """    
    n_interfaces = q_matrix.shape[0]
    
    # Calculate average turn skipping - how many interfaces we typically jump over when making turns
    avg_turn_skip = np.zeros(n_interfaces)
    avg_turn_skip.fill(np.nan)
    
    for i in range(n_interfaces):
        valid_distances = []
        valid_weights = []
        
        for k in range(n_interfaces):
            if i != k and not np.isnan(q_matrix[i, k]):
                # Only consider values with enough samples
                if q_weights is None or q_weights[i, k] >= min_samples:
                    # Distance between turns (number of interfaces skipped)
                    distance = abs(k - i)
                    weight = q_matrix[i, k]
                    if q_weights is not None:
                        sample_weight = q_weights[i, k]
                    else:
                        sample_weight = 1
                    
                    valid_distances.append(distance)
                    valid_weights.append(weight * sample_weight)
        
        if valid_distances:
            # Calculate weighted average distance
            avg_turn_skip[i] = np.average(valid_distances, weights=valid_weights)
    
    # Calculate turn asymmetry - tendency to make forward vs backward turns
    turn_asymmetry = np.zeros(n_interfaces)
    turn_asymmetry.fill(np.nan)
    
    for i in range(1, n_interfaces-1):  # Skip boundary interfaces
        forward_prob = 0
        backward_prob = 0
        forward_count = 0
        backward_count = 0
        
        for k in range(n_interfaces):
            if k > i and not np.isnan(q_matrix[i, k]):
                # Forward turn
                if q_weights is None or q_weights[i, k] >= min_samples:
                    forward_prob += q_matrix[i, k]
                    forward_count += 1
            elif k < i and not np.isnan(q_matrix[i, k]):
                # Backward turn
                if q_weights is None or q_weights[i, k] >= min_samples:
                    backward_prob += q_matrix[i, k]
                    backward_count += 1
        
        # Normalize by count
        if forward_count > 0:
            forward_prob /= forward_count
        if backward_count > 0:
            backward_prob /= backward_count
        
        # Calculate asymmetry (-1 to 1 scale)
        if forward_prob + backward_prob > 0:
            turn_asymmetry[i] = (forward_prob - backward_prob) / (forward_prob + backward_prob)
    
    # Calculate transition sharpness - how focused the transitions are
    # High values indicate sharp transitions (focused on specific targets)
    # Low values indicate diffuse transitions (spread across many targets)
    transition_sharpness = np.zeros(n_interfaces)
    transition_sharpness.fill(np.nan)
    
    for i in range(n_interfaces):
        valid_probs = []
        
        for k in range(n_interfaces):
            if i != k and not np.isnan(q_matrix[i, k]):
                if q_weights is None or q_weights[i, k] >= min_samples:
                    valid_probs.append(q_matrix[i, k])
        
        if len(valid_probs) > 1:
            # Calculate entropy-based measure of sharpness
            # Normalize probabilities first
            norm_probs = np.array(valid_probs) / np.sum(valid_probs)
            # Calculate entropy (higher entropy = lower sharpness)
            entropy = -np.sum(norm_probs * np.log(norm_probs + 1e-10))
            # Convert to sharpness (1 - normalized entropy)
            max_entropy = np.log(len(valid_probs))
            if max_entropy > 0:
                transition_sharpness[i] = 1.0 - entropy / max_entropy
            else:
                transition_sharpness[i] = np.nan
    
    return {
        'avg_turn_skip': avg_turn_skip,
        'turn_asymmetry': turn_asymmetry,
        'transition_sharpness': transition_sharpness
    }
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
import deeptime.markov as dpt

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
    M[N+1:, 2] = P[1:, 0]  # Transitions from states N+1 and beyond to state 2

    # Set up transitions for other states
    for i in range(1, N):
        M[2+i, N+i:2*N] = P[i, i:]  # Transitions from state 2+i to states N+i and beyond
        M[N+i, 3:2+i] = P[i, 1:i]   # Transitions from state N+i to states 3 through 2+i

    # Nonsampled paths
    if not M[N, -1] >= 0:
        M[N, -1] = 1
    
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
    p = np.empty([w_path[0].shape[0], w_path[0].shape[0]])
    q = np.ones([w_path[0].shape[0], w_path[0].shape[0]])
    
    # Calculate q(i,k) - probability to go from i to k via direct transitions
    for i in range(w_path[0].shape[0]):
        for k in range(w_path[0].shape[0]):
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
                    if pe_i > w_path[0].shape[0]-1:
                        break
                    counts += [np.sum(w_path[pe_i][i][k:]), np.sum(w_path[pe_i][i][k-1:])]
                    # print(pe_i-1, i, k, np.sum(w_path[pe_i][i][k:])/np.sum(w_path[pe_i][i][k-1:]), np.sum(w_path[pe_i][i][k-1:]))
            elif i > k:
                # Backward transitions (R→L)
                for pe_i in range(k+2, i+2):
                    if pe_i > w_path[0].shape[0]-1:
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
    for i in range(w_path[0].shape[0]):
        for k in range(w_path[0].shape[0]):
            if i < k:
                # Forward transitions
                if k == w_path[0].shape[0]-1:
                    p[i][k] = np.prod(q[i][i+1:k+1])
                else:
                    p[i][k] = np.prod(q[i][i+1:k+1]) * (1-q[i][k+1])
            elif k < i:
                # Backward transitions
                if k == 0:
                    p[i][k] = np.prod(q[i][k:i])
                else:
                    p[i][k] = np.prod(q[i][k:i]) * (1-q[i][k-1])
                # if i == w_path[0].shape[0]-1:
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
    
    This function provides an alternative method to get_transition_probzz() for calculating
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
    sh = w_path[0].shape
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
    # Initialize probability matrix
    p = np.empty([w_path[0].shape[0], w_path[0].shape[0]])

    # Calculate transition probabilities for each pair of interfaces
    for i in range(w_path[0].shape[0]):
        for k in range(w_path[0].shape[0]):
            if i < k:
                # Forward transitions
                if i == 0 or i >= w_path[0].shape[0]-2:
                    if k == w_path[0].shape[0]-1:
                        p[i][k] = np.sum(w_path[i+1][i][k:]) / np.sum(w_path[i+1][i][i:])
                    else:
                        p[i][k] = (w_path[i+1][i][k]) / np.sum(w_path[i+1][i][i:])
                else:
                    # Use both i+1 and i+2 ensembles for calculation
                    p[i][k] = (w_path[i+1][i][k] + w_path[i+2][i][k]) / (np.sum(w_path[i+1][i][i:]) + np.sum(w_path[i+2][i][i:]))
            elif k < i:
                # Backward transitions
                if i == w_path[0].shape[0]-1:
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
    for i in range(len(interfaces)):
        if tr:
            if i == 2 and X[i][0, 1] == 0:     # In [1*] all LML paths are classified as 1 -> 0 (for now).
                X[i][1, 0] *= 2     # Time reversal needs to be adjusted to compensate for this
            X[i] += X[i].T          # Will not be needed anymore once LML paths are separated in 0 -> 1 and 1 -> 0.
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
    print(f"Sum weights ensemble {pe_id}: ", np.sum(X_path))

    X = X_path
    if tr:
        if pe_id == 2 and X[0, 1] == 0:     # In [1*] all LML paths are classified as 1 -> 0 (for now).
            X[1, 0] *= 2     # Time reversal needs to be adjusted to compensate for this
        X += X.T          # Will not be needed anymore once LML paths are separated in 0 -> 1 and 1 -> 0.
    return X


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
    q_k = np.zeros([2, w_path[0].shape[0]-1, w_path[0].shape[0], w_path[0].shape[0]])
    for ens in range(1, w_path[0].shape[0]):
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
        for intf in range(ens, w_path[0].shape[0]):
            print(f"-> interface {intf-1} is reached, probability to reach interface {intf}:")
            for start in range(ens):
                print(f"    - START at interface {start}: {q_k[0][ens-1][start][intf]} | # weights: {q_k[1][ens-1][start][intf]}")
        print("==== R -> L ====")
        for intf in range(ens-1):
            print(f"-> interface {intf+1} is reached, probability to reach interface {intf}:")
            for start in range(ens-1, w_path[0].shape[0]):
                print(f"    - START at interface {start}: {q_k[0][ens-1][start][intf]} | # weights: {q_k[1][ens-1][start][intf]}")

    q_tot = np.ones([2, w_path[0].shape[0], w_path[0].shape[0]])
    for i in range(w_path[0].shape[0]):
        for k in range(w_path[0].shape[0]):
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
                    if pe_i > w_path[0].shape[0]-1:
                        break
                    counts += [np.sum(w_path[pe_i][i][k:]), np.sum(w_path[pe_i][i][k-1:])]
            elif i > k:
                for pe_i in range(k+2, i+2):
                    if pe_i > w_path[0].shape[0]-1:
                        break
                    counts += [np.sum(w_path[pe_i][i][:k+1]), np.sum(w_path[pe_i][i][:k+2])]

            q_tot[0][i][k] = counts[0] / counts[1] if counts[1] > 0 else np.nan
            q_tot[1][i][k] = counts[1]
    print()
    print(20*'-')
    print(f"TOTAL - ALL ENSEMBLES")
    print(20*'-')
    print("==== L -> R ====")
    for intf in range(1, w_path[0].shape[0]):
        print(f"-> interface {intf-1} is reached, probability to reach interface {intf}:")
        for start in range(intf):
            print(f"    - START at interface {start}: {q_tot[0][start][intf]} | # weights: {q_tot[1][start][intf]}")
    print("==== R -> L ====")
    for intf in range(w_path[0].shape[0]-2):
        print(f"-> interface {intf+1} is reached, probability to reach interface {intf}:")
        for start in range(intf+1, w_path[0].shape[0]):
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

def plot_memory_analysis0(q_tot, p, interfaces=None):
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
        for i in range(k):
            if not np.isnan(q_probs[i, k]) and q_weights[i, k] > 5:  # Minimum sample threshold
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
    ax4.grid(True, alpha=0.3)
    ax4.legend(title='Target Interface k', loc='best')
    
    # Plot 2.2: Backward Transition Probabilities (R→L)
    ax5 = fig2.add_subplot(gs2[0, 1])
    
    # For each target interface k, plot q(i,k) for all starting interfaces i>k
    for idx, k in enumerate(backward_targets):
        target_data = []
        starting_interfaces = []
        for i in range(k+1, n_interfaces):
            if not np.isnan(q_probs[i, k]) and q_weights[i, k] > 5:  # Minimum sample threshold
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
    ax5.grid(True, alpha=0.3)
    ax5.legend(title='Target Interface k', loc='best')
    
    # Plot 2.3: Forward Memory Retention
    ax6 = fig2.add_subplot(gs2[1, 0])
    
    # Calculate memory retention using relative standard deviation
    memory_retention = np.zeros(n_interfaces)
    
    for k in range(1, n_interfaces):
        # Get all probabilities for reaching k from different starting points
        # Exclude i=k (q(i,i)=0) and i=k-1 (q(i,i+1)=1)
        probs = [q_probs[i, k] for i in range(k-1) if not np.isnan(q_probs[i, k]) and q_weights[i, k] > 5]
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
        ax6.grid(True, axis='y', alpha=0.3)
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
        ax7.grid(True, axis='y', alpha=0.3)
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
    ax10.grid(True, alpha=0.3)
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

    # Create grouped bar plot for adjacent transitions with consistent colors
    x = np.arange(n_interfaces - 1)
    width = 0.35

    # Use the same colors from forward/backward transitions plots for consistency
    forward_base_color = 'tab:blue'   # Base color for forward transitions
    backward_base_color = 'tab:orange'  # Base color for backward transitions

    # Create bars with consistent colors that match the legend
    bar1 = ax11.bar(x - width/2, adjacent_forward, width, label='Forward (i → i+1)', 
                color=forward_base_color, edgecolor='black', linewidth=0.5, 
                alpha=(0.4 if v < 0 else 0.85 for v in adjacent_forward))

    bar2 = ax11.bar(x + width/2, adjacent_backward, width, label='Backward (i+1 → i)', 
                color=backward_base_color, edgecolor='black', linewidth=0.5,
                alpha=(0.4 if v < 0 else 0.85 for v in adjacent_backward))

    # Add zero line
    ax11.axhline(y=0, color='k', linestyle='-', alpha=0.5)

    # Add shaded regions to indicate bias direction
    ax11.axhspan(0, y_limit, alpha=0.1, color='green', label='Bias toward crossing')
    ax11.axhspan(-y_limit, 0, alpha=0.1, color='red', label='Bias toward returning')

    # Add annotations with improved visibility
    for i, v in enumerate(adjacent_forward):
        if not np.isnan(v):
            offset = 0.02 if v >= 0 else -0.08
            ax11.text(i - width/2, v + offset, f"{v:.2f}", ha='center', fontsize=9, 
                    fontweight='bold', color='black')

    for i, v in enumerate(adjacent_backward):
        if not np.isnan(v):
            offset = 0.02 if v >= 0 else -0.08
            ax11.text(i + width/2, v + offset, f"{v:.2f}", ha='center', fontsize=9, 
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
    ax11.set_xticks(x)
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
    diff_ref = calculate_diffusive_reference(interfaces, q_tot[0], q_tot[1])
    
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
    cbar1.ax.axhline(y=0.5, color='black', linestyle='--', linewidth=1)
    cbar1.ax.text(1.5, 0.5, '0 (diffusive)', va='center', ha='left', fontsize=9)
    
    # Add annotations for q values, diffusive references, and sample sizes
    for i in range(n_interfaces):
        for j in range(n_interfaces):
            if not np.isnan(memory_effect[i, j]) and not np.ma.is_masked(masked_data[i, j]):
                weight = q_weights[i, j]
                text = f"{q_probs[i, j]:.2f}/{diff_ref[i, j]:.2f}\n(n={int(weight)})" if weight > 0 else "N/A"
                color = 'black' if abs(memory_effect[i, j]) < 0.3 else 'white'
                ax1.text(j, i, text, ha='center', va='center', color=color, fontsize=8)
    
    # Set ticks and labels
    ax1.set_xticks(np.arange(n_interfaces))
    ax1.set_yticks(np.arange(n_interfaces))
    if is_equidistant:
        ax1.set_xticklabels([f"{i}" for i in range(n_interfaces)])
        ax1.set_yticklabels([f"{i}" for i in range(n_interfaces)])
    else:
        ax1.set_xticklabels([f"{interfaces[i]:.2f}" for i in range(n_interfaces)])
        ax1.set_yticklabels([f"{interfaces[i]:.2f}" for i in range(n_interfaces)])
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
    
    # Add annotations for ratio values
    for i in range(n_interfaces):
        for j in range(n_interfaces):
            if not np.isnan(memory_ratio[i, j]) and q_weights[i, j] > 5:
                text_color = 'black'
                if memory_ratio[i, j] > 5 or memory_ratio[i, j] < 0.2:
                    text_color = 'white'
                ax2.text(j, i, f"{memory_ratio[i, j]:.2f}", ha='center', va='center', 
                       color=text_color, fontsize=8)
    
    ax2.set_xlabel('Target Interface k')
    ax2.set_ylabel('Starting Interface i')
    ax2.set_title('Memory Effect Ratio: Deviation from Diffusive Behavior', fontsize=12)
    ax2.set_xticks(range(n_interfaces))
    ax2.set_yticks(range(n_interfaces))
    if is_equidistant:
        ax2.set_xticklabels([f"{i}" for i in range(n_interfaces)])
        ax2.set_yticklabels([f"{i}" for i in range(n_interfaces)])
    else:
        ax2.set_xticklabels([f"{interfaces[i]:.2f}" for i in range(n_interfaces)])
        ax2.set_yticklabels([f"{interfaces[i]:.2f}" for i in range(n_interfaces)])
    
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
    if is_equidistant:
        ax3.set_xticklabels([f"{i}" for i in range(n_interfaces)])
        ax3.set_yticklabels([f"{i}" for i in range(n_interfaces)])
    else:
        ax3.set_xticklabels([f"{interfaces[i]:.2f}" for i in range(n_interfaces)])
        ax3.set_yticklabels([f"{interfaces[i]:.2f}" for i in range(n_interfaces)])
    
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
        Format: actual/diffusive (sample count)
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
            if not np.isnan(q_probs[i, k]) and q_weights[i, k] > 5:  # Minimum sample threshold
                target_data.append(q_probs[i, k])
                starting_positions.append(interfaces[i])
                valid_indices.append(i)
                ref_probs.append(diff_ref[i, k])
        
        if target_data:
            # Plot actual probabilities
            ax4.plot(starting_positions, target_data, 'o-', 
                    label=f'Target: {interfaces[k]:.1f}', linewidth=2, markersize=8,
                    color=forward_colors[idx])
            
            # Plot diffusive reference as dashed lines
            ax4.plot(starting_positions, ref_probs, '--', 
                    color=forward_colors[idx], alpha=0.5)
            
            # Add annotations for sample sizes
            for i, pos in enumerate(starting_positions):
                idx = valid_indices[i]
                ax4.annotate(f"n={int(q_weights[idx, k])}", 
                           (pos, target_data[i] + 0.03), 
                           fontsize=7, ha='center')
    
    # Configure the forward plot
    ax4.set_xlabel('Starting Interface Position (λ)')
    ax4.set_ylabel('Probability q(i,k)')
    ax4.set_title('Forward Transition Probabilities (L→R)', fontsize=12)
    ax4.set_ylim(0, 1.05)
    ax4.grid(True, alpha=0.3)
    
    # Create better x-axis ticks for interface positions
    ax4.set_xlim(min(interfaces) - 0.1, interfaces[n_interfaces-2] + 0.1)
    
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
            if not np.isnan(q_probs[i, k]) and q_weights[i, k] > 5:  # Minimum sample threshold
                target_data.append(q_probs[i, k])
                starting_positions.append(interfaces[i])
                valid_indices.append(i)
                ref_probs.append(diff_ref[i, k])
        
        if target_data:
            # Plot actual probabilities
            ax5.plot(starting_positions, target_data, 'o-', 
                    label=f'Target: {interfaces[k]:.1f}', linewidth=2, markersize=8,
                    color=backward_colors[idx])
            
            # Plot diffusive reference as dashed lines
            ax5.plot(starting_positions, ref_probs, '--', 
                    color=backward_colors[idx], alpha=0.5)
            
            # Add annotations for sample sizes
            for i, pos in enumerate(starting_positions):
                idx = valid_indices[i]
                ax5.annotate(f"n={int(q_weights[idx, k])}", 
                           (pos, target_data[i] + 0.03), 
                           fontsize=7, ha='center')
    
    # Configure the backward plot
    ax5.set_xlabel('Starting Interface Position (λ)')
    ax5.set_ylabel('Probability q(i,k)')
    ax5.set_title('Backward Transition Probabilities (R→L)', fontsize=12)
    ax5.set_ylim(0, 1.05)
    ax5.grid(True, alpha=0.3)
    
    # Create better x-axis ticks for interface positions
    ax5.set_xlim(interfaces[1] - 0.1, max(interfaces) + 0.1)
    
    # Add explanatory text about the dashed lines
    ax5.text(0.02, 0.02, ref_text, transform=ax5.transAxes, fontsize=9,
             bbox=dict(facecolor='white', alpha=0.8))
    
    # Add a legend with reasonable size
    ax5.legend(title='Target Interface', loc='best', fontsize=9)
    
    # Plot 2.3: Forward Memory Retention
    ax6 = fig2.add_subplot(gs2[1, 0])
    
    # Calculate memory retention using normalized deviations from diffusive reference
    memory_retention = np.zeros(n_interfaces)
    memory_retention.fill(np.nan)
    
    for k in range(1, n_interfaces):
        # Get all normalized deviations for reaching k from different starting points
        # Exclude i=k (q(i,i)=0) and adjacent interfaces with too few samples
        deviations = []
        for i in range(k-1):  # Only consider i < k-1 (non-adjacent starting points)
            if not np.isnan(q_probs[i, k]) and q_weights[i, k] > 5 and diff_ref[i, k] > 0:
                normalized_dev = (q_probs[i, k] - diff_ref[i, k])/diff_ref[i, k]
                if not np.isnan(normalized_dev):
                    deviations.append(normalized_dev)
        
        if len(deviations) >= 2:  # Need at least 2 points to calculate meaningful spread
            memory_retention[k] = np.std(deviations) * 100  # As percentage
    
    # Plot the memory retention (standard deviation of normalized deviations)
    valid_k = [k for k in range(1, n_interfaces) if not np.isnan(memory_retention[k])]
    valid_retention = [memory_retention[k] for k in valid_k]
    valid_positions = [interfaces[k] for k in valid_k]
    valid_colors = [forward_colors[k-1] for k in valid_k]  # Match colors to targets in forward plot
    
    if valid_k:
        bars6 = ax6.bar(valid_positions, valid_retention, color=valid_colors, alpha=0.7, width=np.min(np.diff(interfaces))*0.7)
        
        # Add annotations for memory retention values
        for i, pos in enumerate(valid_positions):
            ax6.text(pos, valid_retention[i] + 1, f"{valid_retention[i]:.1f}%", 
                   ha='center', fontsize=9)
        
        ax6.set_xlabel('Target Interface Position (λ)')
        ax6.set_ylabel('Variability of Memory Effects (%)')
        ax6.set_title('Forward Memory Retention: Variability in Forward Transitions', fontsize=12)
        ax6.grid(True, axis='y', alpha=0.3)
        ax6.set_ylim(0, max(valid_retention) * 1.2 if valid_retention else 10)
        
        # Add explanatory note
        mem_text = """
        Memory retention measures the spread in memory effects
        from different starting points to the same target interface.
        Higher values indicate stronger history dependence.
        """
        ax6.text(0.02, 0.95, mem_text, transform=ax6.transAxes, fontsize=9,
                 bbox=dict(facecolor='white', alpha=0.8), va='top')
    else:
        ax6.text(0.5, 0.5, "Insufficient data for forward memory retention analysis", 
               ha='center', va='center', transform=ax6.transAxes)
    
    # Plot 2.4: Backward Memory Retention
    ax7 = fig2.add_subplot(gs2[1, 1])
    
    # Calculate backward memory retention
    backward_memory_retention = np.zeros(n_interfaces)
    backward_memory_retention.fill(np.nan)
    
    for k in range(n_interfaces-1):
        # Get all normalized deviations for reaching k from different starting points i>k+1
        deviations = []
        for i in range(k+2, n_interfaces):  # Only consider i > k+1 (non-adjacent starting points)
            if not np.isnan(q_probs[i, k]) and q_weights[i, k] > 5 and diff_ref[i, k] > 0:
                normalized_dev = (q_probs[i, k] - diff_ref[i, k])/diff_ref[i, k]
                if not np.isnan(normalized_dev):
                    deviations.append(normalized_dev)
        
        if len(deviations) >= 2:
            backward_memory_retention[k] = np.std(deviations) * 100  # As percentage
    
    # Plot the backward memory retention
    valid_k = [k for k in range(n_interfaces-1) if not np.isnan(backward_memory_retention[k])]
    valid_retention = [backward_memory_retention[k] for k in valid_k]
    valid_positions = [interfaces[k] for k in valid_k]
    valid_colors = [backward_colors[k] for k in valid_k]  # Match colors to targets in backward plot
    
    if valid_k:
        bars7 = ax7.bar(valid_positions, valid_retention, color=valid_colors, alpha=0.7, width=np.min(np.diff(interfaces))*0.7)
        
        # Add annotations for memory retention values
        for i, pos in enumerate(valid_positions):
            ax7.text(pos, valid_retention[i] + 1, f"{valid_retention[i]:.1f}%", 
                   ha='center', fontsize=9)
        
        ax7.set_xlabel('Target Interface Position (λ)')
        ax7.set_ylabel('Variability of Memory Effects (%)')
        ax7.set_title('Backward Memory Retention: Variability in Backward Transitions', fontsize=12)
        ax7.grid(True, axis='y', alpha=0.3)
        ax7.set_ylim(0, max(valid_retention) * 1.2 if valid_retention else 10)
        
        # Add explanatory note
        ax7.text(0.02, 0.95, mem_text, transform=ax7.transAxes, fontsize=9,
                 bbox=dict(facecolor='white', alpha=0.8), va='top')
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
    ax8.set_yticklabels([f"{interfaces[i]:.1f}" for i in range(n_interfaces-1)])
    
    # Plot 3.2: Memory Decay with Physical Distance
    ax9 = fig3.add_subplot(gs3[0, 1])
    
    # Create colors for starting interfaces
    start_colors = generate_high_contrast_colors(n_interfaces-1)
    
    # Store fitted relaxation times
    relaxation_times = []
    relaxation_errors = []
    starting_points = []
    
    # Fit exponential decay to memory effect vs distance for each starting interface
    for i in range(n_interfaces-1):
        physical_distances = np.array([interfaces[k] - interfaces[i] for k in range(i+1, n_interfaces)])
        values = np.array([abs(q_probs[i, k] - diff_ref[i, k]) for k in range(i+1, n_interfaces)])
        valid_mask = ~np.isnan(values) & (np.array([q_weights[i, k] for k in range(i+1, n_interfaces)]) > 5)
        
        if np.sum(valid_mask) >= 3:  # Need at least 3 points for meaningful fit
            try:
                # Initial parameter guesses (adjusted for physical scaling)
                mean_spacing = np.mean(np.diff(interfaces))
                p0 = [0.2, mean_spacing, 0.05]  # amplitude, tau, offset
                
                # Curve fitting with physical distances
                popt, pcov = curve_fit(exp_decay, physical_distances[valid_mask], values[valid_mask], p0=p0, 
                                      bounds=([0, 0, 0], [1, np.max(interfaces)*2, 0.5]))
                
                # Extract relaxation distance (tau) and its error
                tau = popt[1]
                tau_err = np.sqrt(np.diag(pcov))[1] if np.all(np.isfinite(pcov)) else 0
                
                relaxation_times.append(tau)
                relaxation_errors.append(tau_err)
                starting_points.append(i)
                
                # Plot fitted curve
                x_fit = np.linspace(np.min(physical_distances[valid_mask]), 
                                   np.max(physical_distances[valid_mask]), 100)
                y_fit = exp_decay(x_fit, *popt)
                ax9.plot(x_fit, y_fit, '--', color=start_colors[i], alpha=0.7)
                
                # Plot original data with physical distances
                ax9.plot(physical_distances[valid_mask], values[valid_mask], 'o-', 
                        label=f'Start: {interfaces[i]:.1f}, τ={tau:.2f}±{tau_err:.2f}', 
                        color=start_colors[i])
                
            except RuntimeError as e:
                # If curve_fit fails, just plot the raw data
                ax9.plot(physical_distances[valid_mask], values[valid_mask], 'o-', 
                        label=f'Start: {interfaces[i]:.1f} (fit failed)', color=start_colors[i])
        else:
            # Just plot the raw data if not enough points for fitting
            if np.any(valid_mask):
                ax9.plot(physical_distances[valid_mask], values[valid_mask], 'o-', 
                        label=f'Start: {interfaces[i]:.1f} (insufficient data)', color=start_colors[i])
    
    ax9.set_xlabel('Physical Distance (λ_k - λ_i)')
    ax9.set_ylabel('|Probability - Diffusive Reference|')
    ax9.set_title('Memory Decay Profile with Physical Distance', fontsize=12)
    ax9.grid(True, alpha=0.3)
    ax9.set_ylim(bottom=0)
    ax9.legend(loc='upper right', ncol=1, fontsize=8)
    
    # Inset: Relaxation times vs starting interface position
    if len(relaxation_times) > 1:
        ax9_inset = ax9.inset_axes([0.55, 0.1, 0.4, 0.3])
        ax9_inset.bar([interfaces[i] for i in starting_points], relaxation_times, 
                    yerr=relaxation_errors, color=[start_colors[i] for i in starting_points], capsize=5)
        ax9_inset.set_xlabel('Starting Position λ')
        ax9_inset.set_ylabel('Relaxation Distance')
        ax9_inset.set_title('Memory Relaxation Distance')
        ax9_inset.grid(True, alpha=0.3)
    
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
    ax10.set_yticklabels([f"{interfaces[i+1]:.1f}" for i in range(n_interfaces-1)])
    
    # Plot 3.4: Memory vs Free Energy Barrier
    ax11 = fig3.add_subplot(gs3[1, 1])
    
    # Calculate forward and backward memory effects for adjacent transitions
    forward_memory = np.zeros(n_interfaces-1)
    backward_memory = np.zeros(n_interfaces-1)
    
    # Estimate free energy differences between adjacent interfaces
    delta_G = np.zeros(n_interfaces-1)
    
    for i in range(n_interfaces-1):
        if p[i, i+1] > 0 and p[i+1, i] > 0:
            # Estimate free energy difference from detailed balance
            delta_G[i] = -np.log(p[i, i+1] / p[i+1, i])
            
            # Forward memory effect (i to i+1)
            if not np.isnan(q_probs[i, i+1]) and q_weights[i, i+1] > 5:
                forward_memory[i] = q_probs[i, i+1] - diff_ref[i, i+1]
                
            # Backward memory effect (i+1 to i)
            if not np.isnan(q_probs[i+1, i]) and q_weights[i+1, i] > 5:
                backward_memory[i] = q_probs[i+1, i] - diff_ref[i+1, i]
    
    # Scatter plot of memory effects vs free energy difference
    valid_idx_f = ~np.isnan(forward_memory) & ~np.isnan(delta_G)
    valid_idx_b = ~np.isnan(backward_memory) & ~np.isnan(delta_G)
    
    if np.any(valid_idx_f):
        ax11.scatter(delta_G[valid_idx_f], forward_memory[valid_idx_f], 
                   s=80, marker='o', color='blue', label='Forward Memory')
        
        for i in np.where(valid_idx_f)[0]:
            ax11.annotate(f"{i}→{i+1}", (delta_G[i], forward_memory[i]), 
                        fontsize=8, alpha=0.8)
    
    if np.any(valid_idx_b):
        ax11.scatter(delta_G[valid_idx_b], backward_memory[valid_idx_b], 
                   s=80, marker='s', color='red', label='Backward Memory')
        
        for i in np.where(valid_idx_b)[0]:
            ax11.annotate(f"{i+1}→{i}", (delta_G[i], backward_memory[i]), 
                        fontsize=8, alpha=0.8)
    
    ax11.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax11.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax11.set_xlabel('Free Energy Difference ΔG (kT)')
    ax11.set_ylabel('Memory Effect (q - q_diff)')
    ax11.set_title('Memory Effect vs. Free Energy Barrier', fontsize=12)
    ax11.grid(True, alpha=0.3)
    ax11.legend()
    
    # Add explanatory text about memory and free energy barriers
    barrier_text = """
    Memory Effect vs. Free Energy Barrier:
    • Positive ΔG: Barrier from left to right interface
    • Negative ΔG: Barrier from right to left interface
    • Correlation between memory effects and barrier height
      indicates barrier-dependent transition mechanisms
    """
    ax11.text(0.02, 0.02, barrier_text, transform=ax11.transAxes, fontsize=9,
            bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig3.suptitle('TIS Memory Effect Analysis - Detailed Memory Decay Profiles', fontsize=14)
    
    return fig1, fig2, fig3

def calculate_diffusive_reference(interfaces, q_matrix, q_weights=None, min_samples=5):
    """
    Calculate diffusive reference probabilities for non-equidistant interfaces using
    conditional crossing probabilities (q matrix).
    
    For a memoryless system, the reference probability for transitions is estimated from 
    free energy differences between adjacent interfaces, derived from conditional 
    crossing probabilities using detailed balance principles.
    
    Parameters:
    -----------
    interfaces : list or array
        The positions of the interfaces along the reaction coordinate
    q_matrix : numpy.ndarray
        Matrix of conditional crossing probabilities where:
        - q_matrix[i,k] for i<k: probability that a path starting at i and reaching k-1 will reach k
        - q_matrix[i,k] for i>k: probability that a path starting at i and reaching k+1 will reach k
    q_weights : numpy.ndarray, optional
        Matrix of sample counts for each q_matrix value, used to validate data quality
    min_samples : int, optional
        Minimum number of samples required to consider a q value valid, default is 5
        
    Returns:
    --------
    diff_ref : numpy.ndarray
        A matrix of diffusive reference probabilities for each i,k pair
    """
    n_interfaces = len(interfaces)
    diff_ref = np.zeros((n_interfaces, n_interfaces))
    diff_ref.fill(np.nan)
    
    # Estimate free energy differences between adjacent interfaces
    delta_G = np.zeros((n_interfaces, n_interfaces))
    delta_G.fill(np.nan)
    
    # Calculate free energy differences for adjacent interfaces using q_matrix
    # For memoryless diffusion, q should be related to barrier heights by:
    # q(i,i+1)/(1-q(i,i+1)) = exp(-ΔG_i,i+1)
    # q(i+1,i)/(1-q(i+1,i)) = exp(-ΔG_i+1,i) = exp(ΔG_i,i+1)
    for i in range(n_interfaces-1):
        # Check data validity
        valid_forward = (not np.isnan(q_matrix[i, i+1]) and 
                        (q_weights is None or q_weights[i, i+1] >= min_samples) and
                        q_matrix[i, i+1] > 0 and q_matrix[i, i+1] < 1)
        
        valid_backward = (not np.isnan(q_matrix[i+1, i]) and 
                         (q_weights is None or q_weights[i+1, i] >= min_samples) and
                         q_matrix[i+1, i] > 0 and q_matrix[i+1, i] < 1)
        
        if valid_forward and valid_backward:
            # Calculate free energy difference using both directions and average
            delta_G_forward = -np.log(q_matrix[i, i+1]/(1-q_matrix[i, i+1]))
            delta_G_backward = np.log(q_matrix[i+1, i]/(1-q_matrix[i+1, i]))
            
            # Use weighted average based on sample counts if available
            if q_weights is not None:
                total_weight = q_weights[i, i+1] + q_weights[i+1, i]
                delta_G[i, i+1] = (delta_G_forward * q_weights[i, i+1] + 
                                  delta_G_backward * q_weights[i+1, i]) / total_weight
            else:
                delta_G[i, i+1] = (delta_G_forward + delta_G_backward) / 2
                
            delta_G[i+1, i] = -delta_G[i, i+1]  # By definition
        elif valid_forward:
            # Only forward data is valid
            delta_G[i, i+1] = -np.log(q_matrix[i, i+1]/(1-q_matrix[i, i+1]))
            delta_G[i+1, i] = -delta_G[i, i+1]
        elif valid_backward:
            # Only backward data is valid
            delta_G[i+1, i] = -np.log(q_matrix[i+1, i]/(1-q_matrix[i+1, i]))
            delta_G[i, i+1] = -delta_G[i+1, i]
    
    # Fill in diagonal with zeros (no transition)
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
            if abs(i - k) > 1:  # Non-adjacent interfaces
                if i < k:  # Forward transitions (i → k)
                    # For non-adjacent forward transitions, the diffusive reference is
                    # the product of conditional probabilities along the path:
                    # q_diff(i,k) = Π_{j=i}^{k-1} q_diff(j,j+1)
                    diff_ref[i, k] = 1.0
                    valid_path = True
                    
                    for j in range(i, k):
                        if np.isnan(diff_ref[j, j+1]):
                            valid_path = False
                            break
                        diff_ref[i, k] *= diff_ref[j, j+1]
                    
                    if not valid_path:
                        # Fallback to 0.5 if we can't compute a valid path
                        diff_ref[i, k] = 0.5
                        
                elif i > k:  # Backward transitions (i → k)
                    # Similar to forward, but going backward
                    # q_diff(i,k) = Π_{j=i}^{k+1} q_diff(j,j-1)
                    diff_ref[i, k] = 1.0
                    valid_path = True
                    
                    for j in range(i, k, -1):
                        if np.isnan(diff_ref[j, j-1]):
                            valid_path = False
                            break
                        diff_ref[i, k] *= diff_ref[j, j-1]
                    
                    if not valid_path:
                        # Fallback to 0.5 if we can't compute a valid path
                        diff_ref[i, k] = 0.5
    
    return diff_ref

def estimate_free_energy_differences_from_q(q_matrix, q_weights=None, min_samples=5):
    """
    Estimate free energy differences between interfaces from conditional crossing probabilities.
    
    Parameters:
    -----------
    q_matrix : numpy.ndarray
        Matrix of conditional crossing probabilities
    q_weights : numpy.ndarray, optional
        Matrix of sample counts for each q value
    min_samples : int, optional
        Minimum number of samples required to consider a q value valid
        
    Returns:
    --------
    delta_G : numpy.ndarray
        Matrix of free energy differences between interfaces
    """
    n = q_matrix.shape[0]
    delta_G = np.zeros((n, n))
    delta_G.fill(np.nan)
    
    for i in range(n):
        for j in range(n):
            if abs(i - j) == 1:  # Only adjacent interfaces
                if (not np.isnan(q_matrix[i, j]) and
                    (q_weights is None or q_weights[i, j] >= min_samples) and
                    q_matrix[i, j] > 0 and q_matrix[i, j] < 1):
                    
                    # For adjacent interfaces, the free energy difference is related to
                    # the conditional crossing probability by:
                    # ΔG = -ln(q/(1-q)) if i < j (forward)
                    # ΔG = ln(q/(1-q)) if i > j (backward)
                    if i < j:
                        delta_G[i, j] = -np.log(q_matrix[i, j]/(1-q_matrix[i, j]))
                    else:
                        delta_G[i, j] = np.log(q_matrix[i, j]/(1-q_matrix[i, j]))
    
    return delta_G

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
    pcca = dpt.markov.PCCA(msm, n_clusters)
    pcca.compute()

    # Print the membership matrix
    print("PCCA+ Membership Matrix:")
    print(pcca.memberships)

    # Print the metastable states
    print("Metastable States:")
    for i, state in enumerate(pcca.metastable_sets):
        print(f"State {i+1}: {state}")

    return pcca
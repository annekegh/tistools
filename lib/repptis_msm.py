"""
Functions to construct and analyse an MSM based on the ensembles.

In the paper, N+1 represents the number of interfaces, while in this code N directly 
represents the number of interfaces. This leads to formula differences:
- In the code: NS = 4*N - 5 (where N is the number of interfaces)
- In the paper: NS = 4*(N-1) - 1 = 4*N - 5 (where N+1 is the number of interfaces)

Both formulations ultimately yield the same result, but with different parameter interpretations.

Elias W, May 2025
"""
import numpy as np

#======================================
# PPTIS
#======================================

def construct_tau_vector(N, NS, taumm, taump, taupm, taupp):
    """
    Constructs a flattened vector of transition times for the PPTIS simulation.

    This function takes the transition times for different path types (LML, LMR, RML, RMR)
    and organizes them into a single flattened vector according to the PPTIS ensemble structure.

    Parameters
    ----------
    N : int
        The number of ensembles. Must be at least 3.
    NS : int
        The expected size of the output vector. Must satisfy NS = 4*N - 5.
    taumm : list of float
        Transition times for the LML (Left-to-Left) path type. Must have length N.
    taump : list of float
        Transition times for the LMR (Left-to-Right) path type. Must have length N.
    taupm : list of float
        Transition times for the RML (Right-to-Left) path type. Must have length N.
    taupp : list of float
        Transition times for the RMR (Right-to-Right) path type. Must have length N.

    Returns
    -------
    tau : numpy.ndarray
        A flattened vector of transition times with length NS.

    Raises
    ------
    ValueError
        If any of the input constraints are not met (e.g., invalid lengths or values).
    """
    # Validate input constraints
    if N < 3:
        raise ValueError(f"N must be at least 3, but got {N}.")
    if NS != 4 * N - 5:
        raise ValueError(f"NS must be equal to 4*N - 5 ({4 * N - 5}), but got {NS}.")
    if len(taumm) != N or len(taump) != N or len(taupm) != N or len(taupp) != N:
        raise ValueError(f"All input transition time lists (taumm, taump, taupm, taupp) must have length N={N}.")

    # Initialize the output vector with zeros
    tau = np.zeros(NS)

    # Assign values for the [0-] ensemble
    tau[0] = taupp[0]

    # Assign values for the [0+-] ensemble
    tau[1] = taumm[1]
    tau[2] = taump[1]
    tau[3] = taupm[1]

    # Assign values for intermediate ensembles [1+-], [2+-], ..., [(N-3)+-]
    for i in range(1, N - 2):
        tau[4 * i] = taumm[i + 1]
        tau[4 * i + 1] = taump[i + 1]
        tau[4 * i + 2] = taupm[i + 1]
        tau[4 * i + 3] = taupp[i + 1]

    # Assign values for the [(N-2)^(-1)] ensemble
    tau[-3] = taumm[-1]
    tau[-2] = taump[-1]

    # Assign a placeholder value for the final ensemble (B)
    tau[-1] = 0.0  # This value is arbitrary and can be adjusted as needed

    return tau

def construct_M(p_mm, p_mp, p_pm, p_pp, N):
    """
    Constructs the transition matrix M for the PPTIS simulation.

    This function builds a transition matrix M that describes the probabilities of transitioning
    between states in the PPTIS framework. The matrix is constructed based on the local crossing
    probabilities for different path types (LML, LMR, RML, RMR).

    Parameters
    ----------
    p_mm : list of float
        Local crossing probabilities for the LML (Left-to-Left) path type. Must have length N-1.
    p_mp : list of float
        Local crossing probabilities for the LMR (Left-to-Right) path type. Must have length N-1.
    p_pm : list of float
        Local crossing probabilities for the RML (Right-to-Left) path type. Must have length N-1.
    p_pp : list of float
        Local crossing probabilities for the RMR (Right-to-Right) path type. Must have length N-1.
    N : int
        The number of interfaces. Must be at least 3.

    Returns
    -------
    M : numpy.ndarray
        The transition matrix of shape (NS, NS), where NS = 4*N - 5.

    Raises
    ------
    ValueError
        If any of the input constraints are not met (e.g., invalid lengths or values).
    """

    # Validate input constraints
    if N < 3:
        raise ValueError(f"N must be at least 3, but got {N}.")
    if len(p_mm) != N - 1 or len(p_mp) != N - 1 or len(p_pm) != N - 1 or len(p_pp) != N - 1:
        raise ValueError(f"All input probability lists (p_mm, p_mp, p_pm, p_pp) must have length N-1={N-1}.")

    NS = 4 * N - 5  # Dimension of the transition matrix

    # Handle the special case for N=3
    if N == 3:
        return construct_M_N3(p_mm, p_mp, NS)

    # Initialize the transition matrix with zeros
    M = np.zeros((NS, NS))

    # Fill in transitions for states [0-] and [0+-]
    M[0, 1:4] = [p_mm[0], p_mp[0], 0]  # Transitions from [0-]
    M[1, 0] = 1  # Transition from [0+-] back to [0-]
    M[2, 4:8] = [p_mm[1], p_mp[1], 0, 0]  # Transitions from [0+-]
    M[3, 0] = 1  # Transition from [0+-] back to [0-]
#======================================
# PPTIS
#======================================

def construct_tau_vector(N, NS, taumm, taump, taupm, taupp):
    """
    Constructs a flattened vector of transition times for the PPTIS simulation.

    This function takes the transition times for different path types (LML, LMR, RML, RMR)
    and organizes them into a single flattened vector according to the PPTIS ensemble structure.

    Parameters
    ----------
    N : int
        The number of ensembles. Must be at least 3.
    NS : int
        The expected size of the output vector. Must satisfy NS = 4*N - 5.
    taumm : list of float
        Transition times for the LML (Left-to-Left) path type. Must have length N.
    taump : list of float
        Transition times for the LMR (Left-to-Right) path type. Must have length N.
    taupm : list of float
        Transition times for the RML (Right-to-Left) path type. Must have length N.
    taupp : list of float
        Transition times for the RMR (Right-to-Right) path type. Must have length N.

    Returns
    -------
    tau : numpy.ndarray
        A flattened vector of transition times with length NS.

    Raises
    ------
    ValueError
        If any of the input constraints are not met (e.g., invalid lengths or values).
    """
    # Validate input constraints
    if N < 3:
        raise ValueError(f"N must be at least 3, but got {N}.")
    if NS != 4 * N - 5:
        raise ValueError(f"NS must be equal to 4*N - 5 ({4 * N - 5}), but got {NS}.")
    if len(taumm) != N or len(taump) != N or len(taupm) != N or len(taupp) != N:
        raise ValueError(f"All input transition time lists (taumm, taump, taupm, taupp) must have length N={N}.")

    # Initialize the output vector with zeros
    tau = np.zeros(NS)

    # Assign values for the [0-] ensemble
    tau[0] = taupp[0]

    # Assign values for the [0+-] ensemble
    tau[1] = taumm[1]
    tau[2] = taump[1]
    tau[3] = taupm[1]

    # Assign values for intermediate ensembles [1+-], [2+-], ..., [(N-3)+-]
    for i in range(1, N - 2):
        tau[4 * i] = taumm[i + 1]
        tau[4 * i + 1] = taump[i + 1]
        tau[4 * i + 2] = taupm[i + 1]
        tau[4 * i + 3] = taupp[i + 1]

    # Assign values for the [(N-2)^(-1)] ensemble
    tau[-3] = taumm[-1]
    tau[-2] = taump[-1]

    # Assign a placeholder value for the final ensemble (B)
    tau[-1] = 0.0  # This value is arbitrary and can be adjusted as needed

    return tau

def construct_M(p_mm, p_mp, p_pm, p_pp, N):
    """
    Constructs the transition matrix M for the PPTIS simulation.

    This function builds a transition matrix M that describes the probabilities of transitioning
    between states in the PPTIS framework. The matrix is constructed based on the local crossing
    probabilities for different path types (LML, LMR, RML, RMR).

    Parameters
    ----------
    p_mm : list of float
        Local crossing probabilities for the LML (Left-to-Left) path type. Must have length N-1.
    p_mp : list of float
        Local crossing probabilities for the LMR (Left-to-Right) path type. Must have length N-1.
    p_pm : list of float
        Local crossing probabilities for the RML (Right-to-Left) path type. Must have length N-1.
    p_pp : list of float
        Local crossing probabilities for the RMR (Right-to-Right) path type. Must have length N-1.
    N : int
        The number of interfaces. Must be at least 3.

    Returns
    -------
    M : numpy.ndarray
        The transition matrix of shape (NS, NS), where NS = 4*N - 5.

    Raises
    ------
    ValueError
        If any of the input constraints are not met (e.g., invalid lengths or values).
    """

    # Validate input constraints
    if N < 3:
        raise ValueError(f"N must be at least 3, but got {N}.")
    if len(p_mm) != N - 1 or len(p_mp) != N - 1 or len(p_pm) != N - 1 or len(p_pp) != N - 1:
        raise ValueError(f"All input probability lists (p_mm, p_mp, p_pm, p_pp) must have length N-1={N-1}.")

    NS = 4 * N - 5  # Dimension of the transition matrix

    # Handle the special case for N=3
    if N == 3:
        return construct_M_N3(p_mm, p_mp, NS)

    # Initialize the transition matrix with zeros
    M = np.zeros((NS, NS))

    # Fill in transitions for states [0-] and [0+-]
    M[0, 1:4] = [p_mm[0], p_mp[0], 0]  # Transitions from [0-]
    M[1, 0] = 1  # Transition from [0+-] back to [0-]
    M[2, 4:8] = [p_mm[1], p_mp[1], 0, 0]  # Transitions from [0+-]
    M[3, 0] = 1  # Transition from [0+-] back to [0-]

    # Fill in transitions for states [1+-] (special case)
    M[4, 3] = 1  # Transition from [1+-] back to [0+-]
    M[6, 3] = 1  # Transition from [1+-] back to [0+-]

    # Fill in transitions for states [(N-2)+-] (special case)
    M[(N - 2) * 4, -5:-3] = [p_pm[N - 3], p_pp[N - 3]]  # Transitions from [(N-2)+-]
    M[(N - 2) * 4 + 1, -1] = 1  # Transition from [(N-2)+-] to state B

    # Fill in transition for state B (special case)
    M[-1, 0] = 1  # Transition from state B back to [0-]

    # Fill in transitions for intermediate states [1+-], [2+-], ..., [(N-3)+-]
    for i in range(1, N - 2):
        M[1 + 4 * i, 4 * (i + 1):4 * (i + 1) + 2] = [p_mm[i + 1], p_mp[i + 1]]  # Forward transitions
        M[3 + 4 * i, 4 * (i + 1):4 * (i + 1) + 2] = [p_mm[i + 1], p_mp[i + 1]]  # Forward transitions

    # Fill in transitions for intermediate states [2+-], [3+-], ..., [(N-3)+-]
    for i in range(2, N - 2):
        M[4 * i, 4 * i - 2:4 * i] = [p_pm[i - 1], p_pp[i - 1]]  # Backward transitions
        M[4 * i + 2, 4 * i - 2:4 * i] = [p_pm[i - 1], p_pp[i - 1]]  # Backward transitions

    return M

def construct_M_N3(p_mm, p_mp, NS):
    """Constructs the transition matrix M for the special case of N=3.

    This function builds a transition matrix M for the PPTIS simulation when there are exactly 3 interfaces.
    The matrix describes the probabilities of transitioning between states in the PPTIS framework.

    Parameters
    ----------
    p_mm : list of float
        Local crossing probabilities for the LML (Left-to-Left) path type. Must have length 2.
    p_mp : list of float
        Local crossing probabilities for the LMR (Left-to-Right) path type. Must have length 2.
    NS : int
        The dimension of the transition matrix. Must satisfy NS = 4*N - 5, where N=3.

    Returns
    -------
    M : numpy.ndarray
        The transition matrix of shape (NS, NS).

    Raises
    ------
    ValueError
        If the input probability lists do not have the correct length or if NS is invalid.
    """

    # Validate input constraints
    if len(p_mm) != 2 or len(p_mp) != 2:
        raise ValueError("For N=3, input probability lists (p_mm, p_mp) must have length 2.")
    if NS != 7:  # NS = 4*N - 5 = 7 when N=3
        raise ValueError(f"For N=3, NS must be 7, but got {NS}.")

    # Initialize the transition matrix with zeros
    M = np.zeros((NS, NS))

    # Fill in transitions for states [0-] and [0+-]
    M[0, 1:4] = [p_mm[0], p_mp[0], 0]  # Transitions from [0-]
    M[1, 0] = 1  # Transition from [0+-] back to [0-]
    M[2, 4:6] = [p_mm[1], p_mp[1]]  # Transitions from [0+-] (modified for N=3)
    M[3, 0] = 1  # Transition from [0+-] back to [0-]

    # Fill in transitions for states [1+-] (special case)
    M[4, 3] = 1  # Transition from [1+-] back to [0+-]

    # Fill in transitions for states [(N-2)+-] (special case)
    M[5, -1] = 1  # Transition from [(N-2)+-] to state B

    # Fill in transition for state B (special case)
    M[-1, 0] = 1  # Transition from state B back to [0-]

    return M

#======================================
# Milestoning
#======================================

def construct_M_milestoning(p_min, p_plus, N):
    """Constructs the transition matrix M for a milestoning-based PPTIS simulation.

    This function builds a transition matrix M that describes the probabilities of transitioning
    between states in a milestoning framework. The states include lambda0-, lambda0+, lambda1, ..., lambda(N-1)=B.

    Parameters
    ----------
    p_min : list of float
        Probabilities of transitioning to the previous milestone. Must have length N-1.
    p_plus : list of float
        Probabilities of transitioning to the next milestone. Must have length N-1.
    N : int
        The number of interfaces (milestones). Must be at least 3.

    Returns
    -------
    M : numpy.ndarray
        The transition matrix of shape (NS, NS), where NS = N + 1.

    Raises
    ------
    ValueError
        If N is less than 3 or if the lengths of p_min and p_plus are not N-1.
    """

    # Validate input constraints
    if N < 3:
        raise ValueError(f"N must be at least 3, but got {N}.")
    if len(p_min) != N - 1 or len(p_plus) != N - 1:
        raise ValueError(f"Both p_min and p_plus must have length N-1={N-1}.")

    # NS: Number of states (lambda0-, lambda0+, lambda1, ..., lambda(N-1)=B)
    NS = N + 1

    # Initialize the transition matrix with zeros
    M = np.zeros((NS, NS))

    # Transitions for lambda0- and lambda0+
    M[0, 1] = 1  # Transition from lambda0- to lambda0+
    M[1, 0] = p_min[0]  # Transition from lambda0+ to lambda0-
    M[1, 2] = p_plus[0]  # Transition from lambda0+ to lambda1

    # Transitions for lambda1
    M[2, 0] = p_min[1]  # Transition from lambda1 to lambda0-
    M[2, 3] = p_plus[1]  # Transition from lambda1 to lambda2

    # Transitions for lambda2 to lambda(N-2)
    for i in range(2, N - 1):  # Shift index by 1 due to lambda0 having 2 rows/cols
        M[i + 1, i - 1 + 1] = p_min[i]  # Transition to the previous milestone
        M[i + 1, i + 1 + 1] = p_plus[i]  # Transition to the next milestone

    # Transitions for state B (lambda(N-1))
    M[N, 0] = 1  # Transition from B back to lambda0-

    return M

#======================================
# crossing probabilities
#======================================

def global_pcross_msm(M, doprint=False):
    """
    Compute global crossing probabilities in a Markov process.

    This function calculates the probability of reaching state -1 before state 0 
    under different conditions, using a transition matrix.

    Parameters
    ----------
    M : np.ndarray
        A square transition matrix representing the Markov process. Must have at least 3 states.
    doprint : bool, optional
        If True, prints intermediate computation details. Default is False.

    Returns
    -------
    z1 : np.ndarray
        A 2-element array containing crossing probabilities for states 0 and -1.
    z2 : np.ndarray
        An (NS-2)-element array containing crossing probabilities for other states.
    y1 : np.ndarray
        A 2-element array with adjusted crossing probabilities for states 0 and -1.
    y2 : np.ndarray
        An (NS-2)-element array with adjusted crossing probabilities for other states.

    Notes
    -----
    - `y1[0]` gives the global crossing probability from state 0 to -1, given that 
      the process starts at 0 and leaves it.
    - The function solves a system of linear equations rather than directly inverting matrices 
      for better numerical stability.
    """
    NS = len(M)
    if NS < 3:
        raise ValueError(f"Transition matrix must have at least 3 states, but got {NS}.")

    # Extract the core transition matrix (excluding boundary states 0 and -1)
    Mp = M[1:-1, 1:-1]  
    a = np.identity(NS-2) - Mp  # Compute (I - Mp) for solving equations

    # Extract transition probabilities related to states 0 and -1
    D = M[1:-1, np.array([0, -1])]  # Transitions from intermediate states to 0 or -1
    E = M[np.array([0, -1]), 1:-1]  # Transitions from 0 or -1 to intermediate states
    M11 = M[np.array([0, -1]), np.array([0, -1])]  # Transitions within states 0 and -1

    # Solve for Z vector (probability of reaching -1 before 0)
    z1 = np.array([[0], [1]])  # Boundary conditions: 0 for state 0, 1 for state -1
    z2 = np.linalg.solve(a, np.dot(D, z1))  # Solve (I - Mp) z2 = D z1

    # Compute Y vector (adjusted probability given that the process leaves its current state)
    y1 = np.dot(M11, z1) + np.dot(E, z2)  # y1[0] = probability of reaching -1 given leaving 0
    y2 = np.dot(D, z1) + np.dot(Mp, z2)   # Adjusted crossing probabilities for other states

    if doprint:
        print("Eigenvalues of Mp:")
        print(np.linalg.eigvals(Mp))
        print("Eigenvalues of (I - Mp):")
        print(np.linalg.eigvals(a))
        print("Transition matrix components:")
        print("D:\n", D)
        print("E:\n", E)
        print("M11:\n", M11)
        print("Vectors:")
        print("z1:\n", z1)
        print("z2:\n", z2)
        print("y1:\n", y1)
        print("y2:\n", y2)
        print("Verification (should be close to 0):", np.sum((y2 - z2) ** 2))

    return z1, z2, y1, y2


#======================================
# Mean first passage times
#======================================

def mfpt_to_absorbing_states(M, tau1, taum, tau2, absor, kept, doprint=False, remove_initial_m=True):
    """
    Compute the mean first passage time (MFPT) to absorbing states.

    This function calculates the MFPT to reach any of the specified absorbing states 
    in a Markov process, both unconditionally (G) and conditionally on leaving the 
    current state (H).

    Parameters
    ----------
    M : np.ndarray
        The transition matrix of the Markov process.
    tau1 : np.ndarray
        Time before the first visit to an absorbing state.
    taum : np.ndarray
        Time spent between the first and last visit to an absorbing state.
    tau2 : np.ndarray
        Time after the last visit to an absorbing state.
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
    - The function avoids explicit matrix inversion for numerical stability.
    - For absorbing states, g1 is initialized to zero (boundary condition).
    - For nonboundary states, g2 is found by solving the system (I - Mp) g2 = D g1 + tp,
      where tp is the average time from nonboundary states.
    - If remove_initial_m is enabled, the middle time component (taum) is subtracted
      from the conditional MFPT calculations.
    """
    taum2 = taum + tau2  # Total time excluding the initial phase

    NS = len(M)
    if NS < 3:
        raise ValueError(f"Transition matrix must have at least 3 states, but got {NS}.")

    check_valid_indices(M, absor, kept)

    if len(M) != len(absor) + len(kept):
        raise ValueError("The number of states must be the sum of absorbing and nonboundary states.")

    # Extract the submatrix for nonboundary states
    Mp = np.take(np.take(M, kept, axis=0), kept, axis=1)

    # Extract transition probabilities between nonboundary and absorbing states
    D = np.take(np.take(M, kept, axis=0), absor, axis=1)  # Transitions from nonboundary to absorbing states
    E = np.take(np.take(M, absor, axis=0), kept, axis=1)  # Transitions from absorbing to nonboundary states
    M11 = np.take(np.take(M, absor, axis=0), absor, axis=1)  # Transitions within absorbing states

    a = np.identity(len(Mp)) - Mp  # Compute (I - Mp) for solving equations

    # Construct time vectors
    # (m2) is the sum of the middle part (m) and the end part (2)
    t1 = taum2[absor].reshape(len(absor), 1)  # Average time (m2) from absorbing states
    tp = taum2[kept].reshape(len(kept), 1)  # Average time (m2) from nonboundary states
    st1 = taum[absor].reshape(len(absor), 1)  # Average time of middle part (m) of the absorbing state
    stp = taum[kept].reshape(len(kept), 1)  # Average time of middle part (m) for nonboundary states

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

    return g1, g2, h1, h2


def mfpt_to_first_last_state(M, tau1, taum, tau2, doprint=False):
    """
    Compute the mean first passage time (MFPT) to reach either state 0 or state -1.

    This function calculates the MFPT to reach the first (0) or last (NS-1) state 
    in a Markov process. It does so by treating these two states as absorbing 
    and solving for the expected first passage times.

    Parameters
    ----------
    M : np.ndarray
        The transition matrix of the Markov process.
    tau1 : np.ndarray
        Time before the first visit to an absorbing state.
    taum : np.ndarray
        Time spent between the first and last visit to an absorbing state.
    tau2 : np.ndarray
        Time after the last visit to an absorbing state.
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
    - Calls `mfpt_to_absorbing_states` with `remove="m"` to exclude the intermediate 
      passage time contribution from the calculations.
    """
    NS = len(M)
    if NS < 3:
        raise ValueError(f"Transition matrix must have at least 3 states, but got {NS}.")

    absor = np.array([0, NS - 1])
    kept = np.array([i for i in range(NS) if i not in absor])

    return mfpt_to_absorbing_states(M, tau1, taum, tau2, absor, kept, doprint=doprint, remove_initial_m="m")


#======================================
# help functions
#======================================

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

def create_labels_states(N):
    """
    Generate labels for absorbing and non-absorbing states.

    This function creates two separate lists of state labels:
    - `labels1`: Labels for the two absorbing states (`0-` and `B`).
    - `labels2`: Labels for the `N-2` non-absorbing states.

    Parameters
    ----------
    N : int
        The total number of states. Must be at least 3.

    Raises
    ------
    ValueError
        If `N` is less than 3.

    Returns
    -------
    labels1 : list of str
        Labels for the two absorbing states.
    labels2 : list of str
        Labels for the `N-2` non-absorbing states.
    """
    if N < 3:
        raise ValueError(f"Expected N >= 3, but got {N}")

    labels1 = ["0-     ", "B      "]
    labels2 = ["0+- LML", "0+- LMR", "0+- RML", "1+- LML", "1+- LMR"]

    if N > 3:
        for i in range(1, N - 2):
            labels2.extend([
                f"{i}+- RML",
                f"{i}+- RMR",
                f"{i+1}+- LML",
                f"{i+1}+- LMR"
            ])

    return labels1, labels2


def create_labels_states_all(N):
    """
    Generate labels for all states, including absorbing and non-absorbing states.

    This function creates a single list of state labels, including:
    - The absorbing state `0-`
    - All non-absorbing states
    - The absorbing state `B`

    Parameters
    ----------
    N : int
        The total number of states. Must be at least 3.

    Raises
    ------
    ValueError
        If `N` is less than 3.

    Returns
    -------
    labels : list of str
        Labels for all `N` states in sequential order.
    """
    if N < 3:
        raise ValueError(f"Expected N >= 3, but got {N}")

    labels = ["0-     ", "0+- LML", "0+- LMR", "0+- RML", "1+- LML", "1+- LMR"]

    if N > 3:
        for i in range(1, N - 2):
            labels.extend([
                f"{i}+- RML",
                f"{i}+- RMR",
                f"{i+1}+- LML",
                f"{i+1}+- LMR"
            ])

    labels.append("B      ")

    return labels

def print_vector(g, states=None, sel=None):
    """
    Print a vector `g` with corresponding state labels.

    Parameters
    ----------
    g : array-like
        The vector to print.
    states : list of str, optional
        Labels corresponding to each state. If `None`, indices are used instead.
    sel : list of int, optional
        Indices of selected states to print. If `None`, all states are printed.

    Raises
    ------
    ValueError
        If `sel` is provided but does not match the length of `g`.
    """
    if sel is not None and len(g) != len(sel):
        raise ValueError("Length of `g` must match length of `sel` if `sel` is provided.")

    for i in range(len(g)):
        state_label = f"state {states[sel[i]]}" if states and sel else f"state {states[i]}" if states else f"state {i}"
        print(f"{state_label}: {g[i][0] if isinstance(g[i], (list, tuple, np.ndarray)) else g[i]}")


def print_all_tau(pathensembles, taumm, taump, taupm, taupp):
    """
    Print all tau values for each path ensemble.

    Parameters
    ----------
    pathensembles : list
        List of path ensemble objects, each having a `.name` attribute.
    taumm : array-like
        First tau metric for each path.
    taump : array-like
        Second tau metric for each path.
    taupm : array-like
        Third tau metric for each path.
    taupp : array-like
        Fourth tau metric for each path.

    Raises
    ------
    ValueError
        If the input arrays do not have the same length as `pathensembles`.
    """
    num_paths = len(pathensembles)
    if not all(len(arr) == num_paths for arr in [taumm, taump, taupm, taupp]):
        raise ValueError("All tau arrays must have the same length as `pathensembles`.")

    print(f"{'Index':<5} {'Name':<5} {'mm':>12} {'mp':>12} {'pm':>12} {'pp':>12}")
    print("-" * 53)

    for i in range(num_paths):
        name_suffix = pathensembles[i].name[-3:] if hasattr(pathensembles[i], 'name') else "N/A"
        print(f"{i:<5} {name_suffix:<5} {taumm[i]:12.1f} {taump[i]:12.1f} {taupm[i]:12.1f} {taupp[i]:12.1f}")

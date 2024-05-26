# functions to construct and analyse an MSM based on the ensembles
import numpy as np

#======================================
# PPTIS
#======================================

def construct_M(p_mm, p_mp, p_pm, p_pp, N):
    """Construct transition matrix M"""
    # N -- number of interfaces
    # NS -- dimension of MSM, 4*N-5 when N>=3
    # p_mp -- list shape (N-1,) with local crossing probability minus-to-plus (mp)
    
    assert N>=3
    assert N==len(p_mm)+1
    assert N==len(p_mp)+1
    assert N==len(p_pm)+1
    assert N==len(p_pp)+1
    NS = 4*N-5

    if N==3:
        return construct_M_N3(p_mm, p_mp, p_pm, p_pp, NS)

    # construct transition matrix
    M = np.zeros((NS,NS))
    
    # states [0-] and [0+-]
    M[0,1:4] = [p_mm[0],p_mp[0],0]
    M[1,0] = 1
    M[2,4:8] = [p_mm[1],p_mp[1],0,0]
    M[3,0] = 1

    # states [1+-] special
    M[4,3] = 1
    M[6,3] = 1

    # states [(N-2)+-] special
    M[(N-2)*4,-5:-3] = [p_pm[N-3],p_pp[N-3]]
    M[(N-2)*4+1,-1] = 1

    # state B=N-1 special
    M[-1,0] = 1

    for i in range(1,N-2):
        #print("starting from state i",i)
        M[1+4*i,4*(i+1):4*(i+1)+2] = [p_mm[i+1],p_mp[i+1]]
        M[3+4*i,4*(i+1):4*(i+1)+2] = [p_mm[i+1],p_mp[i+1]]

    for i in range(2,N-2):
        #print("starting from state i",i)
        M[4*i,  4*i-2:4*i] = [p_pm[i-1],p_pp[i-1]]
        M[4*i+2,4*i-2:4*i] = [p_pm[i-1],p_pp[i-1]]

    return M

def construct_M_N3(p_mm, p_mp, p_pm, p_pp, NS):
    """Construct transition matrix M for N=3"""

    # construct transition matrix
    M = np.zeros((NS,NS))

    # states [0-] and [0+-]
    M[0,1:4] = [p_mm[0],p_mp[0],0]
    M[1,0] = 1
    M[2,4:6] = [p_mm[1],p_mp[1]]   # changed
    M[3,0] = 1

    # states [1+-] special
    M[4,3] = 1
    # states [(N-2)+-] special
    M[5,-1] = 1
    # state B=N-1 special
    M[-1,0] = 1

    return M

#======================================
# Milestoning
#======================================

def construct_M_milestoning(p_min, p_plus, N):
    """Construct transition matrix M

    N -- number of interfaces: 0,..,N-1
    NS -- dimension of MSM, N when N>=3
    p_mp -- list shape (N-1,) with local crossing probability minus-to-plus (mp)
    
    States:
    0-, 0+, 1, .., N-1=B => this is N+1=NS

    Sampled ensembles:
    0+, 1, 2, .., N-2    => this is N-1  
    """
    assert N>=3
    assert N==len(p_min)+1
    assert N==len(p_plus)+1
    
    # NS: 0 to 0, 0 to 1, 1 to 2, ..., N-2 to N-1 (=B)
    NS = N+1

    # construct transition matrix
    M = np.zeros((NS,NS))
    
    # states = lambda0-, lambda0+, lambda1, lambda2, ...
    # lambda0 special
    M[0,1] = 1
    M[1,0] = p_min[0]
    M[1,2] = p_plus[0]
    # lambda1 special
    M[2,0] = p_min[1]
    M[2,3] = p_plus[1]
    
    # lambda2...lambda(N-2)
    for i in range(2,N-1): # shift index by 1, because lambda0 has 2 rows/cols
        M[i+1,i-1+1] = p_min[i]
        M[i+1,i+1+1] = p_plus[i]
    
    # state B, lambda N-1 special
    M[N,0] = 1

    return M

#======================================
# crossing probabilities
#======================================

def global_cross_prob(M, doprint=False):
    """Probabilities to arrive in -1 before 0, various conditions

    Z = probability to arrive in -1 before 0
    Y = probability to arrive in -1 before 0, given that you are leaving your current state

    Arguments:
    M -- transition matrix

    Optional:
    doprint -- print intermediate results (default: False)

    Returns:
    z1 -- array with 2 elements for states 0 and -1
    z2 -- array with NS-2 elements for other states
    y1 -- array with 2 elements for states 0 and -1
    y2 -- array with NS-2 elements for other states
    The interesting answer is y1[0] = global crossing probability from 0 to -1
    given that you are at 0 now and that you are leaving 0
    """
    NS = len(M)
    assert NS>=3

    # take pieces of transition matrix
    Mp = M[1:-1,1:-1]
    a = np.identity(NS-2)-Mp    # 1-Mp
    a1 = np.linalg.inv(a)       # (1-Mp)^(-1)

    # other pieces
    D = M[1:-1, np.array([0,-1])]
    E = M[np.array([0,-1]), 1:-1]
    M11 = M[np.array([0,-1]),np.array([0,-1])]

    # compute Z vector
    z1 = np.array([[0],[1]])
    z2 = np.dot(a1,np.dot(D,z1))

    # compute Y vector
    y1 = np.dot(M11,z1) + np.dot(E,z2)  # y1[0] = Prcross
    y2 = np.dot(D,z1) + np.dot(Mp,z2)

    if doprint:
        print("Mp eigenvals")
        vals, vecs = np.linalg.eig(Mp)
        print(vals)
        print("1-Mp eigenvals")
        vals, vecs = np.linalg.eig(a)
        print(vals)
        #print(np.dot(a,a1)  # identity matrix indeed
        print("other pieces M")
        print(D)
        print(E)
        print(M11)
        print("vector z1,z2")
        print(z1)
        print(z2)
        print("vector y1,y2")
        print(y1)
        print(y2)
        print("check", np.sum((y2-z2)**2))  # 0, so z2 and y2 indeed the same
    return z1, z2, y1, y2


#======================================
# Mean first passage times
#======================================

def mfpt_to_absorbing_states(M, tau1, taum, tau2, absor, kept, doprint=False,
                              remove="m"):
    """Compute MFPT to arrive in any of the absorbing states, indices absor

    G = MFPT to arrive in absor states
    H = MFPT to arrive in absor states, given that you are leaving your current state
    
    Time is measured in number of steps. Endpoints are not included.
    Only parts (m) and (2) are included in the time.

    Arguments:
    M -- transition matrix
    tau1 -- array with time before first hit of M
    taum -- array with time between hitting M first and last
    tau2 -- array with time after last hit of M
    absor -- array with indices of absorbing states
    kept -- array with indices of non-absorbing states

    Optional:
    doprint -- print intermediate results (default: False)
    remove -- whether to remove some time from the starting state in H
              (default: "m", remove middle part m of starting state)

    Returns:
    g1 -- array with n_absorb elements
    g2 -- array with n_kept elements
    h1 -- array with n_absorb elements
    h2 -- array with n_kept elements
    """
    taum2 = taum + tau2
    NS = len(M)
    assert NS>=3
    check_valid_indices(M, absor, kept)
    
    # n_absorb + n_kept = n_states
    assert len(M) == len(absor)+len(kept)

    # take pieces of transition matrix    
    Mp  = np.take(np.take(M, kept, axis=0), kept, axis=1)
    # other pieces
    D   = np.take(np.take(M, kept, axis=0), absor, axis=1)
    E   = np.take(np.take(M, absor, axis=0), kept, axis=1)
    M11 = np.take(np.take(M, absor, axis=0), absor, axis=1)

    a = np.identity(len(Mp))-Mp   # 1-Mp
    a1 = np.linalg.inv(a)         # compute (1-Mp)^(-1)

    # part tau (m2)
    t1 = taum2[absor].reshape(len(absor),1)
    tp = taum2[kept].reshape(len(kept),1)
    # part tau (m). the middle part
    st1 = taum[absor].reshape(len(absor),1)
    stp = taum[kept].reshape(len(kept),1)

    # compute G vector: DESIGN!
    g1 = np.zeros((len(absor),1))  # + t1 # t1 is set to zero
    g2 = np.dot(a1, np.dot(D,g1) + tp)

    # compute H vector
    h1 = np.dot(M11,g1) + np.dot(E,g2) + t1
    h2 = np.dot(D,g1) + np.dot(Mp,g2)  + tp

    # EXTRA: remove middle part m of first time
    #-------------------------------------------
    # TODO write documentation here
    if remove=="m":
        h1 -= st1
        h2 -= stp

    if doprint:
        print("Mp eigenvals")
        vals, vecs = np.linalg.eig(Mp)
        print(vals)
        print("1-Mp eigenvals")
        vals, vecs = np.linalg.eig(a)
        print(vals)
        #print(np.dot(a,a1)  # identity matrix indeed
        print("other pieces M")
        print(D)
        print(E)
        print(M11)
        print("vector tau m2")
        print(t1)
        print(tp)
        print("vector g1,g2")
        print(g1)
        print(g2)
        print("vector h1,h2")
        print(h1)
        print(h2)
        print("check", np.sum((g2-h2)**2))  # 0, so g2 and h2 indeed the same
    return g1, g2, h1, h2


def mfpt_to_first_last_state(M, tau1, taum, tau2, doprint=False):
    """Compute MFPT to arrive in 0 or -1

    The interesting answer is h1[0] = MFPT to absor states 0 or -1
    given that you are at 0 now and that you are leaving 0
    """
    NS = len(M)
    assert NS>=3

    absor = np.array([0,NS-1])
    kept  = np.array([i for i in range(NS) if i not in absor])

    g1, g2, h1, h2 = mfpt_to_absorbing_states(M, tau1, taum, tau2, absor, kept, 
                                              doprint=doprint, remove="m")
    return g1, g2, h1, h2


#======================================
# help functions
#======================================

def check_valid_indices(M, absor, kept):
    NS = len(M)
    assert len(set(absor))==len(absor)   # no accidental duplicates
    assert min(absor)>=0     # valid indices
    assert max(absor)<NS

    assert len(set(kept))==len(kept)
    assert min(kept)>=0
    assert max(kept)<NS
    
    # at least once somewhere but not twice
    for i in range(NS):
        assert (i in absor) != (i in kept)

def get_pieces_matrix(M, absor, kept):
    """take blocks of a square matrix, corresponding to absor and kept states"""
    check_valid_indices(M, absor, kept)
    
    # take pieces of transition matrix
    Mp  = np.take(np.take(M, kept, axis=0), kept, axis=1)
    # other pieces
    D   = np.take(np.take(M, kept, axis=0), absor, axis=1)
    E   = np.take(np.take(M, absor, axis=0), kept, axis=1)
    M11 = np.take(np.take(M, absor, axis=0), absor, axis=1)
    #print(D.shape, E.shape, M11.shape, Mp.shape, absor.shape, kept.shape)

    # TODO
    if len(absor)==1:
        print("D.shape, E.shape, M11.shape, Mp.shape, absor.shape, kept.shape")
        print(D.shape, E.shape, M11.shape, Mp.shape, absor.shape, kept.shape)
        raise Error #reshape all

    return Mp, D, E, M11

def get_pieces_vector(vec, absor, kept):
    """take blocks of a vector, corresponding to absor and kept states"""
    check_valid_indices(M, absor, kept)

    # take pieces of vector
    v1 = vec[absor].reshape(len(absor),1)
    v2 = vec[kept].reshape(len(kept),1)
    return v1, v2

def create_labels_states(N):
    """create list of labels of all states
    Returns:
    labels1 -- list with labels of 2 absorbing states 0- and B
    labels2 -- list with labels of NS-2 non-absorbing states"""
    assert N>=3
    labels1 = ["0-     ","B      "]
    labels2 = ["0+- LML","0+- LMR","0+- RML","1+- LML","1+- LMR"]
    if N>3:
        for i in range(1,N-2):
            labels2.append(str(i)  +"+- RML")
            labels2.append(str(i)  +"+- RMR")
            labels2.append(str(i+1)+"+- LML")
            labels2.append(str(i+1)+"+- LMR")
    return labels1, labels2

def create_labels_states_all(N):
    """create list of labels of all states"""
    assert N>=3
    labels = ["0-     ","0+- LML","0+- LMR","0+- RML",
        "1+- LML","1+- LMR"]
    if N>3:
        for i in range(1,N-2):
            labels.append(str(i)  +"+- RML")
            labels.append(str(i)  +"+- RMR")
            labels.append(str(i+1)+"+- LML")
            labels.append(str(i+1)+"+- LMR")
    labels.append("B      ")
    return labels


def print_vector(g, states=None, sel=None):
    """print the vector g nicely with the corresponding states

    states -- list with labels of all the states
    sel -- list with indices, select only certain states
    """
    if sel is not None: assert len(g)==len(sel)

    for i in range(len(g)):
        if states is None:
            print("state", i, g[i])
        else:
            if sel is None:
                print("state", states[i], g[i][0])
            else:
                print("state", states[sel[i]], g[i][0])

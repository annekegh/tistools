import numpy as np
import logging
from .reading import set_flags_ACC_REJ

# Hard-coded rejection flags found in output files
ACCFLAGS,REJFLAGS = set_flags_ACC_REJ() 

# logger
logger = logging.getLogger(__name__)

def construct_tau_vector(N, NS, taumm, taump, taupm, taupp):
    assert N>=3
    assert NS==4*N-5
    assert len(taumm) == N
    assert len(taump) == N
    assert len(taupm) == N
    assert len(taupp) == N
    # unravel the values into one vector
    tau = np.zeros(NS)
    # [0-]
    tau[0] = taupp[0]
    # [0+-]
    tau[1] = taumm[1]
    tau[2] = taump[1]
    tau[3] = taupm[1]
    # [1+-] etc
    for i in range(1,N-2):
        tau[4*i]   = taumm[i+1]
        tau[4*i+1] = taump[i+1]
        tau[4*i+2] = taupm[i+1]
        tau[4*i+3] = taupp[i+1]
    # [(N-2)^(-1)]
    tau[-3] = taumm[-1]
    tau[-2] = taump[-1]
    # B
    tau[-1] = 0.   # whatever
    return tau


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
    # a1 = np.linalg.inv(a)       # (1-Mp)^(-1)  --> bad practice!

    # other pieces
    D = M[1:-1, np.array([0,-1])]
    E = M[np.array([0,-1]), 1:-1]
    M11 = M[np.array([0,-1]),np.array([0,-1])]

    # compute Z vector
    z1 = np.array([[0],[1]])
    # z2 = np.dot(a1,np.dot(D,z1))
    z2 = np.linalg.solve(a, np.dot(D,z1))

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
    # a1 = np.linalg.inv(a)         # compute (1-Mp)^(-1)

    # part tau (m2)
    t1 = taum2[absor].reshape(len(absor),1)
    tp = taum2[kept].reshape(len(kept),1)
    # part tau (m). the middle part
    st1 = taum[absor].reshape(len(absor),1)
    stp = taum[kept].reshape(len(kept),1)

    # compute G vector: DESIGN!
    g1 = np.zeros((len(absor),1))  # + t1 # t1 is set to zero
    # g2 = np.dot(a1, np.dot(D,g1) + tp)
    g2 = np.linalg.solve(a, np.dot(D,g1) + tp)

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

def print_all_tau(pathensembles, taumm, taump, taupm, taupp):
    # print all tau
    print(f"                  mm            mp            pm            pp")
    for i in range(len(pathensembles)):
        print(f"{i} {pathensembles[i].name[-3:]}  {taumm[i]:13.1f} {taump[i]:13.1f} {taupm[i]:13.1f} {taupp[i]:13.1f}")
def set_tau_first_hit_M_distrib(pe, do_last = True):
    """Set, for each pathtype, the average pathlength before the middle 
    interface is crossed. The phasepoint at the beginning, and right after 
    the crossing will still be included.
    
    Parameters
    ----------
    pe : object like :py:class:`.PathEnsemble`
        Tistools PathEnsemble object must be from a PPTIS simulation,
        for which the weights and the orders have been set.

    do_last: whether tau2 is computed (default: True)
        
    Returns
    -------
    Nothing, but sets the attribute pe.tau1 and pe.tau1avg.

    """
    pe.tau1 = []
    pe.tau1avg = {"LML": None, "LMR": None, "RML": None, "RMR": None}
    # select the accepted paths
    for i in range(len(pe.cyclenumbers)):
        if pe.flags[i] != "ACC" or pe.generation[i] == "ld":
            pe.tau1.append(0)
            continue 
        pe.tau1.append(get_tau1_path(pe.orders[i], pe.lmrs[i], pe.interfaces[0]))
    pe.tau1 = np.array(pe.tau1) 

    # get the average tau1 for each path type. Each path has a weight w.
    for ptype in ("LML", "LMR", "RML", "RMR"):
        totweight = np.sum(pe.weights[pe.lmrs == ptype])
        if totweight != 0:   # make sure total weight is not zero
            pe.tau1avg[ptype] = np.average(pe.tau1[pe.lmrs == ptype], 
                                       weights=pe.weights[pe.lmrs == ptype])

    if not do_last:
        return

    pe.tau2 = []
    pe.tau2avg = {"LML": None, "LMR": None, "RML": None, "RMR": None}
    # select the accepted paths
    for i in range(len(pe.cyclenumbers)):
        if pe.flags[i] != "ACC" or pe.generation[i] == "ld":
            pe.tau2.append(0)
            continue
        pe.tau2.append(get_tau2_path(pe.orders[i], pe.lmrs[i], pe.interfaces[0]))
    pe.tau2 = np.array(pe.tau2)

    # get the average tau2 for each path type. Each path has a weight w.
    for ptype in ("LML", "LMR", "RML", "RMR"):
        totweight = np.sum(pe.weights[pe.lmrs == ptype])
        if totweight != 0:   # make sure total weight is not zero
            pe.tau2avg[ptype] = np.average(pe.tau2[pe.lmrs == ptype],
                                       weights=pe.weights[pe.lmrs == ptype])


def get_tau1_path(orders, ptype, intfs):
    """Return the number of steps it took for this path to cross the M interface
        
    get 1st index after starting point = a    # usually a=1
    get 1st index after crossing M     = b
    return b-a = the number of phase points in the zone

    Example:
    interface     L              M           R
    phasepoint  x | x    x  x  x | x  x    x | x
    index       0   1    2  3  4   5  6    7   8
                    a              b
    then b-a = 5-1 = 4 phasepoints in the zone
    """
    if ptype in ("LMR", "LML"):
        a = np.where(orders[:,0] >= intfs[0])[0][0]  # L->M->. cross
        b = np.where(orders[:,0] >= intfs[1])[0][0]  # L->M->. cross
        return b-a
    elif ptype in ("RML", "RMR"):
        a = np.where(orders[:,0] <= intfs[2])[0][0]  # .<-M<-R cross
        b = np.where(orders[:,0] <= intfs[1])[0][0]  # .<-M<-R cross
        return b-a
    else:
        raise ValueError(f"Unknown path type {ptype}")

def get_tau2_path(orders, ptype, intfs):
    """Return the number of steps in the path after the last crossing of M interface"""
    #very similar to get_tau1_path
    # "a" serves to cut off a piece, usually a=1
    if ptype in ("LML", "RML"):
        a = np.where(orders[::-1,0] >= intfs[0])[0][0]  # L<-M<-. cross
        b = np.where(orders[::-1,0] >= intfs[1])[0][0]  # L<-M<-. cross
        return b-a
    elif ptype in ("LMR", "RMR"):
        a = np.where(orders[::-1,0] <= intfs[2])[0][0]  # .->M->R cross
        b = np.where(orders[::-1,0] <= intfs[1])[0][0]  # .->M->R cross
        return b-a
    else:
        raise ValueError(f"Unknown path type {ptype}")

def get_tau_path(orders, ptype, intfs):
    # cut off piece at start
    if ptype in ("LMR", "LML"):
        a1 = np.where(orders[:,0] >= intfs[0])[0][0]  # L->M->. cross
    elif ptype in ("RML", "RMR"):
        a1 = np.where(orders[:,0] <= intfs[2])[0][0]  # .<-M<-R cross
    else:
        raise ValueError(f"Unknown path type {ptype}")
    # cut off piece at end
    if ptype in ("LML", "RML"):
        a2 = np.where(orders[::-1,0] >= intfs[0])[0][0]  # L<-M<-. cross
    elif ptype in ("LMR", "RMR"):
        a2 = np.where(orders[::-1,0] <= intfs[2])[0][0]  # .->M->R cross
    else:
        raise ValueError(f"Unknown path type {ptype}")
    b = len(orders)       # len(pe.orders[i]) = path length of path i
    return b-a1-a2
    

def set_tau_distrib(pe):
    """Set, for each pathtype, the average total pathlength. The phasepoint at
    the beginning, and right after the crossing will still be included.
    
    Parameters
    ----------
    pe : object like :py:class:`.PathEnsemble`
        Tistools PathEnsemble object must be from a PPTIS simulation,
        for which the weights and the orders have been set.
        
    Returns
    -------
    Nothing, but sets the attribute pe.tau and pe.tauavg.
    """
    pe.tau = []
    pe.tauavg = {"LML": None, "LMR": None, "RML": None, "RMR": None,
                 "L*L": None, "R*R": None}
    # determine pathtypes
    if pe.in_zero_minus:
        if pe.has_zero_minus_one:
            ptypes = ["LML", "LMR", "RML", "RMR", "L*L", "R*R"]
        else:
            ptypes = ["RMR",]
    else:
            ptypes = ["LML", "LMR", "RML", "RMR",]
    #pe.tauavg = {}
    for ptype in ptypes:
        pe.tauavg[ptype] = None

    # select the accepted paths and collect path lengths
    for i in range(len(pe.cyclenumbers)):
        if pe.flags[i] != "ACC" or pe.generation[i] == "ld":
            pe.tau.append(0)
            continue 
        pe.tau.append(get_tau_path(pe.orders[i], pe.lmrs[i], pe.interfaces[0]))
        # pe.orders[i] contains the order parameters of the i-th path
        # including the start/end point
    pe.tau = np.array(pe.tau)

    # get the average tau for each path type. Each path has a weight w.
    for ptype in ptypes:
        totweight = np.sum(pe.weights[pe.lmrs == ptype])
        if totweight != 0:   # make sure total weight is not zero
            pe.tauavg[ptype] = np.average(pe.tau[pe.lmrs == ptype], 
                                      weights=pe.weights[pe.lmrs == ptype])


# COLLECTING
# collect_tau
# collect_tau1
# collect_tau2
# collect_taum

def collect_tau(pathensembles):
    """Compute average path lengths"""
    
    # pathensembles -- list of pathensemble instances
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
    # TODO pieces missing [0-]   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    return taumm, taump, taupm, taupp


def collect_tau1(pathensembles):
    """Compute and collect average time to hit M"""
    # average path lengths, but only the part before the 1st crossing
    # points before start and after M are not counted
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
    """Compute and collect average time after hitting M"""
    # average path lengths, but only the part after the last crossing
    # points before M and after end are not counted
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
    """Compute and collect average time between first and last hit of M"""
    # average path lengths, but only the part after the first crossing
    # and before the last crossing of M
    # so, point in the middle (m)
    # other points are not counted
    print("Collect taum")
    taum_mm = np.zeros(len(pathensembles))
    taum_mp = np.zeros(len(pathensembles))
    taum_pm = np.zeros(len(pathensembles))
    taum_pp = np.zeros(len(pathensembles))

    for i in range(len(pathensembles)):
        pe = pathensembles[i]
        # check if tau exists for this path type, then rest should exist too
        if pe.tauavg['LML'] is not None:
            taum_mm[i] = pathensembles[i].tauavg['LML'] \
                    - pathensembles[i].tau1avg['LML'] \
                    - pathensembles[i].tau2avg['LML']
        if pe.tauavg['LMR'] is not None:            
            taum_mp[i] = pathensembles[i].tauavg['LMR'] \
                   - pathensembles[i].tau1avg['LMR'] \
                   - pathensembles[i].tau2avg['LMR']
        if pe.tauavg['RML'] is not None:
            taum_pm[i] = pathensembles[i].tauavg['RML'] \
                   - pathensembles[i].tau1avg['RML'] \
                   - pathensembles[i].tau2avg['RML']
        if pe.tauavg['RMR'] is not None:
            taum_pp[i] = pathensembles[i].tauavg['RMR'] \
                   - pathensembles[i].tau1avg['RMR'] \
                   - pathensembles[i].tau2avg['RMR']

    return taum_mm, taum_mp, taum_pm, taum_pp

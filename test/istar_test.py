from json import load
import numpy as np
import numpy as np
import logging
import matplotlib.pyplot as plt
import deeptime.markov as dpt
#%matplotlib qt   # doesn't work on my laptop
from tistools import read_inputfile, get_LMR_interfaces, read_pathensemble, get_weights
from tistools import set_tau_distrib, set_tau_first_hit_M_distrib, cross_dist_distr, pathlength_distr
from tistools import ACCFLAGS, REJFLAGS

from tistools import get_lmr_masks, get_generation_mask, get_flag_mask, select_with_masks
from tistools import unwrap_by_weight, running_avg_local_probs, get_local_probs, get_global_probs_from_dict, get_global_probs_from_local, construct_M_milestoning, global_pcross_msm

from pprint import pprint    # to print the vars of the pathensemble object
# Created file and added transition function
# EW - April 2024

# Hard-coded rejection flags found in output files

logger = logging.getLogger(__name__)

def global_pcross_msm_star(M, doprint=False):
    # probability to arrive in -1 before 0
    # given that you are at 0 now and that you are leaving 0
    # = crossing probability from 0 to -1

    NS = len(M)
    assert NS>2

    # take pieces of transition matrix
    Mp = M[2:-1,2:-1]
    a = np.identity(NS-3)-Mp    # 1-Mp
    # a1 = np.linalg.inv(a)       # (1-Mp)^(-1)  --> bad practice!

    # other pieces
    D = M[2:-1, np.array([0,-1])]
    E = M[np.array([0,-1]), 2:-1]
    M11 = M[np.array([0,-1]),np.array([0,-1])]

    # compute Z vector
    z1 = np.array([[0],[1]])
    # z2 = np.dot(a1,np.dot(D,z1))
    z2 = np.linalg.solve(a, np.dot(D,z1))

    # compute H vector
    y1 = np.dot(M11,z1) + np.dot(E,z2)
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

def construct_M_istar(P, NS, N):
    """Construct transition matrix M"""
    # N -- number of interfaces
    # NS -- dimension of MSM, 4*N-5 when N>=4
    # P -- ndarray of probabilities for paths between end turns
    
    # assert N>=3
    assert N==P.shape[0]
    assert N==P.shape[1]
    assert NS==max(4, 2*N)

    # construct transition matrix
    M = np.zeros((NS,NS))
    
    # states [0-] and [0*+-]
    M[0,2] = 1
    M[2,1] = P[0,0]
    M[2,N+1:] = P[0, 1:]
    M[1,0] = 1
    M[-1,0] = 1
    M[N+1:,1] = P[1:, 0]

    # non-sampled paths
    # M[N+1, -1] = 1

    for i in range(1,N):
        #print("starting from state i",i)
        M[2+i, N+i:2*N] = P[i,i:]
        M[N+i, 3:2+i] = P[i, 1:i]
    

    # for i in range(NS):
    #     if np.sum(M[i]) > 0:
    #         M[i] = M[i]/np.sum(M[i])
    #     else:
    #         M[i] = 0 
       
    # non-sampled paths
    if not M[N, -1] >= 0:
        M[N, -1] = 0
    M[N+1,1] = 1
    # return np.delete(np.delete(M, N, 0), N, 1)
    return M

def get_transition_probs(w_path, weights = None, tr=False):
    """
    Returns the local crossing probabilities for the PPTIS ensemble pe.
    This is only for the [i^+-] or [0^+-'] ensembles, and NOT for [0^-'].

    Parameters
    ----------
    pes : List of PathEnsemble objects
        The PathEnsemble object must be from an [i*] simulation.
    w : array-like, optional
        The weights of the paths. If None, the weights are calculated from
        the flags. The default is None.
    tr : bool, optional
        If True, infinite time-reversal reweighting is performed.
    Returns
    -------
    p : ndarray(2, 2)
        Matrix of all local crossing probabilities from one turn to another,
        in both directions.
    """
    sh = w_path[0].shape
    p = np.empty([sh[0], sh[0]])
    for i in range(sh[0]):
        for k in range(sh[1]):
            if i == k:
                if i == 0:
                    p[i][k] = np.sum(w_path[i+1][i][k]) / np.sum(w_path[i+1][i][i:]) if np.sum(w_path[i+1][i][i:]) != 0 else 0
                else:
                    p[i][k] = 0
            elif i < k:
                p_reachedj = np.empty(k-i+1)
                p_jtillend = np.empty(k-i+1)
                w_reachedj = np.empty(k-i+1)
                w_jtillend = np.empty(k-i+1)
                for j in range(i, k+1):
                    p_reachedj[j-i] = np.sum(w_path[i+1][i][j:]) / np.sum(w_path[i+1][i][i:]) if np.sum(w_path[i+1][i][i:]) != 0 else 0
                    w_reachedj[j-i] = np.sum(w_path[i+1][i][i:])
                    if j < sh[0]-1:
                        p_jtillend[j-i] = np.sum(w_path[j+1][i][k]) / np.sum(w_path[j+1][i][i:]) if np.sum(w_path[j+1][i][i:]) != 0 else 0
                        w_jtillend[j-i] = np.sum(w_path[j+1][i][i:])
                    else: 
                        p_jtillend[j-i] = 1
                        w_jtillend[j-i] = 1
                print(f"i={i}, #j = {k-i}, k={k}")
                print("P_i(j reached) =", p_reachedj)
                print("P_j(k) =", p_jtillend)
                print("full P_i(k) =", p_reachedj*p_jtillend)
                print("weights: ", w_reachedj*w_jtillend)
                print("weighted P_i(k) =", np.average(p_reachedj * p_jtillend, weights=w_reachedj*w_jtillend) if np.sum(w_reachedj*w_jtillend) != 0 else 0)
                print("vs normal avg: ", np.average(p_reachedj * p_jtillend))
                p[i][k] = np.average(p_reachedj * p_jtillend, weights=w_reachedj*w_jtillend) if np.sum(w_reachedj*w_jtillend) != 0 else 0                
                # p[i][k] = np.average(p_reachedj * p_jtillend)
            elif i > k:
                p_reachedj = np.empty(i-k+1)
                p_jtillend = np.empty(i-k+1)
                w_reachedj = np.empty(i-k+1)
                w_jtillend = np.empty(i-k+1)
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
                    
                print(f"i={i}, #j = {k-i}, k={k}")
                print("P_i(j reached) =", p_reachedj)
                print("P_j(k) =", p_jtillend)
                print("full P_i(k) =", p_reachedj*p_jtillend)
                print("weights: ", w_reachedj*w_jtillend)
                print("weighted P_i(k) =", np.average(p_reachedj * p_jtillend, weights=w_reachedj*w_jtillend) if np.sum(w_reachedj*w_jtillend) != 0 else 0)
                print("vs normal avg: ", np.average(p_reachedj * p_jtillend))
                p[i][k] = np.average(p_reachedj * p_jtillend, weights=w_reachedj*w_jtillend) if np.sum(w_reachedj*w_jtillend) != 0 else 0
                # p[i][k] = np.average(p_reachedj * p_jtillend)

    msg = "Local crossing probabilities computed"
    print(msg)

    return p


def get_transition_probz(pes, interfaces, weights = None, tr=False): # cut memory
    """
    Returns the local crossing probabilities for the PPTIS ensemble pe.
    This is only for the [i^+-] or [0^+-'] ensembles, and NOT for [0^-'].

    Parameters
    ----------
    pes : List of PathEnsemble objects
        The PathEnsemble object must be from an [i*] simulation.
    w : array-like, optional
        The weights of the paths. If None, the weights are calculated from
        the flags. The default is None.
    tr : bool, optional
        If True, infinite time-reversal reweighting is performed.
    Returns
    -------
    p : ndarray(2, 2)
        Matrix of all local crossing probabilities from one turn to another,
        in both directions.
    """
    masks = {}
    w_path = {}

    for i, pe in enumerate(pes):
        # Get the lmr masks, weights, ACCmask, and loadmask of the paths
        masks[i] = get_lmr_masks(pe)
        if weights is None:
            w, ncycle_true = get_weights(pe.flags, ACCFLAGS, REJFLAGS, verbose = False)
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

        w_path[i] = {}

        w_path[i] = np.empty([len(interfaces),len(interfaces)])
        for j in range(len(interfaces)):
            for k in range(len(interfaces)):
                if j == k:
                        if i == 1 and j == 0:
                            w_path[i][j][k] = np.sum(select_with_masks(w, [masks[i]["LML"], accmask, ~loadmask]))
                        else:
                            w_path[i][j][k] = 0  
                elif j < k:
                    dir_mask = pe.dirs == 1
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
                    dir_mask = pe.dirs == -1
                    if k == 0:
                        start_cond = pe.lambmins <= interfaces[k]
                    else: 
                        start_cond = np.logical_and(pe.lambmins <= interfaces[k], pe.lambmins >= interfaces[k-1])
                    if j == len(interfaces)-1:
                        end_cond = pe.lambmaxs >= interfaces[j]
                    else: 
                        end_cond = np.logical_and(pe.lambmaxs >= interfaces[j], pe.lambmaxs <= interfaces[j+1])

                    w_path[i][j][k] = np.sum(select_with_masks(w, [start_cond, end_cond, dir_mask, accmask, ~loadmask]))
                    

        # if tr:  # TR reweighting. Note this is not block-friendly TODO
        #     w_path[i]['RMR'] *= 2
        #     w_path[i]['LML'] *= 2
        #     temp = w_path[i]['RML'] + w_path[i]['LMR']
        #     w_path[i]['RML'] = temp
        #     w_path[i]['LMR'] = temp

    p = np.empty([len(interfaces), len(interfaces)])
    for i in range(len(interfaces)):
        for k in range(len(interfaces)):
            if i == k:
                if i == 0:
                    p[i][k] = np.sum(w_path[i+1][i][k]) / np.sum(w_path[i+1][i][i:]) if np.sum(w_path[i+1][i][i:]) != 0 else 0
                else:
                    p[i][k] = 0
            elif i < k:
                p_reachedj = np.empty(k-i+1)
                p_jtillend = np.empty(k-i+1)
                w_reachedj = np.empty(k-i+1)
                w_jtillend = np.empty(k-i+1)
                for j in range(i, k+1):
                    p_reachedj[j-i] = np.sum(w_path[i+1][i][j:]) / np.sum(w_path[i+1][i][i:]) if np.sum(w_path[i+1][i][i:]) != 0 else 0
                    w_reachedj[j-i] = np.sum(w_path[i+1][i][i:])
                    if j < len(interfaces)-1:
                        p_jtillend[j-i] = np.sum(w_path[j+1][:k, k]) / np.sum(np.triu(w_path[j+1])[:k]) if np.sum(np.triu(w_path[j+1])[:k]) != 0 else 0
                        w_jtillend[j-i] = np.sum(np.triu(w_path[j+1])[:k])
                    else: 
                        p_jtillend[j-i] = 1
                        w_jtillend[j-i] = 1
                print(f"i={interfaces[i]}, #j = {k-i}, k={interfaces[k]}")
                print("P_i(j reached) =", p_reachedj)
                print("P_j(k) =", p_jtillend)
                print("full P_i(k) =", p_reachedj*p_jtillend)
                print("weights: ", w_reachedj*w_jtillend)
                print("weighted P_i(k) =", np.average(p_reachedj * p_jtillend, weights=w_reachedj*w_jtillend) if np.sum(w_reachedj*w_jtillend) != 0 else 0)
                print("vs normal avg: ", np.average(p_reachedj * p_jtillend))
                p[i][k] = np.average(p_reachedj * p_jtillend, weights=w_reachedj*w_jtillend) if np.sum(w_reachedj*w_jtillend) != 0 else 0                
                # p[i][k] = np.average(p_reachedj * p_jtillend)
            elif i > k:
                p_reachedj = np.empty(i-k+1)
                p_jtillend = np.empty(i-k+1)
                w_reachedj = np.empty(i-k+1)
                w_jtillend = np.empty(i-k+1)
                for j in range(k, i+1):
                    if i < len(interfaces)-1:
                        p_reachedj[j-k] = np.sum(w_path[i+1][i][:j+1]) / np.sum(w_path[i+1][i][:i+1]) if np.sum(w_path[i+1][i][:i+1]) != 0 else 0
                        p_jtillend[j-k] = np.sum(w_path[j+1][i][k]) / np.sum(np.tril(w_path[j+1])[k:]) if np.sum(np.tril(w_path[j+1])[k:]) != 0 else 0
                        w_reachedj[j-k] = np.sum(w_path[i+1][i][:i+1])
                        w_jtillend[j-k] = np.sum(np.tril(w_path[j+1])[k:])
                    else: 
                        p_reachedj[j-k] = 0
                        p_jtillend[j-k] = 0
                        w_reachedj[j-k] = 0
                        w_jtillend[j-k] = 0
                    
                print(f"i={interfaces[i]}, #j = {k-i}, k={interfaces[k]}")
                print("P_i(j reached) =", p_reachedj)
                print("P_j(k) =", p_jtillend)
                print("full P_i(k) =", p_reachedj*p_jtillend)
                print("weights: ", w_reachedj*w_jtillend)
                print("weighted P_i(k) =", np.average(p_reachedj * p_jtillend, weights=w_reachedj*w_jtillend) if np.sum(w_reachedj*w_jtillend) != 0 else 0)
                print("vs normal avg: ", np.average(p_reachedj * p_jtillend))
                p[i][k] = np.average(p_reachedj * p_jtillend, weights=w_reachedj*w_jtillend) if np.sum(w_reachedj*w_jtillend) != 0 else 0
                # p[i][k] = np.average(p_reachedj * p_jtillend)

    msg = "Local crossing probabilities computed"
    print(msg)

    return p


def get_transition_probzz2(pes, interfaces, weights = None, tr=False):
    """
    Returns the local crossing probabilities for the PPTIS ensemble pe.
    This is only for the [i^+-] or [0^+-'] ensembles, and NOT for [0^-'].

    Parameters
    ----------
    pes : List of PathEnsemble objects
        The PathEnsemble object must be from an [i*] simulation.
    w : array-like, optional
        The weights of the paths. If None, the weights are calculated from
        the flags. The default is None.
    tr : bool, optional
        If True, infinite time-reversal reweighting is performed.
    Returns
    -------
    p : ndarray(2, 2)
        Matrix of all local crossing probabilities from one turn to another,
        in both directions.
    """
    masks = {}
    w_path = {}

    for i, pe in enumerate(pes):
        # Get the lmr masks, weights, ACCmask, and loadmask of the paths
        masks[i] = get_lmr_masks(pe)
        if weights is None:
            w, ncycle_true = get_weights(pe.flags, ACCFLAGS, REJFLAGS, verbose = False)
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

        w_path[i] = {}

        w_path[i] = np.empty([len(interfaces),len(interfaces)])
        for j in range(len(interfaces)):
            for k in range(len(interfaces)):
                if j == k:
                        if i == 1 and j == 0:
                            w_path[i][j][k] = np.sum(select_with_masks(w, [masks[i]["LML"], accmask, ~loadmask]))
                        else:
                            w_path[i][j][k] = 0  
                elif j < k:
                    if i in {1,2} and j == 0 and k == 1:
                        dir_mask = pe.dirs < 2
                    else:
                        dir_mask = pe.dirs == 1
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
                    if i in {1,2} and j == 1 and k == 0:
                        dir_mask = pe.dirs > 2
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

                    w_path[i][j][k] = np.sum(select_with_masks(w, [start_cond, end_cond, dir_mask, accmask, ~loadmask]))
                    

    p = np.empty([len(interfaces), len(interfaces)])
    q_tot = np.ones([len(interfaces), len(interfaces)])
    for i in range(len(interfaces)):
        for k in range(len(interfaces)):
            counts = np.zeros(2)
            if i == k:
                if i == 0:
                    q_tot[i][k] = 1
                    continue
                else:
                    q_tot[i][k] = 0
                    continue
            elif i == 0 and k == 1:
                q_tot[i][k] = (np.sum(w_path[i+1][i][k:])) / (np.sum(w_path[i+1][i][k-1:]))
                continue
            elif i < k:
                counts += [np.sum(w_path[k][i][k:]), np.sum(w_path[k][i][k-1:])]
                print(i,k,np.sum(w_path[k][i][k:])/np.sum(w_path[k][i][k-1:]), np.sum(w_path[k][i][k-1:]))
            elif i > k:
                if k+2 > len(interfaces)-1:
                    break
                counts += [np.sum(w_path[k+2][i][:k+1]), np.sum(w_path[k+2][i][:k+2])]
                print (i,k,np.sum(w_path[k+2][i][:k+1])/np.sum(w_path[k+2][i][:k+2]), np.sum(w_path[k+2][i][:k+2]))

            q_tot[i][k] = counts[0] / counts[1] if counts[1] > 0 else 0
            if 0 in counts:
                print("zero:", q_tot[i][k], counts, i,k)
    print("q_tot: ", q_tot)

    for i in range(len(interfaces)):
        for k in range(len(interfaces)):
            if i < k:
                if k == len(interfaces)-1:
                    p[i][k] = np.prod(q_tot[i][i+1:k+1])
                else:
                    p[i][k] = np.prod(q_tot[i][i+1:k+1]) * (1-q_tot[i][k+1])
            elif k < i:
                if k == 0:
                    p[i][k] = np.prod(q_tot[i][k:i])
                else:
                    p[i][k] = np.prod(q_tot[i][k:i]) * (1-q_tot[i][k-1])
                if i == len(interfaces)-1:
                    p[i][k] = 0
            else:
                if i == 0:
                    p[i][k] = 1-q_tot[i][1]
                else:
                    p[i][k] = 0
    print("p: ", p)

    msg = "Local crossing probabilities computed"
    print(msg)

    return p


def get_transition_probzz(w_path):
    """
    Returns the local crossing probabilities for the PPTIS ensemble pe.
    This is only for the [i^+-] or [0^+-'] ensembles, and NOT for [0^-'].

    Parameters
    ----------
    pes : List of PathEnsemble objects
        The PathEnsemble object must be from an [i*] simulation.
    w : array-like, optional
        The weights of the paths. If None, the weights are calculated from
        the flags. The default is None.
    tr : bool, optional
        If True, infinite time-reversal reweighting is performed.
    Returns
    -------
    p : ndarray(2, 2)
        Matrix of all local crossing probabilities from one turn to another,
        in both directions.
    """
    p = np.empty([w_path[0].shape[0], w_path[0].shape[0]])
    q = np.ones([w_path[0].shape[0], w_path[0].shape[0]])
    for i in range(w_path[0].shape[0]):
        for k in range(w_path[0].shape[0]):
            counts = np.zeros(2)
            if i == k:
                if i == 0:
                    q[i][k] = 1
                    continue
                else:
                    q[i][k] = 0
                    continue
            elif i == 0 and k == 1:
                q[i][k] = (np.sum(w_path[i+1][i][k:])) / (np.sum(w_path[i+1][i][k-1:]))
                continue
            elif i < k:
                for pe_i in range(i+1,k+1):
                    if pe_i > w_path[0].shape[0]-1:
                        break
                    # if k-i > 2 and pe_i >= k-1:
                    #     continue
                    counts += [np.sum(w_path[pe_i][i][k:]), np.sum(w_path[pe_i][i][k-1:])]
                    print(pe_i-1,i,k,np.sum(w_path[pe_i][i][k:])/np.sum(w_path[pe_i][i][k-1:]), np.sum(w_path[pe_i][i][k-1:]))
            elif i > k:
                for pe_i in range(k+2,i+2):
                    if pe_i > w_path[0].shape[0]-1:
                        break
                    # if i-k > 2 and pe_i <= k+3:
                    #     continue
                    counts += [np.sum(w_path[pe_i][i][:k+1]), np.sum(w_path[pe_i][i][:k+2])]
                    print (pe_i-1,i,k,np.sum(w_path[pe_i][i][:k+1])/np.sum(w_path[pe_i][i][:k+2]), np.sum(w_path[pe_i][i][:k+2]))

            q[i][k] = counts[0] / counts[1] if counts[1] > 0 else 0
            if 0 in counts:
                print(q[i][k], counts, i,k)
    print("q: ", q)

    for i in range(w_path[0].shape[0]):
        for k in range(w_path[0].shape[0]):
            if i < k:
                if k == w_path[0].shape[0]-1:
                    p[i][k] = np.prod(q[i][i+1:k+1])
                else:
                    p[i][k] = np.prod(q[i][i+1:k+1]) * (1-q[i][k+1])
            elif k < i:
                if k == 0:
                    p[i][k] = np.prod(q[i][k:i])
                else:
                    p[i][k] = np.prod(q[i][k:i]) * (1-q[i][k-1])
                if i == w_path[0].shape[0]-1:
                    p[i][k] = 0
            else:
                if i == 0:
                    p[i][k] = 1-q[i][1]
                else:
                    p[i][k] = 0
    print("p: ", p)

    msg = "Local crossing probabilities computed"
    print(msg)

    return p


def get_transition_probss(pes, interfaces, weights = None, tr=False):
    """
    Returns the local crossing probabilities for the PPTIS ensemble pe.
    This is only for the [i^+-] or [0^+-'] ensembles, and NOT for [0^-'].

    Parameters
    ----------
    pes : List of PathEnsemble objects
        The PathEnsemble object must be from an [i*] simulation.
    w : array-like, optional
        The weights of the paths. If None, the weights are calculated from
        the flags. The default is None.
    tr : bool, optional
        If True, infinite time-reversal reweighting is performed.
    Returns
    -------
    p : ndarray(2, 2)
        Matrix of all local crossing probabilities from one turn to another,
        in both directions.
    """
    masks = {}
    n_path = {}

    for i, pe in enumerate(pes):
        # Get the lmr masks, weights, ACCmask, and loadmask of the paths
        masks[i] = get_lmr_masks(pe)
        if weights is None:
            w, ncycle_true = get_weights(pe.flags, ACCFLAGS, REJFLAGS, verbose = False)
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

        n_path[i] = {}

        n_path[i] = np.empty([len(interfaces),len(interfaces)])
        for j in range(len(interfaces)):
            for k in range(len(interfaces)):
                if j == k:
                        if i == 1 and j == 0:
                            n_path[i][j][k] = np.sum(select_with_masks(w, [masks[i]["LML"], accmask, ~loadmask]))
                        else:
                            n_path[i][j][k] = 0  
                elif j < k:
                    if j == 0 and k == 1:
                        if i == 1:
                            dir_mask = pe.dirs < 2
                        elif i == 2:
                            dir_mask = pe.dirs > 2
                    else:
                        dir_mask = pe.dirs == 1
                    if j == 0:
                        start_cond = pe.lambmins <= interfaces[j]
                    else: 
                        start_cond = np.logical_and(pe.lambmins <= interfaces[j], pe.lambmins >= interfaces[j-1])
                    if k == len(interfaces)-1:
                        end_cond = pe.lambmaxs >= interfaces[k]
                    else: 
                        end_cond = np.logical_and(pe.lambmaxs >= interfaces[k], pe.lambmaxs <= interfaces[k+1])
                
                    n_path[i][j][k] = np.sum(select_with_masks(w, [start_cond, end_cond, dir_mask, accmask, ~loadmask]))
                else:
                    if j == 1 and k == 0:
                        if i == 1:
                            dir_mask = pe.dirs > 2
                        elif i == 2:
                            dir_mask = pe.dirs < 2
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

                    n_path[i][j][k] = np.sum(select_with_masks(w, [start_cond, end_cond, dir_mask, accmask, ~loadmask]))


    p = np.empty([len(interfaces), len(interfaces)])
    q_tot = np.ones([len(interfaces), len(interfaces)])
    for i in range(len(interfaces)):
        for k in range(len(interfaces)):
            counts = np.zeros(2)
            if i == k:
                if i == 0:
                    q_tot[i][k] = 1
                    continue
                else:
                    q_tot[i][k] = 0
                    continue
            elif i == 0 and k == 1:
                q_tot[i][k] = np.sum(n_path[i+1][i][k:]) / np.sum(n_path[i+1][i][k-1:])
                continue
            elif i < k:
                for pe_i in range(i+1,k+1):
                    if pe_i > len(interfaces)-1:
                        break
                    # if k-i > 2 and pe_i >= k-1:
                    #     continue
                    counts += [np.sum(n_path[pe_i][i][k:]), np.sum(n_path[pe_i][i][k-1:])]
                    print("log ", pe_i-1,i,k,np.sum(n_path[pe_i][i][k:])/np.sum(n_path[pe_i][i][k-1:]), np.sum(n_path[pe_i][i][k-1:]))
            elif i > k:
                for pe_i in range(k+2,i+2):
                    if pe_i > len(interfaces)-1:
                        break
                    # if i-k > 2 and pe_i <= k+3:
                    #     continue
                    counts += [np.sum(n_path[pe_i][i][:k+1]), np.sum(n_path[pe_i][i][:k+2])]
                    print ("log ", pe_i-1,i,k,np.sum(n_path[pe_i][i][:k+1])/np.sum(n_path[pe_i][i][:k+2]), np.sum(n_path[pe_i][i][:k+2]))

            q_tot[i][k] = counts[0] / counts[1] if counts[1] > 0 else 0
            if 0 in counts:
                print("0 paths: ", q_tot[i][k], counts, i,k)
    print("q_tot: ", q_tot)
    
    counts_prime = np.zeros(2)
    for i in range(len(interfaces)):
        for k in range(len(interfaces)):
            if i < k:
                for pe_i in range(i+1,k+1):
                    if pe_i > len(interfaces)-1:
                        break
                    counts_prime += [n_path[pe_i][i][k], np.sum(n_path[pe_i][i][k-1:])]
                q_prime = counts_prime[0] / counts_prime[1]

                if k == len(interfaces)-1:
                    p[i][k] = np.prod(q_tot[i][i+1:k]) * q_prime
                else:

                    p[i][k] = np.prod(q_tot[i][i+1:k]) * q_prime
            elif k < i:
                for pe_i in range(k+2,i+2):
                    if pe_i > len(interfaces)-1:
                        break
                    counts_prime += [n_path[pe_i][i][k], np.sum(n_path[pe_i][i][:k+2])]
                q_prime = counts_prime[0] / counts_prime[1]

                if k == 0:
                    p[i][k] = np.prod(q_tot[i][k+1:i]) * q_prime
                else:
                    p[i][k] = np.prod(q_tot[i][k+1:i]) * q_prime
                if i == len(interfaces)-1:
                    p[i][k] = 0
            else:
                if i == 0:
                    p[i][k] = 1-q_tot[i][1]
                else:
                    p[i][k] = 0
    print("p: ", p)

    msg = "Local crossing probabilities computed"
    print(msg)

    return p




def get_simple_probs(w_path):
    """
    Returns the local crossing probabilities for the PPTIS ensemble pe.
    This is only for the [i^+-] or [0^+-'] ensembles, and NOT for [0^-'].

    Parameters
    ----------
    pes : List of PathEnsemble objects
        The PathEnsemble object must be from an [i*] simulation.
    w : array-like, optional
        The weights of the paths. If None, the weights are calculated from
        the flags. The default is None.
    tr : bool, optional
        If True, infinite time-reversal reweighting is performed.
    Returns
    -------
    p : ndarray(2, 2)
        Matrix of all local crossing probabilities from one turn to another,
        in both directions.
    """
    p = np.empty([w_path[0].shape[0], w_path[0].shape[0]])

    for i in range(w_path[0].shape[0]):
        for k in range(w_path[0].shape[0]):
            if i < k:
                if i == 0 or i >= w_path[0].shape[0]-2:
                    if k == w_path[0].shape[0]-1:
                        p[i][k] = np.sum(w_path[i+1][i][k:]) / np.sum(w_path[i+1][i][i:])
                    else:
                        p[i][k] = (w_path[i+1][i][k]) / np.sum(w_path[i+1][i][i:])
                else:
                    p[i][k] = (w_path[i+1][i][k] + w_path[i+2][i][k]) / (np.sum(w_path[i+1][i][i:]) + np.sum(w_path[i+2][i][i:]))
            elif k < i:
                if i == w_path[0].shape[0]-1:
                    p[i][k] = 0
                else:
                    p[i][k] = (w_path[i+1][i][k] + w_path[i][i][k]) / (np.sum(w_path[i+1][i][:i]) + np.sum(w_path[i][i][:i]))
            else:
                if i == 0:
                    p[i][k] = w_path[i+1][i][k] / np.sum(w_path[i+1][i][i:])
                else:
                    p[i][k] = 0
    print("p: ", p)
    

    msg = "Local crossing probabilities computed"
    print(msg)

    return p


def get_summed_probs(pes, interfaces, weights = None, dbens=False):
    """
    Returns the local crossing probabilities for the PPTIS ensemble pe.
    This is only for the [i^+-] or [0^+-'] ensembles, and NOT for [0^-'].

    Parameters
    ----------
    pes : List of PathEnsemble objects
        The PathEnsemble object must be from an [i*] simulation.
    w : array-like, optional
        The weights of the paths. If None, the weights are calculated from
        the flags. The default is None.
    tr : bool, optional
        If True, infinite time-reversal reweighting is performed.
    Returns
    -------
    p : ndarray(2, 2)
        Matrix of all local crossing probabilities from one turn to another,
        in both directions.
    """
    masks = {}
    w_path = {}

    for i, pe in enumerate(pes):
        # Get the lmr masks, weights, ACCmask, and loadmask of the paths
        masks[i] = get_lmr_masks(pe)
        if weights is None:
            w, ncycle_true = get_weights(pe.flags, ACCFLAGS, REJFLAGS, verbose = False)
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

        w_path[i] = {}

        w_path[i] = np.zeros([len(interfaces),len(interfaces)])
        for j in range(len(interfaces)):
            for k in range(len(interfaces)):
                if j == k:
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
        print(f"sun {i}=", np.sum(w_path[i]))
                    

    p = np.zeros([len(interfaces), len(interfaces)])

    for i in range(len(interfaces)):
        for k in range(len(interfaces)):
            counts = np.zeros(2)
            if i < k:
                for pe_i in range(i+1,k+2):
                    if pe_i > len(interfaces)-1:
                        break
                    counts += [w_path[pe_i][i][k], np.sum(w_path[pe_i][i][i:])]
            elif k < i:
                if i == len(interfaces)-1:
                    p[i][k] = 0
                    continue
                for pe_i in range(min(k+1,i+1),i+2):
                    if pe_i > len(interfaces)-1:
                        break
                    counts += [w_path[pe_i][i][k], np.sum(w_path[pe_i][i][:i+1])]
            else:
                if i == 0:
                    counts += [w_path[1][i][k], np.sum(w_path[1][i][i:])]
                else:
                    counts[1] += 1
            p[i][k] = counts[0] / counts[1] if counts[1] != 0 else 0
            if 0 in counts:
                print("zerooo ", p[i][k], counts, i,k)
    print("p: ", p)

    msg = "Local crossing probabilities computed"
    print(msg)

    return p


def compute_weight_matrices(pes, interfaces, n_int=None, tr = False, weights = None):
    masks = {}
    w_path = {}
    X = {}
    if n_int is None:
        n_int = len(pes)
    # if pes[-1].orders is not None:
    #     ax = plt.figure().add_subplot()
    for i, pe in enumerate(pes):
        # Get the lmr masks, weights, ACCmask, and loadmask of the paths
        masks[i] = get_lmr_masks(pe)
        if weights is None:
            # w, ncycle_true = get_weights(pe.flags, ACCFLAGS, REJFLAGS, verbose = True)
            w = get_weights_staple(i, pe.flags, pe.generation, pe.lmrs, n_int, ACCFLAGS, REJFLAGS, verbose = True)
            # assert ncycle_true == pe.ncycle
        else:
            w = weights[i]
        accmask = get_flag_mask(pe, "ACC")
        loadmask = get_generation_mask(pe, "ld")
        msg = f"Ensemble {pe.name[-3:]} has {len(w)} paths.\n The total "+\
                    f"weight of the ensemble is {np.sum(w)}\nThe total amount of "+\
                    f"accepted paths is {np.sum(accmask)}\nThe total amount of "+\
                    f"load paths is {np.sum(loadmask)}"
        logger.debug(msg)

        w_path[i] = np.zeros([len(interfaces),len(interfaces)])
        X[i] = np.zeros([len(interfaces),len(interfaces)])

        for j in range(len(interfaces)):
            for k in range(len(interfaces)):
                if j == k:
                        if i == 1 and j == 0:
                            # np.logical_and(pe.lambmaxs >= interfaces[1], pe.lambmaxs <= interfaces[2])
                            w_path[i][j][k] = np.sum(select_with_masks(w, [masks[i]["LML"], accmask, ~loadmask]))
                                #   + np.sum(select_with_masks(w, [start_cond, end_cond, dir_mask, accmask, ~loadmask]))
                            pass
                        else:
                            w_path[i][j][k] = 0
                elif j < k:
                    if j == 0 and k == 1:
                        if i > 2:
                            # dir_mask = np.logical_or(pe.dirs == 1, masks[i]["LML"])
                            dir_mask = pe.dirs == 1
                        elif i == 1:
                            # dir_mask = pe.dirs == 1
                            # dir_mask = np.full_like(pe.dirs, True)
                            dir_mask = masks[i]["LMR"]      # Distinction for 0 -> 1 paths in [0*] 
                        elif i == 2:
                            # dir_mask = pe.dirs == 1
                            dir_mask = np.full_like(pe.dirs, True)     # For now no distinction yet for [1*] paths: classify all paths as 1 -> 0. Later: check if shooting point comes before or after crossing lambda_1/lambda_max
                        else:
                            dir_mask = np.full_like(pe.dirs, False)
                    elif j == len(interfaces)-2 and k == len(interfaces)-1:
                        dir_mask = pe.dirs == 1
                        # dir_mask = masks[i]["RMR"]
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
                    if j == 1 and k == 0:
                        if i > 2:
                            dir_mask = pe.dirs == -1
                            # dir_mask = np.logical_or(pe.dirs == -1, masks[i]["RMR"])
                        elif i == 1:
                            # dir_mask = pe.dirs == -1
                            dir_mask = masks[i]["RML"]      # Distinction for 1 -> 0 paths in [0*] 
                            # dir_mask = np.full_like(pe.dirs, False)
                        elif i == 2:
                            # dir_mask = pe.dirs == -1
                            dir_mask = np.full_like(pe.dirs, True)     # For now no distinction yet for [1*] paths: classify all paths as 1 -> 0. Later: check if shooting point comes before or after crossing lambda_1/lambda_max
                        else:
                            dir_mask = np.full_like(pe.dirs, False)
                    elif j == len(interfaces)-2 and k == len(interfaces)-1:
                        # dir_mask = pe.dirs == -1
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

            # color = next(ax._get_lines.prop_cycler)['color']
        #         if j == 1 and k == 3 and pe.orders is not None:
        #             # idxs = select_with_masks(pe.cyclenumbers, [masks[i]["LML"], accmask, ~loadmask])
        #             idxs = select_with_masks(pe.cyclenumbers, [start_cond, end_cond, dir_mask,accmask, ~loadmask])
        #             for p in np.random.choice(idxs, min(1, len(idxs))):
        #                 if len(pe.orders[p][0]) > 1:
        #                     ax.plot([pe.orders[p][i][0] for i in range(len(pe.orders[p]))], [pe.orders[p][i][0] for i in range(len(pe.orders[p]))], "-x")
        #                 else:
        #                     ax.plot([i  for i in range(len(pe.orders[p]))],
        #                             [ph[0] for ph in pe.orders[p]], "-x")
        #                 ax.plot(0, pe.orders[p][0][0], "^",
        #                             color=ax.lines[-1].get_color(), ms = 7)
        #                 ax.plot(len(pe.orders[p]) - 1,
        #                         pe.orders[p][-1][0], "v",
        #                         color=ax.lines[-1].get_color(), ms = 7)
        #                     # plot the first and last point again to highlight start/end phasepoints
        #                     # it must have the same color as the line for the path
        #             if interfaces is not None:
        #                 for intf in interfaces:
        #                     ax.axhline(intf, color="k", ls="--", lw=.5)
        # if pe.orders is not None:
        #     plt.tight_layout()
        #     plt.show()
        print(f"sum weights ensemble {i}=", np.sum(w_path[i]))

    # weighting: consistent within ensemble by doubling forward and backwards LML/RMR paths (except in [0*] and [N-1*])
    #            consistent between different ensembles, more sampled paths in some ensembles appropriately weighted as a consequence of internal weighting
    # for i in range(1,len(pes)):
    #     for j in range(len(interfaces)):
    #         for k in range(len(interfaces)):
    #             if (i == 1) or \
    #                (i == 2 and j in [0,1] and k in [0,1]) or \
    #                (i == len(pes)-1 and j in [len(pes)-2, len(pes)-1] and k in [len(pes)-2, len(pes)-1]) or \
    #                (j != i-1 and k != i-1):
    #                 X[i][j][k] = w_path[i][j][k]
    #                 continue
    #             X[i][j][k] = 2*w_path[i][j][k]

    # for i in range(1,len(pes)):
    #     for j in range(len(interfaces)):
    #         for k in range(len(interfaces)):
    #             if (i == 1) or \
    #                (i == 2 and j in [0,1] and k in [0,1]) or \
    #                (i == len(pes)-1 and j in [len(pes)-2, len(pes)-1] and k in [len(pes)-2, len(pes)-1]) or \
    #                (j != i-1 and k != i-1):
    #                 X[i][j][k] = w_path[i][j][k]
    #                 continue
    #             X[i][j][k] = w_path[i][j][k] + w_path[i][k][j]
    
    X = w_path
    for i in range(len(interfaces)):
        if tr:
            if i == 2 and X[i][0, 1] == 0:     # In [1*] all LML paths are classified as 1 -> 0 (for now).
                X[i][1, 0] *= 2     # Time reversal needs to be adjusted to compensate for this
            X[i] += X[i].T          # Will not be needed anymore once LML paths are separated in 0 -> 1 and 1 -> 0.
    return X


def compute_weight_matrix(pe, pe_id, interfaces, n_int=None, tr = False,weights = None):

    if n_int is None:
        n_int = len(interfaces)
    # Get the lmr masks, weights, ACCmask, and loadmask of the paths
    masks = get_lmr_masks(pe)
    if weights is None:
        w = get_weights_staple(pe_id, pe.flags, pe.generation, pe.lmrs, len(interfaces), ACCFLAGS, REJFLAGS, verbose = True)
        # w, ncycle_true = get_weights(pe.flags, ACCFLAGS, REJFLAGS, verbose = False)
        # w, ncycle_true = get_weights(pe.flags, pe.dirs, ACCFLAGS, REJFLAGS, verbose = False)
        # assert ncycle_true == pe.ncycle
    else:
        w = weights
    accmask = get_flag_mask(pe, "ACC")
    loadmask = get_generation_mask(pe, "ld")
    msg = f"Ensemble {pe.name[-3:]} has {len(w)} paths.\n The total "+\
                f"weight of the ensemble is {np.sum(w)}\nThe total amount of "+\
                f"accepted paths is {np.sum(accmask)}\nThe total amount of "+\
                f"load paths is {np.sum(loadmask)}"
    logger.debug(msg)

    X_path = np.empty([len(interfaces),len(interfaces)])

    for j in range(len(interfaces)):
        for k in range(len(interfaces)):
            if j == k:
                    if pe_id == 1 and j == 0:
                        X_path[j][k] = np.sum(select_with_masks(w, [masks["LML"], accmask, ~loadmask]))
                        continue
                    else:
                        X_path[j][k] = 0  
            elif j < k:
                if j == 0 and k == 1:
                    if pe_id > 2:
                        # dir_mask = np.logical_or(pe.dirs == 1, masks[i]["LML"])
                        dir_mask = pe.dirs == 1
                    elif pe_id == 1:
                        dir_mask = np.full_like(pe.dirs, True)
                    else:
                        dir_mask = np.full_like(pe.dirs, False)
                elif j == len(interfaces)-2 and k == len(interfaces)-1:
                    dir_mask = masks["RMR"]
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
            else:
                if j == 1 and k == 0:
                    if pe_id > 2:
                        dir_mask = pe.dirs == -1
                        # dir_mask = np.logical_or(pe.dirs == -1, masks[i]["RMR"])
                    elif pe_id == 2:
                        dir_mask = np.full_like(pe.dirs, False) # don't count to prevent them from being included in TR, p_1,0 is always 1 anyway
                    else:
                        dir_mask = np.full_like(pe.dirs, False)
                elif j == len(interfaces)-2 and k == len(interfaces)-1:
                    dir_mask = masks["RMR"]
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

                X_path[j][k] = np.sum(select_with_masks(w, [start_cond, end_cond, dir_mask, accmask, ~loadmask]))
    print(f"sum weights ensemble {pe_id}=", np.sum(X_path))

    if tr:
        if X_path[1, 0] == 0:
            X_path[0, 1] *= 2
        X_path = X_path + X_path.T

    return X_path


def get_weights_staple(pe_i, flags,gen,ptypes,n_pes,ACCFLAGS,REJFLAGS,verbose=True):
    """
    Returns:
      weights -- array with weight of each trajectory, 0 if not accepted
      ncycle_true -- sum of weights
    """

    ntraj = len(flags)
    assert len(flags) == len(gen) == len (ptypes)
    weights = np.zeros(ntraj,int)
    w_mc = np.zeros(ntraj,int)

    accepted = 0
    rejected = 0
    omitted = 0

    acc_w = 0
    acc_index = 0
    tot_w = 0
    prev_ha = 1
    assert flags[0] == 'ACC'
    for i,fgp in enumerate(zip(flags,gen, ptypes)):
        flag, gg, ptype = fgp
        if flag in ACCFLAGS:
            # store previous traj with accumulated weight
            weights[acc_index] = prev_ha*acc_w
            w_mc[acc_index] = acc_w
            tot_w += prev_ha*acc_w
            # info for new traj
            acc_index = i
            acc_w = 1
            accepted += 1
            if gg == 'sh' and n_pes > 3 and\
                ((pe_i == 2 and ptype == "RMR") or\
                (pe_i == n_pes-1 and ptype == "LML") or\
                (2 < pe_i < n_pes-1 and ptype in ["LML", "RMR"])) :
            # if gg == 'sh' and ptype in ["LML", "RMR"] :
                prev_ha = 2
            else:
                prev_ha = 1
        elif flag in REJFLAGS:
            acc_w += 1    # weight of previous accepted traj increased
            rejected += 1
        else:
            omitted += 1
    #if flag[-1] in REJFLAGS:
        # I did not store yet the weight of the previous accepted path
        # because I do not have the next accepted path yet
        # so neglect this path, I guess.
    # at the end: store the last accepted path with its weight
    weights[acc_index] = prev_ha*acc_w
    w_mc[acc_index] = acc_w
    tot_w += prev_ha*acc_w

    if verbose:
        print("weights:")
        print("accepted     ",accepted)
        print("rejected     ",rejected)
        print("omitted      ",omitted)
        print("total trajs  ",ntraj)
        print("total MC weights",np.sum(w_mc))
        print("total MC + HA weights",np.sum(weights))

    assert omitted == 0
    # ncycle_true = np.sum(weights)
    # miss = len(flags)-1 - ncycle_true
    # for i in range(miss):
    #     assert flags[-(i+1)] in REJFLAGS
        # the reason why this could happen

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
            w = get_weights_staple(i, pe.flags, pe.generation, pe.lmrs, len(interfaces), ACCFLAGS, REJFLAGS, verbose = False)
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
        plocistar[i]["LML"] = [np.sum(X[i][:max(1,i-1), i-1])/np.sum(X[i][:max(1,i-1), i-1:]), np.sum(X[i][:max(1,i-1), i-1:])]
        plocistar[i]["LMR"] = [1-plocistar[i]["LML"][0], np.sum(X[i][:max(1,i-1), i-1:])]
        plocistar[i]["RMR"] = [np.sum(X[i][i:, max(1, i-1)])/np.sum(X[i][i:, :i]), np.sum(X[i][i:, :i])]
        plocistar[i]["RML"] = [1-plocistar[i]["RMR"][0], np.sum(X[i][i:, :i])]

        plocistar[i]["startLMR"] = np.zeros(max(2, i))
        wsum = np.sum(X[i][:max(1,i-1), i-1:])
        # plocistar[i]["startLMR"][1] = np.array([np.sum(X[i][j, max(0,i-2):]) for j in range(max(1, i-2))])
        plocistar[i]["startLMR"][:-1] = np.array([np.sum(X[i][j, i:]) for j in range(max(1, i-1))]) / wsum
        plocistar[i]["startLMR"][-1] = wsum
        plocistar[i]["startLMLf"] = plocistar[i]["startLMR"].copy()
        plocistar[i]["startLMLf"][:-1] = np.array([np.sum(X[i][j, i-1]) for j in range(max(1, i-1))]) / wsum
        plocistar[i]["startLMLf"][-1] = wsum
        plocistar[i]["startRML"] = np.zeros(len(interfaces)-i+1)
        wsum = np.sum(X[i][i:, :i])
        # plocistar[i]["startRML"][1] = np.array([np.sum(X[i][j, :i]) for j in range(i, len(interfaces))])
        plocistar[i]["startRML"][:-1] = np.array([np.sum(X[i][j, :max(1,i-1)]) for j in range(i, len(interfaces))]) / wsum
        plocistar[i]["startRML"][-1] = wsum
        plocistar[i]["startRMRb"] = plocistar[i]["startRML"].copy()
        plocistar[i]["startRMRb"][:-1] = np.array([np.sum(X[i][j, max(1,i-1)]) for j in range(i, len(interfaces))]) / wsum
        plocistar[i]["startRMRb"][-1] = wsum

        plocistar[i]["endLMR"] = np.zeros(len(interfaces)-i+1)
        wsum = np.sum(X[i][:max(1,i-1), i-1:]) 
        plocistar[i]["endLMR"][:-1] = np.array([np.sum(X[i][:max(1,i-1), j]) for j in range(i, len(interfaces))]) / wsum
        plocistar[i]["endLMR"][-1] = wsum
        plocistar[i]["endRML"] = np.zeros(max(2, i))
        wsum = np.sum(X[i][i:, :i])
        plocistar[i]["endRML"][:-1] = np.array([np.sum(X[i][i:, j]) for j in range(max(1,i-1))]) / wsum
        plocistar[i]["endRML"][-1] = wsum

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

        print(f"Ensemble {i} ([{i-1}*]):")
        print(f"REPPTIS pLMR approx: {plocistar[i]['LMR'][0]} (vs. REPPTIS = {plocrepptis[i]['LMR']}) with # weights = {plocistar[i]['LMR'][1]}")
        print(f"    Conditional pLMR with start turns from 0 till {max(0, i-2)}: {plocistar[i]['startLMR'][:-1]} with sum {np.sum(plocistar[i]['startLMR'][:-1])} \n    with # weights {plocistar[i]['startLMR'][-1]}")
        print(f"    Conditional pLMR with end turns from {i} till {len(interfaces)-1}: {plocistar[i]['endLMR'][:-1]} with sum {np.sum(plocistar[i]['endLMR'][:-1])} \n    with # weights {plocistar[i]['endLMR'][-1]}")
        print(f"REPPTIS pLML fw approx: {plocistar[i]['LML'][0]} (vs. REPPTIS = {plocrepptis[i]['LML']}) with # weights = {plocistar[i]['LML'][1]}")
        print(f"    Conditional pLML with start turns from 0 till {max(0, i-2)}: {plocistar[i]['startLMLf'][:-1]} with sum {np.sum(plocistar[i]['startLMLf'][:-1])} \n    with # weights {plocistar[i]['startLMLf'][-1]}")
        # print(f"Conditional pLML with end turns from {i} till {len(interfaces)}: {plocistar[i]["endLML"][0]} with sum {np.sum(plocistar[i]["endLML"][0])} \nwith # weights {plocistar[i]["endLML"][1]}")
        print(f"REPPTIS pRML approx: {plocistar[i]['RML'][0]} (vs. REPPTIS = {plocrepptis[i]['RML']}) with # weights = {plocistar[i]['RML'][1]}")
        print(f"    Conditional pRML with start turns from {i} till {len(interfaces)-1}: {plocistar[i]['startRML'][:-1]} with sum {np.sum(plocistar[i]['startRML'][:-1])} \n    with # weights {plocistar[i]['startRML'][-1]}")
        print(f"    Conditional pRML with end turns from {0} till {i-1}: {plocistar[i]['endRML'][:-1]} with sum {np.sum(plocistar[i]['endRML'][:-1])} \n    with # weights {plocistar[i]['endRML'][-1]}")
        print(f"REPPTIS pRMR bw approx: {plocistar[i]['RMR'][0]} with # weights = {plocistar[i]['RMR'][1]}")
        print(f"    Conditional pRMR with start turns from {i} till {len(interfaces)-1}: {plocistar[i]['startRMRb'][:-1]} with sum {np.sum(plocistar[i]['startRMRb'][:-1])} \n    with # weights {plocistar[i]['startRMRb'][-1]}")
        # print(f"Conditional pRMR with end turns from {i-1} till {0}: {list(reversed(plocistar[i]["endRMR"][0]))} with sum {np.sum(plocistar[i]["endRMR"][0])} \nwith # weights {plocistar[i]["endRMR"][1]}")
        print(f"Full conditional turn probability matrix for ensemble {i}:")
        print(np.array_str(plocistar[i]["full"], precision=5, suppress_small=True))
        print("\n")

    return plocrepptis, plocistar

def cprobs_repptis_istar2(pes, interfaces, n_int=None):
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
            w = get_weights_staple(i, pe.flags, pe.generation, pe.lmrs, len(interfaces), ACCFLAGS, REJFLAGS, verbose = False)
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

def plot_rv_star(pes, interfaces, numberof):
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
            # l_cond = pe.orders[id][:, 0] <= interfaces[max(0, i-2)]
            # r_cond = pe.orders[id][:, 0] >= interfaces[i]
            # m_cond = np.logical_and(interfaces[i] >= pe.orders[id][:, 0], pe.orders[id][:, 0] >= interfaces[max(0, i-2)])
            # l_piece = pe.orders[id][l_cond, :]
            # m_piece = pe.orders[id][m_cond, :]
            # r_piece = pe.orders[id][r_cond, :]
            l_piece = pe.orders[id][:pe.istar_idx[id][0]+1, :]
            m_piece = pe.orders[id][pe.istar_idx[id][0]-1:pe.istar_idx[id][1]+2, :]
            r_piece = pe.orders[id][pe.istar_idx[id][1]:, :]
            # if np.all(m_piece[:, 1] >= 0):
            if len(m_piece) <= 100 and len(l_piece)+len(r_piece) <= 20550:
                ax.plot(m_piece[:, 0], m_piece[:, 1], "*-", color=linecolor, label=i)
                ax.plot(l_piece[:, 0], l_piece[:, 1], "--", alpha=0.5, color=linecolor)
                ax.plot(r_piece[:, 0], r_piece[:, 1], "--", alpha=0.5, color=linecolor)
                # linecolor = lines[0].get_color()
                count += 1
            it += 1
            if it > 200000:
                id = np.random.choice(pe.cyclenumbers[pe.lmrs == "LMR"])
                # l_cond = pe.orders[id][:, 0] <= interfaces[max(0, i-2)]
                # r_cond = pe.orders[id][:, 0] >= interfaces[i]
                # m_cond = np.logical_and(interfaces[i] >= pe.orders[id][:, 0], pe.orders[id][:, 0] >= interfaces[max(0, i-2)])
                # l_piece = pe.orders[id][l_cond, :]
                # m_piece = pe.orders[id][m_cond, :]
                # r_piece = pe.orders[id][r_cond, :]
                l_piece = pe.orders[id][:pe.istar_idx[id][0]+1, :]
                m_piece = pe.orders[id][pe.istar_idx[id][0]-1:pe.istar_idx[id][1]+2, :]
                r_piece = pe.orders[id][pe.istar_idx[id][1]:, :]
                if np.all(m_piece[:, 1] >= 0):
                    ax.plot(m_piece[:, 0], m_piece[:, 1], color=linecolor, label=i)
                    ax.plot(l_piece[:, 0], l_piece[:, 1], "--", alpha=0.5, color=linecolor)
                    ax.plot(r_piece[:, 0], r_piece[:, 1], "--", alpha=0.5, color=linecolor)
                    # linecolor = lines[0].get_color()
                    count += 1
    fig.legend()
    fig.show()

def display_data(pes, interfaces, n_int = None, weights = None):
    tresholdw=0.03
    tresholdtr = 0.05
    masks = {}
    X = {}
    C = {}
    C_md = {} # number of paths where new MD steps are performed (shooting/WF)
    if n_int is None:
        n_int = len(pes)
    for i, pe in enumerate(pes):
        print(10*'-')
        print(f"ENSEMBLE [{i-1 if i>0 else 0}{"*" if i>0 else "-"}] | ID {i}")
        print(10*'-')
        # Get the lmr masks, weights, ACCmask, and loadmask of the paths
        masks[i] = get_lmr_masks(pe)
        if weights is None:
            w = get_weights_staple(i, pe.flags, pe.generation, pe.lmrs, n_int, ACCFLAGS, REJFLAGS, verbose = True)
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

        X[i] = np.zeros([len(interfaces),len(interfaces)])
        C[i] = np.zeros([len(interfaces),len(interfaces)])
        C_md[i] = np.zeros([len(interfaces),len(interfaces)])
        X_val = compute_weight_matrix(pe, i, interfaces, tr = False, weights=weights)

        # 1. Displaying raw data, only unweighted X_ijk
        # 2. Displaying weighted data W_ijk
        # 3. Displaying weighted data with time reversal
        no_w = np.ones_like(w)
        for j in range(len(interfaces)):
            for k in range(len(interfaces)):
                if j == k:
                        if i == 1 and j == 0:
                            # np.logical_and(pe.lambmaxs >= interfaces[1], pe.lambmaxs <= interfaces[2])
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
                            dir_mask = np.full_like(pe.dirs, True)  # For now no distinction yet for [1*] paths: classify all paths as 1 -> 0. Later: check if shooting point comes before or after crossing lambda_1/lambda_max
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
                            # dir_mask = np.logical_or(pe.dirs == -1, masks[i]["RMR"])
                        elif i == 1:
                            # dir_mask = pe.dirs == -1
                            dir_mask = masks[i]["RML"]    # Distinction for 1 -> 0 paths in [0*]
                        elif i == 2:
                            dir_mask = np.full_like(pe.dirs, True)   # For now no distinction yet for [1*] paths: classify all paths as 1 -> 0. Later: check if shooting point comes before or after crossing lambda_1/lambda_max
                        else:
                            dir_mask = np.full_like(pe.dirs, False)
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

        print("1a. Raw data: unweighted C matrices")
        print(f"C[{i}] = ")
        pprint(C[i])
        print("1b. Raw data: unweighted path counts with new MD steps")
        print(f"C_md[{i}] = ")
        pprint(C_md[i])
        print("\n2. Weighted data: including high acceptance weights")
        print(f"X[{i}] = ")
        pprint(X[i])
        # pprint(X_val)
        print(f"sum weights ensemble {i}=", np.sum(X[i]))
        if len(idx_weirdw) > 0:
            print("[WARNING]")
            for idx in idx_weirdw:
                print(f"The weighted data significantly differs from the raw path count for paths that go from {idx[0]} to {idx[1]}. Counts: {C[i][idx[0]][idx[1]]} vs. weights: {X[i][idx[0]][idx[1]]} --> difference in fraction:{difffrac[idx[0], idx[1]]}. The number of new MD paths is {C_md[i][idx[0]][idx[1]]}")
        print("\n3a. Weighted data with time reversal")
        print(f"TR X[{i}] = ")
        X_tr = (X[i]+X[i].T)/2.0
        # if i == 2 and X[i][0, 1] == 0:     # In [1*] all LML paths are classified as 1 -> 0 (for now).
        #     X_tr[1, 0] *= 2          # Time reversal needs to be adjusted to compensate for this
        #     X_tr[0, 1] *= 2  
        pprint(X_tr)
        if len(idx_tr) > 0:
            print("[WARNING]")
            for idx in idx_tr:
                print(f"The reverse equivalent paths are significantly different for paths going from {idx[0]} to {idx[1]}. Relative difference: {tr_diff[idx[0],idx[1]]}. Weights L->R path: {X[i][idx[0]][idx[1]]} | Weights R->L path: {X[i][idx[1]][idx[0]]}")
        print("\n3b. Unweighted data with time reversal")
        print(f"TR C[{i}] = ")
        C_tr = (C[i]+C[i].T)/2.0
        # if i == 2 and C[i][0, 1] == 0:     # In [1*] all LML paths are classified as 1 -> 0 (for now).
        #     C_tr[1, 0] *= 2          # Time reversal needs to be adjusted to compensate for this
        #     C_tr[0, 1] *= 2  
        pprint(C_tr)
        # assert np.all(X[i] == X_val)

    print(10*'='+'\n')
    print(10*'-')
    print(f"ALL ENSEMBLES COMBINED")
    print(10*'-')
    W = np.zeros_like(X[1])
    for j in range(len(interfaces)):
        for k in range(len(interfaces)):
            W[j][k] = np.sum([X[i][j][k] for i in range(n_int)])
    print("4. Weights of all ensembles combined (sum), no TR")
    pprint(W)
    print("5. Weights of all ensembles combined (sum), with TR")
    W_tr = (W+W.T)/2.
    if W[1,0] == 0:
        W_tr[0,1] *= 1 
        W_tr[1,0] *= 1
    pprint(W_tr)

    return C, X, W


def memory_analysis(w_path, tr = False):
    ''' Print for every interface k > 0:
            For every starting interface i:
                -> how many paths reach k, given that they reached k-1
        This can be a measure for how much memory is in the system.
        A purely diffuse system would have this probability at 1/2 at all circumstances.
        Add diagram?

        First for every ensemble `ens`, then in total.
    '''
    q_k = np.zeros([2, w_path[0].shape[0]-1, w_path[0].shape[0], w_path[0].shape[0]])
    for ens in range(1, w_path[0].shape[0]):
        if tr:
            # if w_path[ens][1,0] == 0:
            #     w_path[ens][0,1] *= 2 
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
                    # for pe_i in range(i+1,k+1):
                    if i <= ens <= k:
                        # if k-i > 2 and pe_i >= k-1:
                        #     continue
                        counts += [np.sum(w_path[ens][i][k:]), np.sum(w_path[ens][i][k-1:])]
                        # print(ens-1,i,k,np.sum(w_path[ens][i][k:])/np.sum(w_path[ens][i][k-1:]), np.sum(w_path[ens][i][k-1:]))
                elif i > k:
                    if k+2 <= ens <= i+1:
                        # if i-k > 2 and pe_i <= k+3:
                        #     continue
                        counts += [np.sum(w_path[ens][i][:k+1]), np.sum(w_path[ens][i][:k+2])]
                        # print (ens-1,i,k,np.sum(w_path[ens][i][:k+1])/np.sum(w_path[ens][i][:k+2]), np.sum(w_path[ens][i][:k+2]))

                q_k[0][ens-1][i][k] = counts[0] / counts[1] if counts[1] > 0 else np.nan
                q_k[1][ens-1][i][k] = counts[1]
                # if 0 in counts:
                #     print(q_k[0][ens-1][i][k], counts, i,k)
        
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
                for pe_i in range(i+1,k+1):
                    if pe_i > w_path[0].shape[0]-1:
                        break
                    # if k-i > 2 and pe_i >= k-1:
                    #     continue
                    counts += [np.sum(w_path[pe_i][i][k:]), np.sum(w_path[pe_i][i][k-1:])]
                    # print(pe_i-1,i,k,np.sum(w_path[pe_i][i][k:])/np.sum(w_path[pe_i][i][k-1:]), np.sum(w_path[pe_i][i][k-1:]))
            elif i > k:
                for pe_i in range(k+2,i+2):
                    if pe_i > w_path[0].shape[0]-1:
                        break
                    # if i-k > 2 and pe_i <= k+3:
                    #     continue
                    counts += [np.sum(w_path[pe_i][i][:k+1]), np.sum(w_path[pe_i][i][:k+2])]
                    # print (pe_i-1,i,k,np.sum(w_path[pe_i][i][:k+1])/np.sum(w_path[pe_i][i][:k+2]), np.sum(w_path[pe_i][i][:k+2]))

            q_tot[0][i][k] = counts[0] / counts[1] if counts[1] > 0 else np.nan
            q_tot[1][i][k] = counts[1]
            # if 0 in counts:
            #     print(q_tot[0][i][k], counts, i,k)
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
    print("q_tot: ", q_tot)

    return(q_k, q_tot)

def ploc_memory(pathensembles, interfaces, trr=True):
    plocs = {}
    plocs["mlst"] = [1.,]
    plocs["apptis"] = [1.,]
    repptisploc = []

    # w = compute_weight_matrices(pathensembles, interfaces, len(interfaces), tr=tr)
    # p = get_transition_probzz(w)
    # # pi = get_simple_probs(wi)
    # M = construct_M_istar(p, 2*len(interfaces), len(interfaces))
    
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
            # plocs["mlst"].append(repptisploc[i]["2R"]*plocs["mlst"][-1])

        # APPTIS p_loc
        if i < len(pathensembles)-1:
            wi = compute_weight_matrices(pathensembles[:i+2], interfaces[:i+2], tr=trr)
            pi = get_transition_probzz(wi)
            # pi = get_simple_probs(wi)
            Mi = construct_M_istar(pi, max(4, 2*len(interfaces[:i+2])), len(interfaces[:i+2]))
            z1, z2, y1, y2 = global_pcross_msm_star(Mi)
            plocs["apptis"].append(y1[0][0])


    _, _, plocs["repptis"] = get_global_probs_from_dict(repptisploc)

    print("Milestoning p_loc: ", plocs["mlst"])
    print("REPPTIS p_loc: ", plocs["repptis"])
    print("APPTIS p_loc: ", plocs["apptis"])

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

def msm_metrics(M, interfaces):
    
    return
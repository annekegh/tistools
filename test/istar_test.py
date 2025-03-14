from json import load
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
    plt.imshow(q_tot[0], cmap='hot', interpolation='none')
    plt.colorbar()
    plt.show()

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

def plot_q_matrix(q_tot, interfaces=None):
    """
    Visualize the q_tot matrix from TIS simulations, showing memory effects in transitions.
    
    Parameters:
    -----------
    q_tot : numpy.ndarray
        A matrix where:
        - q_tot[0][i][k] for i<k: probability that a path starting at interface i and reaching k-1 will reach k
        - q_tot[0][i][k] for i>k: probability that a path starting at interface i and reaching k+1 will reach k
        - q_tot[1][i][k] contains the corresponding number of samples
    interfaces : list, optional
        The interface positions (for axis labeling)
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, ListedColormap
    import matplotlib.gridspec as gridspec
    import matplotlib.cm as cm
    from matplotlib.colors import to_hex
    
    # Extract the probability matrix and weights matrix from q_tot
    q_probs = q_tot[0]
    q_weights = q_tot[1]
    n_interfaces = q_probs.shape[0]
    
    if interfaces is None:
        interfaces = list(range(n_interfaces))
    
    # Create a figure with custom layout: heatmap and transition plots
    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.2, 1], width_ratios=[1.2, 1])
    
    # Create custom colormap centered at 0.5
    # Blue for < 0.5 (more diffusive), red for > 0.5 (less diffusive)
    heatmap_cmap = LinearSegmentedColormap.from_list('memory_effect', 
                                           [(0, 'blue'), (0.5, 'white'), (1, 'red')], N=256)
    
    # Generate high-contrast heatmap-like colors
    def generate_high_contrast_colors(n):
        if n <= 1:
            return ["#1f77b4"]  # Default blue for single item
            
        # Create a custom colormap with enhanced contrast between adjacent colors
        if n <= 10:
            # For fewer interfaces (≤10), offer two colormap options
            
            # Option 1: tab10 - qualitatively different colors that are still ordered
            # base_cmap = plt.cm.get_cmap('tab10', n)
            # return [to_hex(base_cmap(i)) for i in range(n)]
            
            # Option 2: viridis with enhanced spacing for better contrast
            viridis_cmap = plt.cm.get_cmap('viridis')
            # Use wider spacing to enhance contrast between adjacent colors
            return [to_hex(viridis_cmap(i/(n-1) if n > 1 else 0.5)) for i in range(n)]
        else:
            # For more interfaces, use viridis with adjusted spacing
            cmap1 = plt.cm.get_cmap('viridis')
            
            # Get colors with deliberate spacing for better contrast
            colors = []
            for i in range(n):
                # Distribute colors with slight variations in spacing
                # This avoids adjacent indices having too similar colors
                pos = (i / max(1, n-1)) * 0.85 + 0.1  # Scale to range 0.1-0.95
                
                # Introduce small oscillations in color position for adjacent indices
                if i % 2 == 1:
                    pos = min(0.95, pos + 0.05)
                    
                colors.append(to_hex(cmap1(pos)))
                
            return colors
    
    # Plot 1: Heatmap of q_probs
    ax_heat = plt.subplot(gs[0, 0])
    masked_data = np.ma.masked_invalid(q_probs)  # Mask NaN values
    im = ax_heat.imshow(masked_data, cmap=heatmap_cmap, vmin=0, vmax=1, 
                       interpolation='none', aspect='auto')
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax_heat, label='Probability')
    
    # Add reference line at 0.5
    cbar.ax.axhline(y=0.5, color='black', linestyle='--', linewidth=1)
    cbar.ax.text(1.5, 0.5, '0.5 (diffusive)', va='center', ha='left', fontsize=9)
    
    # Add annotations for probability values and sample sizes
    for i in range(n_interfaces):
        for k in range(n_interfaces):
            if not np.isnan(q_probs[i, k]) and not np.ma.is_masked(masked_data[i, k]):
                weight = q_weights[i, k]
                text = f"{q_probs[i, k]:.2f}\n(n={int(weight)})" if weight > 0 else "N/A"
                color = 'black' if 0.2 < q_probs[i, k] < 0.8 else 'white'
                ax_heat.text(k, i, text, ha='center', va='center', color=color, fontsize=8)
    
    # Set ticks and labels for heatmap
    ax_heat.set_xticks(np.arange(n_interfaces))
    ax_heat.set_yticks(np.arange(n_interfaces))
    if interfaces is not None:
        ax_heat.set_xticklabels([f"{i}" for i in range(n_interfaces)])
        ax_heat.set_yticklabels([f"{i}" for i in range(n_interfaces)])
    ax_heat.set_xlabel('Target Interface k')
    ax_heat.set_ylabel('Starting Interface i')
    ax_heat.set_title('Memory Effect Matrix: q(i,k)', fontsize=14)
    
    # No grid for heatmap
    
    # Add explanatory text
    desc_text = """
    This matrix shows the probability q(i,k) that:
    • For i<k: a path starting at interface i and reaching k-1 will reach k
    • For i>k: a path starting at interface i and reaching k+1 will reach k
    
    In a purely diffusive process, all values would be 0.5.
    Values > 0.5 (red) indicate a bias toward crossing the next interface.
    Values < 0.5 (blue) indicate a bias toward returning without crossing.
    """
    fig.text(0.02, 0.02, desc_text, fontsize=10, wrap=True)
    
    # Plot 2: Forward transitions (L→R) - q(i,k) for i<k
    ax_forward = plt.subplot(gs[0, 1])
    
    # Create colors for forward transitions
    forward_targets = [k for k in range(1, n_interfaces)]
    forward_colors = generate_high_contrast_colors(len(forward_targets))
    
    # For each target interface k, plot q(i,k) for all starting interfaces i<k
    for idx, k in enumerate(forward_targets):
        target_data = []
        starting_interfaces = []
        for i in range(k):
            if not np.isnan(q_probs[i, k]) and q_weights[i, k] > 5:  # Minimum sample threshold
                target_data.append(q_probs[i, k])
                starting_interfaces.append(i)
        
        if target_data:
            ax_forward.plot(starting_interfaces, target_data, 'o-', 
                          label=f'Target: {k}', linewidth=2, markersize=8,
                          color=forward_colors[idx])
    
    # Add reference line at 0.5
    ax_forward.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7)
    
    # Configure the forward plot
    ax_forward.set_xlabel('Starting Interface i')
    ax_forward.set_ylabel('Probability q(i,k)')
    ax_forward.set_title('Forward Transition Probabilities (L→R)', fontsize=12)
    ax_forward.set_ylim(0, 1.05)
    ax_forward.set_xlim(-0.5, n_interfaces-1.5)
    ax_forward.set_xticks(range(n_interfaces))
    ax_forward.grid(True, alpha=0.3)
    ax_forward.legend(title='Target Interface k', loc='upper center', 
                     bbox_to_anchor=(0.5, -0.15), ncol=min(5, n_interfaces-1))
    
    # Plot 3: Memory effect by distance - grouping transitions by the same jump size
    ax_jump = plt.subplot(gs[1, 0])
    
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
            ax_jump.plot(starting_positions, probs, 'o-', 
                       label=f'Δλ = {dist}', linewidth=2, markersize=8,
                       color=jump_colors[idx])
    
    # Add reference line at 0.5
    ax_jump.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7)
    
    # Configure the jump distance plot
    ax_jump.set_xlabel('Starting Interface i')
    ax_jump.set_ylabel('Probability q(i, i+dist)')
    ax_jump.set_title('Memory Effect by Jump Distance (L→R)', fontsize=12)
    ax_jump.set_ylim(0, 1.05)
    ax_jump.set_xlim(-0.5, n_interfaces-1.5)
    ax_jump.set_xticks(range(n_interfaces))
    ax_jump.grid(True, alpha=0.3)
    ax_jump.legend(title='Jump Distance Δλ', loc='upper center', 
                  bbox_to_anchor=(0.5, -0.15), ncol=min(5, max_distance))
    
    # Plot 4: Backward transitions (R→L) - q(i,k) for i>k
    ax_backward = plt.subplot(gs[1, 1])
    
    # Create colors for backward transitions
    backward_targets = [k for k in range(n_interfaces-1)]
    backward_colors = generate_high_contrast_colors(len(backward_targets))
    
    # For each target interface k, plot q(i,k) for all starting interfaces i>k
    for idx, k in enumerate(backward_targets):
        target_data = []
        starting_interfaces = []
        for i in range(k+1, n_interfaces):
            if not np.isnan(q_probs[i, k]) and q_weights[i, k] > 5:  # Minimum sample threshold
                target_data.append(q_probs[i, k])
                starting_interfaces.append(i)
        
        if target_data:
            ax_backward.plot(starting_interfaces, target_data, 'o-', 
                           label=f'Target: {k}', linewidth=2, markersize=8,
                           color=backward_colors[idx])
    
    # Add reference line at 0.5
    ax_backward.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7)
    
    # Configure the backward plot
    ax_backward.set_xlabel('Starting Interface i')
    ax_backward.set_ylabel('Probability q(i,k)')
    ax_backward.set_title('Backward Transition Probabilities (R→L)', fontsize=12)
    ax_backward.set_ylim(0, 1.05)
    ax_backward.set_xlim(0.5, n_interfaces-0.5)
    ax_backward.set_xticks(range(n_interfaces))
    ax_backward.grid(True, alpha=0.3)
    ax_backward.legend(title='Target Interface k', loc='upper center', 
                      bbox_to_anchor=(0.5, -0.15), ncol=min(5, n_interfaces-1))
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.suptitle('Transition Interface Sampling Memory Analysis', fontsize=16)
    
    return fig

# Example usage - call this function after running your memory_analysis function
# q_k, q_tot = memory_analysis(w_path, tr=True)
# fig = plot_q_matrix(q_tot, interfaces)
# plt.show()


def plot_memory_analysis(q_tot, p, interfaces=None):
    """
    Create alternative visualizations for memory effects in TIS simulations.
    
    Parameters:
    -----------
    q_tot : numpy.ndarray
        A matrix where:
        - q_tot[0][i][k] for i<k: probability that a path starting at interface i and reaching k-1 will reach k
        - q_tot[0][i][k] for i>k: probability that a path starting at interface i and reaching k+1 will reach k
        - q_tot[1][i][k] contains the corresponding number of samples
    p : numpy.ndarray
        Transition probability matrix between interfaces
    interfaces : list, optional
        The interface positions (for axis labeling)
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    import matplotlib.gridspec as gridspec
    import matplotlib.cm as cm
    import matplotlib.colors as colors
    from matplotlib.ticker import MaxNLocator
    import seaborn as sns
    from scipy.optimize import curve_fit
    
    # Extract the probability matrix and weights matrix from q_tot
    q_probs = q_tot[0]
    q_weights = q_tot[1]
    n_interfaces = q_probs.shape[0]
    
    if interfaces is None:
        interfaces = list(range(n_interfaces))
    
    # ================ Figure 1: Forward Memory Decay Analysis ================
    fig1 = plt.figure(figsize=(16, 14))
    gs1 = gridspec.GridSpec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1])
    
    # Create custom colormap for heatmaps
    cmap = LinearSegmentedColormap.from_list('memory_effect', 
                                          [(0, 'blue'), (0.5, 'white'), (1, 'red')], N=256)
    
    # Plot 1: Memory Decay with Distance (Forward transitions)
    ax1 = fig1.add_subplot(gs1[0, 0])
    
    # Calculate memory effect as deviation from 0.5 (diffusive behavior)
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
    sns.heatmap(memory_data, cmap='viridis', ax=ax1, 
                cbar_kws={'label': '|Probability - 0.5| (Memory Effect Strength)'})
    
    ax1.set_xlabel('Interface Distance (k - i)')
    ax1.set_ylabel('Starting Interface i')
    ax1.set_title('Memory Effect Decay with Distance (Forward L→R)', fontsize=12)
    ax1.set_xticks(np.arange(n_interfaces-1) + 0.5)
    ax1.set_xticklabels(np.arange(1, n_interfaces))
    ax1.set_yticks(np.arange(n_interfaces-1) + 0.5)
    ax1.set_yticklabels(np.arange(n_interfaces-1))
    
    # Plot 2: Memory Decay Profile and Relaxation Times
    ax2 = fig1.add_subplot(gs1[0, 1])
    
    # Generate colors for different starting positions
    cmap_start = plt.cm.viridis
    colrs = [cmap_start(i/(n_interfaces-1)) for i in range(n_interfaces-1)]
    
    # Exponential decay function for fitting memory effects
    def exp_decay(x, a, tau, c):
        return a * np.exp(-x / tau) + c
    
    # Store fitted relaxation times
    relaxation_times = []
    relaxation_errors = []
    starting_points = []
    
    # Fit exponential decay to memory effect vs distance for each starting interface
    for i in range(n_interfaces-1):
        distances = np.array(range(1, n_interfaces - i))
        values = np.array([memory_data[i, d-1] for d in distances])
        valid_mask = ~np.isnan(values)
        
        if np.sum(valid_mask) >= 3:  # Need at least 3 points for meaningful fit
            try:
                # Initial parameter guesses
                p0 = [0.2, 1.0, 0.05]  # amplitude, tau, offset
                
                # Curve fitting
                popt, pcov = curve_fit(exp_decay, distances[valid_mask], values[valid_mask], p0=p0, 
                                      bounds=([0, 0, 0], [1, 10, 0.5]))
                
                # Extract relaxation time (tau) and its error
                tau = popt[1]
                tau_err = np.sqrt(np.diag(pcov))[1] if np.all(np.isfinite(pcov)) else 0
                
                relaxation_times.append(tau)
                relaxation_errors.append(tau_err)
                starting_points.append(i)
                
                # Plot fitted curve
                x_fit = np.linspace(min(distances), max(distances), 100)
                y_fit = exp_decay(x_fit, *popt)
                ax2.plot(x_fit, y_fit, '--', color=colrs[i], alpha=0.7)
                
                # Plot original data
                ax2.plot(distances[valid_mask], values[valid_mask], 'o-', 
                        label=f'Start: {i}, τ={tau:.2f}±{tau_err:.2f}', color=colrs[i])
                
            except RuntimeError:
                # If curve_fit fails, just plot the raw data
                ax2.plot(distances[valid_mask], values[valid_mask], 'o-', 
                        label=f'Start: {i} (fit failed)', color=colrs[i])
        else:
            # Just plot the raw data if not enough points for fitting
            if np.any(valid_mask):
                ax2.plot(distances[valid_mask], values[valid_mask], 'o-', 
                        label=f'Start: {i} (insufficient data)', color=colrs[i])
    
    ax2.set_xlabel('Interface Distance (k - i)')
    ax2.set_ylabel('Memory Effect Strength |P - 0.5|')
    ax2.set_title('Memory Decay Profile and Relaxation Times (Forward)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=0)
    ax2.legend(loc='upper right', ncol=1)
    
    # Inset: Relaxation times vs starting interface
    if len(relaxation_times) > 1:
        ax2_inset = ax2.inset_axes([0.55, 0.1, 0.4, 0.3])
        ax2_inset.bar(starting_points, relaxation_times, yerr=relaxation_errors, color=colrs, capsize=5)
        ax2_inset.set_xlabel('Starting Interface')
        ax2_inset.set_ylabel('Relaxation Time τ')
        ax2_inset.set_title('Memory Relaxation Time by Starting Point')
        ax2_inset.grid(True, alpha=0.3)
    
    # Plot 3: Deviations from Diffusive Behavior in Forward Transitions (using p matrix)
    ax3 = fig1.add_subplot(gs1[1, 0])
    
    # Create array for adjacency transitions (i to i+1) using p matrix
    adjacent_forward = np.zeros(n_interfaces - 1)
    
    for i in range(n_interfaces - 1):
        if p is not None and i < p.shape[0] and i+1 < p.shape[1]:
            # Use p matrix for adjacent transitions
            adjacent_forward[i] = p[i, i+1] - 0.5
        else:
            # Fall back to q_probs if p is not available
            if not np.isnan(q_probs[i, i+1]) and q_weights[i, i+1] > 5:
                adjacent_forward[i] = q_probs[i, i+1] - 0.5
            else:
                adjacent_forward[i] = np.nan
    
    # Create bar plot with positive/negative coloring
    bars = ax3.bar(range(n_interfaces - 1), adjacent_forward, 
                  color=['red' if x > 0 else 'blue' for x in adjacent_forward if not np.isnan(x)])
    ax3.axhline(y=0, color='k', linestyle='-', alpha=0.5)
    
    # Add annotations
    for i, v in enumerate(adjacent_forward):
        if not np.isnan(v):
            ax3.text(i, v + np.sign(v)*0.02, f"{v:.2f}", ha='center', fontsize=9)
    
    ax3.set_xlabel('Interface i')
    ax3.set_ylabel('Deviation from Diffusive (P - 0.5)')
    ax3.set_title('Memory Effect in Adjacent Forward Transitions (i → i+1)', fontsize=12)
    ax3.set_xticks(range(n_interfaces - 1))
    ax3.set_xticklabels([f"{i}" for i in range(n_interfaces - 1)])
    
    # Plot 4: Memory Retention Across Interfaces (using Relative Standard Deviation)
    ax4 = fig1.add_subplot(gs1[1, 1])
    
    # Calculate memory retention using relative standard deviation (coefficient of variation)
    # Excluding q(i,i) and q(i,i+1) as they are 0 and 1 respectively
    memory_retention = np.zeros(n_interfaces)
    
    for k in range(1, n_interfaces):
        # Get all probabilities for reaching k from different starting points
        # Exclude i=k (q(i,i)=0) and i=k-1 (q(i,i+1)=1)
        probs = [q_probs[i, k] for i in range(k-1) if not np.isnan(q_probs[i, k]) and q_weights[i, k] > 5]
        if len(probs) > 1:
            # Calculate relative standard deviation (coefficient of variation)
            # RSD = (standard deviation / mean) * 100%
            mean_prob = np.mean(probs)
            if mean_prob > 0:  # Avoid division by zero
                std_prob = np.std(probs)
                memory_retention[k] = (std_prob / mean_prob) * 100  # As percentage
            else:
                memory_retention[k] = np.nan
    
    # Plot the memory retention (RSD)
    valid_k = [k for k in range(1, n_interfaces) if not np.isnan(memory_retention[k])]
    valid_retention = [memory_retention[k] for k in valid_k]
    
    if valid_k:
        bars4 = ax4.bar(valid_k, valid_retention, color='purple', alpha=0.7)
        
        # Add annotations for memory retention values
        for i, k in enumerate(valid_k):
            ax4.text(k, valid_retention[i] + 1, f"{valid_retention[i]:.1f}%", ha='center', fontsize=9)
        
        ax4.set_xlabel('Target Interface k')
        ax4.set_ylabel('Relative Standard Deviation (%)')
        ax4.set_title('Memory Retention: Variability in Forward Transitions\n(excluding q(i,i) and q(i,i+1))', fontsize=12)
        ax4.set_xticks(range(1, n_interfaces))
        ax4.set_xticklabels([f"{i}" for i in range(1, n_interfaces)])
        ax4.grid(True, axis='y', alpha=0.3)
        ax4.set_ylim(0, max(valid_retention) * 1.2 if valid_retention else 10)  # Adjust y-axis with some headroom
    else:
        ax4.text(0.5, 0.5, "Insufficient data for memory retention analysis", 
               ha='center', va='center', transform=ax4.transAxes)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig1.suptitle('TIS Memory Effect Analysis - Forward Transitions', fontsize=16)
    
    # ================ Figure 3: Backward Memory Decay Analysis (NEW) ================
    fig3 = plt.figure(figsize=(16, 14))
    gs3 = gridspec.GridSpec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1])
    
    # Plot 1: Memory Decay with Distance (Backward transitions)
    ax7 = fig3.add_subplot(gs3[0, 0])
    
    # Calculate memory effect as deviation from 0.5 (diffusive behavior)
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
    sns.heatmap(backward_memory_data, cmap='viridis', ax=ax7, 
                cbar_kws={'label': '|Probability - 0.5| (Memory Effect Strength)'})
    
    ax7.set_xlabel('Interface Distance (i - k)')
    ax7.set_ylabel('Starting Interface i')
    ax7.set_title('Memory Effect Decay with Distance (Backward R→L)', fontsize=12)
    ax7.set_xticks(np.arange(n_interfaces-1) + 0.5)
    ax7.set_xticklabels(np.arange(1, n_interfaces))
    ax7.set_yticks(np.arange(n_interfaces-1) + 0.5)
    ax7.set_yticklabels(np.arange(1, n_interfaces))
    
    # Plot 2: Backward Memory Decay Profile and Relaxation Times
    ax8 = fig3.add_subplot(gs3[0, 1])
    
    # Generate colors for different starting positions
    backward_colrs = [cmap_start(i/(n_interfaces-1)) for i in range(1, n_interfaces)]
    
    # Store fitted relaxation times for backward transitions
    backward_relaxation_times = []
    backward_relaxation_errors = []
    backward_starting_points = []
    
    # Fit exponential decay to memory effect vs distance for each starting interface
    for i_idx, i in enumerate(range(1, n_interfaces)):
        distances = np.array(range(1, i+1))
        values = np.array([backward_memory_data[i-1, d-1] for d in distances])
        valid_mask = ~np.isnan(values)
        
        if np.sum(valid_mask) >= 3:  # Need at least 3 points for meaningful fit
            try:
                # Initial parameter guesses
                p0 = [0.2, 1.0, 0.05]  # amplitude, tau, offset
                
                # Curve fitting
                popt, pcov = curve_fit(exp_decay, distances[valid_mask], values[valid_mask], p0=p0, 
                                      bounds=([0, 0, 0], [1, 10, 0.5]))
                
                # Extract relaxation time (tau) and its error
                tau = popt[1]
                tau_err = np.sqrt(np.diag(pcov))[1] if np.all(np.isfinite(pcov)) else 0
                
                backward_relaxation_times.append(tau)
                backward_relaxation_errors.append(tau_err)
                backward_starting_points.append(i)
                
                # Plot fitted curve
                x_fit = np.linspace(min(distances), max(distances), 100)
                y_fit = exp_decay(x_fit, *popt)
                ax8.plot(x_fit, y_fit, '--', color=backward_colrs[i_idx], alpha=0.7)
                
                # Plot original data
                ax8.plot(distances[valid_mask], values[valid_mask], 'o-', 
                        label=f'Start: {i}, τ={tau:.2f}±{tau_err:.2f}', color=backward_colrs[i_idx])
                
            except RuntimeError:
                # If curve_fit fails, just plot the raw data
                ax8.plot(distances[valid_mask], values[valid_mask], 'o-', 
                        label=f'Start: {i} (fit failed)', color=backward_colrs[i_idx])
        else:
            # Just plot the raw data if not enough points for fitting
            if np.any(valid_mask):
                ax8.plot(distances[valid_mask], values[valid_mask], 'o-', 
                        label=f'Start: {i} (insufficient data)', color=backward_colrs[i_idx])
    
    ax8.set_xlabel('Interface Distance (i - k)')
    ax8.set_ylabel('Memory Effect Strength |P - 0.5|')
    ax8.set_title('Memory Decay Profile and Relaxation Times (Backward)', fontsize=12)
    ax8.grid(True, alpha=0.3)
    ax8.set_ylim(bottom=0)
    ax8.legend(loc='upper right', ncol=1)
    
    # Inset: Backward Relaxation times vs starting interface
    if len(backward_relaxation_times) > 1:
        ax8_inset = ax8.inset_axes([0.55, 0.1, 0.4, 0.3])
        ax8_inset.bar(backward_starting_points, backward_relaxation_times, yerr=backward_relaxation_errors, 
                    color=backward_colrs, capsize=5)
        ax8_inset.set_xlabel('Starting Interface')
        ax8_inset.set_ylabel('Relaxation Time τ')
        ax8_inset.set_title('Backward Memory Relaxation Time')
        ax8_inset.grid(True, alpha=0.3)
    
    # Plot 3: Deviations from Diffusive Behavior in Backward Transitions (using p matrix)
    ax9 = fig3.add_subplot(gs3[1, 0])
    
    # Create array for adjacency transitions (i to i-1) using p matrix
    adjacent_backward = np.zeros(n_interfaces - 1)
    
    for i in range(1, n_interfaces):
        if p is not None and i < p.shape[0] and i-1 < p.shape[1]:
            # Use p matrix for adjacent transitions
            adjacent_backward[i-1] = p[i, i-1] - 0.5
        else:
            # Fall back to q_probs if p is not available
            if not np.isnan(q_probs[i, i-1]) and q_weights[i, i-1] > 5:
                adjacent_backward[i-1] = q_probs[i, i-1] - 0.5
            else:
                adjacent_backward[i-1] = np.nan
    
    # Create bar plot with positive/negative coloring
    bars9 = ax9.bar(range(1, n_interfaces), adjacent_backward, 
                  color=['red' if x > 0 else 'blue' for x in adjacent_backward if not np.isnan(x)])
    ax9.axhline(y=0, color='k', linestyle='-', alpha=0.5)
    
    # Add annotations
    for i, v in enumerate(adjacent_backward):
        if not np.isnan(v):
            ax9.text(i+1, v + np.sign(v)*0.02, f"{v:.2f}", ha='center', fontsize=9)
    
    ax9.set_xlabel('Starting Interface i')
    ax9.set_ylabel('Deviation from Diffusive (P - 0.5)')
    ax9.set_title('Memory Effect in Adjacent Backward Transitions (i → i-1)', fontsize=12)
    ax9.set_xticks(range(1, n_interfaces))
    ax9.set_xticklabels([f"{i}" for i in range(1, n_interfaces)])
    
    # Plot 4: Backward Memory Retention Across Interfaces
    ax10 = fig3.add_subplot(gs3[1, 1])
    
    # Calculate backward memory retention using relative standard deviation
    backward_memory_retention = np.zeros(n_interfaces)
    
    for k in range(n_interfaces-1):
        # Get all probabilities for reaching k from different starting points i>k+1
        # Exclude i=k (q(i,i)=0) and i=k+1 (q(i,i-1)=1)
        probs = [q_probs[i, k] for i in range(k+2, n_interfaces) if not np.isnan(q_probs[i, k]) and q_weights[i, k] > 5]
        if len(probs) > 1:
            # Calculate relative standard deviation (coefficient of variation)
            mean_prob = np.mean(probs)
            if mean_prob > 0:  # Avoid division by zero
                std_prob = np.std(probs)
                backward_memory_retention[k] = (std_prob / mean_prob) * 100  # As percentage
            else:
                backward_memory_retention[k] = np.nan
    
    # Plot the backward memory retention (RSD)
    valid_k = [k for k in range(n_interfaces-1) if not np.isnan(backward_memory_retention[k])]
    valid_retention = [backward_memory_retention[k] for k in valid_k]
    
    if valid_k:
        bars10 = ax10.bar(valid_k, valid_retention, color='teal', alpha=0.7)
        
        # Add annotations for memory retention values
        for i, k in enumerate(valid_k):
            ax10.text(k, valid_retention[i] + 1, f"{valid_retention[i]:.1f}%", ha='center', fontsize=9)
        
        ax10.set_xlabel('Target Interface k')
        ax10.set_ylabel('Relative Standard Deviation (%)')
        ax10.set_title('Memory Retention: Variability in Backward Transitions\n(excluding q(i,i) and q(i,i-1))', fontsize=12)
        ax10.set_xticks(range(n_interfaces-1))
        ax10.set_xticklabels([f"{i}" for i in range(n_interfaces-1)])
        ax10.grid(True, axis='y', alpha=0.3)
        ax10.set_ylim(0, max(valid_retention) * 1.2 if valid_retention else 10)
    else:
        ax10.text(0.5, 0.5, "Insufficient data for backward memory retention analysis", 
                ha='center', va='center', transform=ax10.transAxes)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig3.suptitle('TIS Memory Effect Analysis - Backward Transitions', fontsize=16)
    
    # ================ Figure 2: Memory Effect Ratio Analysis ================
    fig2 = plt.figure(figsize=(16, 7))
    gs2 = gridspec.GridSpec(1, 2)
    
    # Plot 1: Memory Effect Heat Ratio (divergence from diffusive behavior)
    ax5 = fig2.add_subplot(gs2[0, 0])
    
    # Calculate memory effect ratio: P/(1-P) compared to diffusive 0.5/(1-0.5)=1
    memory_ratio = np.zeros_like(q_probs)
    memory_ratio.fill(np.nan)
    
    for i in range(n_interfaces):
        for k in range(n_interfaces):
            if not np.isnan(q_probs[i, k]) and q_probs[i, k] != 0 and q_probs[i, k] != 1:
                memory_ratio[i, k] = (q_probs[i, k] / (1 - q_probs[i, k])) / 1.0  # Normalize by diffusive ratio of 1
    
    # Plot heatmap with logarithmic scale
    im5 = ax5.imshow(memory_ratio, cmap='RdBu_r', norm=colors.LogNorm(vmin=0.1, vmax=10))
    
    # Add colorbar
    cbar5 = fig2.colorbar(im5, ax=ax5, label='Probability Ratio P/(1-P) [log scale]')
    
    # Add annotations for ratio values
    for i in range(n_interfaces):
        for k in range(n_interfaces):
            if not np.isnan(memory_ratio[i, k]) and q_weights[i, k] > 5:
                text_color = 'black'
                if memory_ratio[i, k] > 5 or memory_ratio[i, k] < 0.2:
                    text_color = 'white'
                ax5.text(k, i, f"{memory_ratio[i, k]:.2f}", ha='center', va='center', 
                       color=text_color, fontsize=8)
    
    ax5.set_xlabel('Target Interface k')
    ax5.set_ylabel('Starting Interface i')
    ax5.set_title('Memory Effect Ratio: Deviation from Diffusive Behavior', fontsize=12)
    ax5.set_xticks(range(n_interfaces))
    ax5.set_yticks(range(n_interfaces))
    
    # Plot 2: Memory Asymmetry - Forward vs Backward transitions
    ax6 = fig2.add_subplot(gs2[0, 1])
    
    # Calculate memory asymmetry for pairs of interfaces (i, j)
    memory_asymmetry = np.zeros_like(q_probs)
    memory_asymmetry.fill(np.nan)
    
    for i in range(n_interfaces):
        for j in range(n_interfaces):
            if i != j and not np.isnan(q_probs[i, j]) and not np.isnan(q_probs[j, i]):
                if q_weights[i, j] > 5 and q_weights[j, i] > 5:
                    # Asymmetry is the difference between forward and backward probabilities
                    memory_asymmetry[i, j] = q_probs[i, j] - q_probs[j, i]
    
    # Plot heatmap
    im6 = ax6.imshow(memory_asymmetry, cmap='RdBu', vmin=-0.5, vmax=0.5)
    
    # Add colorbar
    cbar6 = fig2.colorbar(im6, ax=ax6, label='Probability Asymmetry (i→j vs j→i)')
    
    # Add annotations
    for i in range(n_interfaces):
        for j in range(n_interfaces):
            if not np.isnan(memory_asymmetry[i, j]):
                text_color = 'black'
                if abs(memory_asymmetry[i, j]) > 0.3:
                    text_color = 'white'
                ax6.text(j, i, f"{memory_asymmetry[i, j]:.2f}", ha='center', va='center', 
                       color=text_color, fontsize=8)
    
    ax6.set_xlabel('Target Interface j')
    ax6.set_ylabel('Starting Interface i')
    ax6.set_title('Memory Asymmetry: Forward vs. Backward Transitions', fontsize=12)
    ax6.set_xticks(range(n_interfaces))
    ax6.set_yticks(range(n_interfaces))
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig2.suptitle('TIS Memory Effect Analysis - Advanced Metrics', fontsize=16)
    
    return fig1, fig2, fig3
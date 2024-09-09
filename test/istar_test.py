from json import load
import numpy as np
import numpy as np
import logging
import matplotlib.pyplot as plt
#%matplotlib qt   # doesn't work on my laptop
from tistools import read_inputfile, get_LMR_interfaces, read_pathensemble, get_weights
from tistools import set_tau_distrib, set_tau_first_hit_M_distrib, cross_dist_distr, pathlength_distr
from tistools import ACCFLAGS, REJFLAGS

from tistools import get_lmr_masks, get_generation_mask, get_flag_mask, select_with_masks
from tistools import unwrap_by_weight, running_avg_local_probs, get_local_probs, get_globall_probs, get_global_probz

from pprint import pprint    # to print the vars of the pathensemble object
# Created file and added transition function
# EW - April 2024

# Hard-coded rejection flags found in output files

logger = logging.getLogger(__name__)

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
    q = np.ones([len(interfaces), len(interfaces)])
    for i in range(len(interfaces)):
        for k in range(len(interfaces)):
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
                counts += [np.sum(w_path[k][i][k:]), np.sum(w_path[k][i][k-1:])]
                print(i,k,np.sum(w_path[k][i][k:])/np.sum(w_path[k][i][k-1:]), np.sum(w_path[k][i][k-1:]))
            elif i > k:
                if k+2 > len(interfaces)-1:
                    break
                counts += [np.sum(w_path[k+2][i][:k+1]), np.sum(w_path[k+2][i][:k+2])]
                print (i,k,np.sum(w_path[k+2][i][:k+1])/np.sum(w_path[k+2][i][:k+2]), np.sum(w_path[k+2][i][:k+2]))

            q[i][k] = counts[0] / counts[1] if counts[1] > 0 else 0
            if 0 in counts:
                print("zero:", q[i][k], counts, i,k)
    print("q: ", q)

    for i in range(len(interfaces)):
        for k in range(len(interfaces)):
            if i < k:
                if k == len(interfaces)-1:
                    p[i][k] = np.prod(q[i][i+1:k+1])
                else:
                    p[i][k] = np.prod(q[i][i+1:k+1]) * (1-q[i][k+1])
            elif k < i:
                if k == 0:
                    p[i][k] = np.prod(q[i][k:i])
                else:
                    p[i][k] = np.prod(q[i][k:i]) * (1-q[i][k-1])
                if i == len(interfaces)-1:
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
    q = np.ones([len(interfaces), len(interfaces)])
    for i in range(len(interfaces)):
        for k in range(len(interfaces)):
            counts = np.zeros(2)
            if i == k:
                if i == 0:
                    q[i][k] = 1
                    continue
                else:
                    q[i][k] = 0
                    continue
            elif i == 0 and k == 1:
                q[i][k] = np.sum(n_path[i+1][i][k:]) / np.sum(n_path[i+1][i][k-1:])
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

            q[i][k] = counts[0] / counts[1] if counts[1] > 0 else 0
            if 0 in counts:
                print("0 paths: ", q[i][k], counts, i,k)
    print("q: ", q)
    
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
                    p[i][k] = np.prod(q[i][i+1:k]) * q_prime
                else:

                    p[i][k] = np.prod(q[i][i+1:k]) * q_prime
            elif k < i:
                for pe_i in range(k+2,i+2):
                    if pe_i > len(interfaces)-1:
                        break
                    counts_prime += [n_path[pe_i][i][k], np.sum(n_path[pe_i][i][:k+2])]
                q_prime = counts_prime[0] / counts_prime[1]

                if k == 0:
                    p[i][k] = np.prod(q[i][k+1:i]) * q_prime
                else:
                    p[i][k] = np.prod(q[i][k+1:i]) * q_prime
                if i == len(interfaces)-1:
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


def compute_weight_matrices(pes, interfaces, weights = None):
    masks = {}
    w_path = {}
    X = {}
    if pes[-1].orders is not None:
        ax = plt.figure().add_subplot()
    for i, pe in enumerate(pes):
        # Get the lmr masks, weights, ACCmask, and loadmask of the paths
        masks[i] = get_lmr_masks(pe)
        if weights is None:
            # w, ncycle_true = get_weights(pe.flags, ACCFLAGS, REJFLAGS, verbose = True)
            w = get_weights_staple(i, pe.flags, pe.generation, pe.lmrs, len(pes), ACCFLAGS, REJFLAGS, verbose = True)
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
                        if i != 2:
                            # dir_mask = np.logical_or(pe.dirs == 1, masks[i]["LML"])
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

            # color = next(ax._get_lines.prop_cycler)['color']
                if j == 1 and k == 3 and pe.orders is not None:
                    # idxs = select_with_masks(pe.cyclenumbers, [masks[i]["LML"], accmask, ~loadmask])
                    idxs = select_with_masks(pe.cyclenumbers, [start_cond, end_cond, dir_mask,accmask, ~loadmask])
                    for p in np.random.choice(idxs, min(1, len(idxs))):
                        if len(pe.orders[p][0]) > 1:
                            ax.plot([pe.orders[p][i][0] for i in range(len(pe.orders[p]))], [pe.orders[p][i][0] for i in range(len(pe.orders[p]))], "-x")
                        else:
                            ax.plot([i  for i in range(len(pe.orders[p]))],
                                    [ph[0] for ph in pe.orders[p]], "-x")
                        ax.plot(0, pe.orders[p][0][0], "^",
                                    color=ax.lines[-1].get_color(), ms = 7)
                        ax.plot(len(pe.orders[p]) - 1,
                                pe.orders[p][-1][0], "v",
                                color=ax.lines[-1].get_color(), ms = 7)
                            # plot the first and last point again to highlight start/end phasepoints
                            # it must have the same color as the line for the path
                    if interfaces is not None:
                        for intf in interfaces:
                            ax.axhline(intf, color="k", ls="--", lw=.5)
        if pe.orders is not None:
            plt.tight_layout()
            plt.show()
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

    for i in range(1,len(pes)):
        X[i] += X[i].T

    return X


def compute_weight_matrix(pe, pe_id, interfaces, weights = None):

    # Get the lmr masks, weights, ACCmask, and loadmask of the paths
    masks = get_lmr_masks(pe)
    if weights is None:
        w, ncycle_true = get_weights(pe.flags, ACCFLAGS, REJFLAGS, verbose = False)
        w, ncycle_true = get_weights(pe.flags, pe.dirs, ACCFLAGS, REJFLAGS, verbose = False)
        assert ncycle_true == pe.ncycle
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
                    if pe_id == 1:
                        dir_mask = pe.dirs < 2
                    else:
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
            
                X_path[j][k] = np.sum(select_with_masks(w, [start_cond, end_cond, dir_mask, accmask, ~loadmask]))
            else:
                if j == 1 and k == 0:
                    if pe_id == 1:
                        dir_mask = pe.dirs > 2
                    else:
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

                X_path[j][k] = np.sum(select_with_masks(w, [start_cond, end_cond, dir_mask, accmask, ~loadmask]))
    print(f"sum weights ensemble {pe_id}=", np.sum(X_path))

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
            tot_w += prev_ha*acc_w
            # info for new traj
            acc_index = i
            acc_w = 1
            accepted += 1
            if gg == 'sh' and \
                ((pe_i == 2 and ptype == "RMR") or\
                (pe_i == n_pes-1 and ptype == "LML") or\
                (2 < pe_i < n_pes-1 and ptype in ["LML", "RMR"])) :
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
    tot_w += prev_ha*acc_w

    if verbose:
        print("weights:")
        print("accepted     ",accepted)
        print("rejected     ",rejected)
        print("omitted      ",omitted)
        print("total trajs  ",ntraj)
        print("total weights",np.sum(weights))

    assert omitted == 0
    # ncycle_true = np.sum(weights)
    # miss = len(flags)-1 - ncycle_true
    # for i in range(miss):
    #     assert flags[-(i+1)] in REJFLAGS
        # the reason why this could happen

    return weights


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
            if len(m_piece) <= 100 and len(l_piece)+len(r_piece) <= 150:
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
from json import load
import numpy as np
from .reading import *
import logging
import bisect
from repptis_analysis import *

# Created file and added transition function
# EW - April 2024

# Hard-coded rejection flags found in output files
ACCFLAGS,REJFLAGS = set_flags_ACC_REJ() 

# logger
logger = logging.getLogger(__name__)

def get_transition_probs(pes, interfaces, weights = None, tr=False):
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
    p : dict
        A dictionary with the local crossing probabilities for the PPTIS
        ensemble pe. The keys are the path types (RMR, RML, LMR, LML), and 
        the values are the local crossing probabilities.
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
        l_mins = {}
        l_maxs = {}
        # Get the weights of the RMR, RML, LMR and LML paths
        for pathtype in ["RMR", "RML", "LMR", "LML"]:
            w_path[i][pathtype] = np.sum(select_with_masks(w, [masks[i][pathtype],
                                                            accmask, ~loadmask]))
            l_mins[pathtype] = select_with_masks(pe.lambmins, [masks[i][pathtype], accmask, ~loadmask])
            l_maxs[pathtype] = select_with_masks(pe.lambmaxs, [masks[i][pathtype], accmask, ~loadmask])
        msg = "Weights of the different paths:\n"+f"wRMR = {w_path[i]['RMR']}\n"+\
                f"wRML = {w_path[i]['RML']}\nwLMR = {w_path[i]['LMR']}\n"+\
                f"wLML = {w_path[i]['LML']}"
        print(msg)
        
        # Compute indices of rows with end turn
        # w_path[i]["end"] = [0 for i in interfaces]
        # for int_i, int in enumerate(interfaces):
        #     if int_i < i:
        #         w_path[i]["end"][int_i] = np.asarray([el[0] for el in enumerate(l_mins["LML"]) if el[1] <= int and el[1] >= interfaces[int_i-1]])
        #     elif int_i > i:
        #         w_path[i]["end"][int_i] = np.asarray([el[0] for el in enumerate(l_maxs["RMR"]) if el[1] >= int and el[1] <= interfaces[int_i+1]])
            # else:
            #     w_path[i]["end"][int_i] = (w_path[i]["LML"], w_path[i]["RMR"])

        # idx_ends[i] = {}
        # for pathtype in ["RML", "LML"]:
        #     idx_ends[i][pathtype] = [0 for i in interfaces]
        #     for int_i, int in enumerate(interfaces):
        #         idx_ends[i][pathtype][int_i] = np.asarray([el[0] for el in enumerate(l_mins[pathtype]) if el[1] <= int and el[1] >= interfaces[int_i-1]])
        # for pathtype in ["RMR", "LMR"]:
        #     idx_ends[i][pathtype] = [0 for i in interfaces]
        #     for int_i, int in enumerate(interfaces):
        #         idx_ends[i][pathtype][int_i] = np.asarray([el[0] for el in enumerate(l_maxs[pathtype]) if el[1] >= int and el[1] <= interfaces[int_i+1]])

        w_path[i]["ends"] = np.empty([len(interfaces),len(interfaces)])
        for j in range(len(interfaces)):
            for k in range(len(interfaces)):
                if j == k:
                    w_path[i]["ends"][j][k] = 0
                elif j < k:
                    w_path[i]["ends"][j][k] = np.sum(w, [pe.lambmins <= interfaces[j], pe.lambmaxs >= interfaces[k],
                                                 masks[i]["LMR"], masks[i]["RMR"], accmask, ~loadmask])
                else:
                    w_path[i]["ends"][j][k] = np.sum(w, [pe.lambmins <= interfaces[k], pe.lambmaxs >= interfaces[j],
                                                 masks[i]["LML"], masks[i]["RML"], accmask, ~loadmask])
                    

        # if tr:  # TR reweighting. Note this is not block-friendly TODO
        #     w_path[i]['RMR'] *= 2
        #     w_path[i]['LML'] *= 2
        #     temp = w_path[i]['RML'] + w_path[i]['LMR']
        #     w_path[i]['RML'] = temp
        #     w_path[i]['LMR'] = temp

    p = np.empty([2*len(pes)-3, 2*len(pes)-3])
    for i in range(len(interfaces)):
        for k in range(len(interfaces)):
            if i == k:
                p[i][k] = 0
            elif i < k:
                p_reachedj = np.empty(k-i)
                p_jtillend = np.empty(k-i)
                for j in range(i+1, k+1):
                    p_reachedj.append(np.sum([w.size for w in idx_ends[i]["RMR"][j:]]) / w_path[i]["RMR"])
                    p_jtillend.append((idx_ends[j]["RMR"][k].size + idx_ends[j]["LMR"][k].size) / (w_path[j]["RMR"] + w_path[j]["LMR"]))
                print(p_reachedj)
                print(p_jtillend)
                print(p_reachedj*p_jtillend)
                p[i][k] = np.average(p_reachedj * p_jtillend)
            elif i > k:
                p_reachedj = np.empty(i-k)
                p_jtillend = np.empty(i-k)
                for j in range(k, i):
                    p_reachedj.append(np.sum([w.size for w in idx_ends[i]["LML"][j:]]) / w_path[i]["LML"])
                    p_jtillend.append((idx_ends[j]["RMR"][k].size + idx_ends[j]["LMR"][k].size) / (w_path[j]["RMR"] + w_path[j]["LMR"]))
                print(p_reachedj)
                print(p_jtillend)
                print(p_reachedj*p_jtillend)
                p[i][k] = np.average(p_reachedj * p_jtillend)

    # Get the total weight of paths starting from left, or from right
    wR = w_path['RMR'] + w_path['RML']
    wL = w_path['LMR'] + w_path['LML']
    # And calculate local crossing probabilities
    p = {}
    for pathtype, direc_w in zip(("RMR", "RML", "LMR", "LML"),
                                (wR, wR, wL, wL)):
        p[pathtype] = w_path[pathtype]/direc_w if direc_w != 0 else np.nan
    msg = "Local crossing probabilities:\n"+f"pRMR = {p['RMR']}\n"+\
            f"pRML = {p['RML']}\npLMR = {p['LMR']}\npLML = {p['LML']}"
    print(msg)

    return p
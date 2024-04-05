from json import load
import numpy as np
from .reading import *
import logging
import bisect
from repptis_analysis import *

# Hard-coded rejection flags found in output files
ACCFLAGS,REJFLAGS = set_flags_ACC_REJ() 

# logger
logger = logging.getLogger(__name__)

def get_transition_probs(pes, w = None, tr=False):
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
    for pe in pes:
        # Get the lmr masks, weights, ACCmask, and loadmask of the paths
        masks = get_lmr_masks(pe)
        if w is None:
            w, _ = get_weights(pe.flags, ACCFLAGS, REJFLAGS, verbose = False)
        accmask = get_flag_mask(pe, "ACC")
        loadmask = get_generation_mask(pe, "ld")
        msg = f"Ensemble {pe.name} has {len(w)} paths.\n The total "+\
                    f"weight of the ensemble is {np.sum(w)}\nThe total amount of "+\
                    f"accepted paths is {np.sum(accmask)}\nThe total amount of "+\
                    f"load paths is {np.sum(loadmask)}"
        logger.debug(msg)


    # Get the weights of the RMR, RML, LMR and LML paths
    w_path = np.empty([2*pe.interfaces])
    for pathtype in ["RMR", "RML", "LMR", "LML"]:
        w_path[pathtype] = np.sum(select_with_masks(w, [masks[pathtype],
                                                        accmask, ~loadmask]))
    msg = "Weights of the different paths:\n"+f"wRMR = {w_path['RMR']}\n"+\
            f"wRML = {w_path['RML']}\nwLMR = {w_path['LMR']}\n"+\
            f"wLML = {w_path['LML']}"
    print(msg)
    if tr:  # TR reweighting. Note this is not block-friendly TODO
        w_path['RMR'] = 2*w_path['RMR']
        w_path['LML'] = 2*w_path['LML']
        temp = w_path['RML'] + w_path['LMR']
        w_path['RML'] = temp
        w_path['LMR'] = temp
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
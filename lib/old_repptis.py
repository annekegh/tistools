from .pathlengths import *
from json import load
import numpy as np
from .reading import * 

### CROSSING PROBABILITIES ###
def get_pptis_shortcross_probabilities(pe, inzero=False, lambda_minone=True, verbose=False):
    """
    Calculate short crossing probabilities for a given path ensemble.

    Parameters
    ----------
    pe : :py:class:`.PathEnsemble`
        The path ensemble object.
    inzero : bool, optional
        If True, includes zero ensemble paths. Default is False.
    lambda_minone : bool, optional
        If True, includes lambda_-1 interface. Default is True.
    verbose : bool, optional
        If True, prints detailed information. Default is False.

    Returns
    -------
    tuple
        Short crossing probabilities (pPP, pPN, pNP, pNN).
    """
    # Get the LMR masks
    masks, masknames = get_lmr_masks(pe)
    # Get weights of the paths
    w, _ = get_weights(pe.flags, ACCFLAGS, REJFLAGS, verbose=False)
    # Get acceptance mask
    flagmask = get_flag_mask(pe, "ACC")
    # Get load mask
    load_mask = get_generation_mask(pe, "ld")
    
    if verbose:
        print(f"Ensemble {pe.name} has {len(w)} paths")
        print(f"Total weight of the ensemble: {np.sum(w)}")
        print(f"Total amount of the accepted paths: {np.sum(flagmask)}")
        print(f"Amount of loaded masks is {np.sum(load_mask)}")
    
    # Calculate weights of different paths
    wRMR = np.sum(select_with_masks(w, [masks[masknames.index("RMR")], flagmask, ~load_mask]))
    wRML = np.sum(select_with_masks(w, [masks[masknames.index("RML")], flagmask, ~load_mask]))
    wLMR = np.sum(select_with_masks(w, [masks[masknames.index("LMR")], flagmask, ~load_mask]))
    wLML = np.sum(select_with_masks(w, [masks[masknames.index("LML")], flagmask, ~load_mask]))
    
    if verbose:
        print(f"weights of the different paths: wRMR = {wRMR}, wRML = {wRML}, wLMR = {wLMR}, wLML = {wLML}")
        print(f"sum of weights = {wRMR + wRML + wLMR + wLML}")

    if not inzero:
        # Calculate weights of probabilities starting from left and right
        wR = wRMR + wRML
        wL = wLMR + wLML
        # Calculate probabilities
        pPP = wRMR / wR
        pPN = wRML / wR
        pNP = wLMR / wL
        pNN = wLML / wL
        return pPP, pPN, pNP, pNN
    else:
        # Include zero ensemble paths
        wLSL = np.sum(select_with_masks(w, [masks[masknames.index("L*L")], flagmask, ~load_mask]))
        wRSR = np.sum(select_with_masks(w, [masks[masknames.index("R*R")], flagmask, ~load_mask]))
        
        if verbose:
            print(f"Extra weights in zero ensemble: wLSL = {wLSL}, wRSR = {wRSR}")
            print(f"sum of weights = {wRMR + wRML + wLMR + wLML + wLSL + wRSR}")
        
        # Calculate weights of probabilities starting from left and right
        wR = wRMR + wRML + wRSR
        wL = wLMR + wLML + wLSL
        # Calculate probabilities
        pPP = wRMR / wR
        pPN = wRML / wR
        pNP = wLMR / wL
        pNN = wLML / wL
        return pPP, pPN, pNP, pNN

def get_all_shortcross_probabilities(pathensembles, verbose=True):
    """
    Calculate short crossing probabilities for all path ensembles.

    Parameters
    ----------
    pathensembles : list of :py:class:`.PathEnsemble`
        List of path ensemble objects.
    verbose : bool, optional
        If True, prints detailed information. Default is True.

    Returns
    -------
    tuple
        Short crossing probabilities for all path ensembles (pPP, pPN, pNP, pNN).
    """
    pPP, pPN, pNP, pNN = [], [], [], []
    for i, pe in enumerate(pathensembles):
        inzero = (i == 0)
        pPP_pe, pPN_pe, pNP_pe, pNN_pe = get_pptis_shortcross_probabilities(pe, inzero=inzero, verbose=verbose)
        pPP.append(pPP_pe)
        pPN.append(pPN_pe)
        pNP.append(pNP_pe)
        pNN.append(pNN_pe)
    return pPP, pPN, pNP, pNN

def get_longcross_probabilities(pPP, pPN, pNP, pNN):
    """
    Calculate long crossing probabilities from short crossing probabilities.

    Parameters
    ----------
    pPP, pPN, pNP, pNN : list of float
        Short crossing probabilities.

    Returns
    -------
    tuple
        Long crossing probabilities (P_plus, P_min).
    """
    # Discard the first two elements
    pPP, pPN, pNP, pNN = pPP[2:], pPN[2:], pNP[2:], pNN[2:]
    P_plus, P_min = [1], [1]
    
    for i in range(len(pPP)):
        P_plus.append((pNP[i] * P_plus[i]) / (pNP[i] + pNN[i] * P_min[i]))
        P_min.append((pPN[i] * P_min[i]) / (pNP[i] + pNN[i] * P_min[i]))
    
    return P_plus, P_min

def get_TIS_cross_from_PPTIS_cross(P_plus, pNP):
    """
    Calculate TIS crossing probabilities from PPTIS crossing probabilities.

    Parameters
    ----------
    P_plus : list of float
        Long crossing probabilities.
    pNP : list of float
        Short crossing probabilities.

    Returns
    -------
    list of float
        TIS crossing probabilities.
    """
    return [pNP[1] * P_plus[i] for i in range(len(P_plus))]

def calculate_cross_probabilities(pathensembles, verbose=True):
    """
    Calculate and print TIS and PPTIS crossing probabilities for path ensembles.

    Parameters
    ----------
    pathensembles : list of :py:class:`.PathEnsemble`
        List of path ensemble objects.
    verbose : bool, optional
        If True, prints detailed information. Default is True.

    Returns
    -------
    tuple
        Short and long crossing probabilities (pPP, pPN, pNP, pNN, P_plus, P_min, P_A).
    """
    pPP, pPN, pNP, pNN = get_all_shortcross_probabilities(pathensembles, verbose=verbose)
    P_plus, P_min = get_longcross_probabilities(pPP, pPN, pNP, pNN)
    P_A = get_TIS_cross_from_PPTIS_cross(P_plus, pNP)
    
    for i, pe in enumerate(pathensembles):
        pe_LMR_values = pe.interfaces[0]
        pe_LMR_strings = pe.interfaces[1]
        print("##############################################")
        print(f"Path ensemble: {pathensembles[i].name}")
        print("----------------------------------------------")
        print(f"Interfaces: {pe_LMR_values}")
        print(f"Interfaces: {pe_LMR_strings}")
        print("----------------------------------------------")
        print(f"pPP = {pPP[i]}")
        print(f"pPN = {pPN[i]}")
        print(f"pNP = {pNP[i]}")
        print(f"pNN = {pNN[i]}")
        print("----------------------------------------------")
        print("##############################################")

    print("\nLong crossing probabilities:")
    print("----------------------------------------------")
    for i, (pp, pm, pa) in enumerate(zip(P_plus, P_min, P_A)):
        print(f"P{i+1}_plus = {pp}")
        print(f"P{i+1}_min = {pm}")
        print(f"P{i+1}_A = {pa}")
        print("----------------------------------------------")
    
    return pPP, pPN, pNP, pNN, P_plus, P_min, P_A

### BLOCKAVERAGING ALTERNATIVES TO THE FUNCTIONS ABOVE ###
### AKA: GET AN ERROR BAR ON THE CROSSING PROBABILITIES ###
def calculate_cross_probabilities_blockavg(pathensembles, Nblocks, verbose=True):
    """
    Calculate TIS and PPTIS crossing probabilities with block averaging.

    Parameters
    ----------
    pathensembles : list of :py:class:`.PathEnsemble`
        List of path ensemble objects.
    Nblocks : int
        Number of blocks for block averaging.
    verbose : bool, optional
        If True, prints detailed information. Default is True.

    Returns
    -------
    tuple
        Short and long crossing probabilities with errors (pPP, pPN, pNP, pNN, P_plus, P_min, P_A, pPP_err, pPN_err, pNP_err, pNN_err, P_plus_err, P_min_err, P_A_err).
    """
    # Get short crossing probabilities with block averaging
    pPP, pPN, pNP, pNN, pPP_err, pPN_err, pNP_err, pNN_err, pPPs, pPNs, pNPs, pNNs, blockweights_list = get_all_shortcross_probabilities_blockavg(pathensembles, Nblocks, verbose=verbose)
    # Get long crossing probabilities with block averaging
    P_plus, P_min, P_plus_err, P_min_err, P_plus_list, _ = get_longcross_probabilities_blockavg(pPPs, pPNs, pNPs, pNNs, pPP_err, pPN_err, pNP_err, pNN_err, blockweights_list)
    # Get TIS crossing probabilities with block averaging
    P_A, P_A_err, P_A_rel_err, _ = get_TIS_cross_from_PPTIS_cross_blockavg(P_plus_list, pNPs)

    for i, pe in enumerate(pathensembles):
        pe_LMR_values = pe.interfaces[0]
        pe_LMR_strings = pe.interfaces[1]
        print("##############################################")
        print(f"Path ensemble: {pathensembles[i].name}")
        print("----------------------------------------------")
        print(f"Interfaces: {pe_LMR_values}")
        print(f"Interfaces: {pe_LMR_strings}")
        print("----------------------------------------------")
        print(f"pPP = {pPP[i]} +/- {pPP_err[i]} ({(pPP_err[i]/pPP[i]*100) if pPP[i] != 0 else 0}%)")
        print(f"pPN = {pPN[i]} +/- {pPN_err[i]} ({(pPN_err[i]/pPN[i]*100) if pPN[i] != 0 else 0}%)")
        print(f"pNP = {pNP[i]} +/- {pNP_err[i]} ({(pNP_err[i]/pNP[i]*100) if pNP[i] != 0 else 0}%)")
        print(f"pNN = {pNN[i]} +/- {pNN_err[i]} ({(pNN_err[i]/pNN[i]*100) if pNN[i] != 0 else 0}%)")
        print("----------------------------------------------")
        print("##############################################")

    print("\nLong crossing probabilities PPTIS:")
    print("----------------------------------------------")
    for i, (pp, pm, pa, pp_err, pm_err, pa_err) in enumerate(zip(P_plus, P_min, P_A, P_plus_err, P_min_err, P_A_err)):
        print(f"P{i+1}_plus = {pp} +/- {pp_err} ({(pp_err/pp*100) if pp != 0 else 0}%)")
        print(f"P{i+1}_min = {pm} +/- {pm_err} ({(pm_err/pm*100) if pm != 0 else 0}%)")
        print(f"P{i+1}_A = {pa} +/- {pa_err} ({(pa_err/pa*100) if pa != 0 else 0}%)")
        print("----------------------------------------------")
    
    return pPP, pPN, pNP, pNN, P_plus, P_min, P_A, pPP_err, pPN_err, pNP_err, pNN_err, P_plus_err, P_min_err, P_A_err

def get_all_shortcross_probabilities_blockavg(pathensembles, Nblocks, verbose=True):
    """
    Calculate short crossing probabilities with block averaging for all path ensembles.

    Parameters
    ----------
    pathensembles : list of :py:class:`.PathEnsemble`
        List of path ensemble objects.
    Nblocks : int
        Number of blocks for block averaging.
    verbose : bool, optional
        If True, prints detailed information. Default is True.

    Returns
    -------
    tuple
        Short crossing probabilities with errors for all path ensembles (pPP, pPN, pNP, pNN, pPP_err, pPN_err, pNP_err, pNN_err, pPPs_list, pPNs_list, pNPs_list, pNNs_list, blockweights_list).
    """
    pPP, pPN, pNP, pNN = [], [], [], []
    pPP_err, pPN_err, pNP_err, pNN_err = [], [], [], []
    pPPs_list, pPNs_list, pNPs_list, pNNs_list = [], [], [], []
    blockweights_list = []
    
    for i, pe in enumerate(pathensembles):
        inzero = (i == 0)
        pPP_pe, pPN_pe, pNP_pe, pNN_pe, pPP_pe_err, pPN_pe_err, pNP_pe_err, pNN_pe_err, pPPs, pPNs, pNPs, pNNs, blockweights = get_shortcross_probabilities_blockavg(pe, Nblocks, inzero=inzero, verbose=verbose)
        pPP.append(pPP_pe)
        pPN.append(pPN_pe)
        pNP.append(pNP_pe)
        pNN.append(pNN_pe)
        pPP_err.append(pPP_pe_err)
        pPN_err.append(pPN_pe_err)
        pNP_err.append(pNP_pe_err)
        pNN_err.append(pNN_pe_err)
        pPPs_list.append(pPPs)
        pPNs_list.append(pPNs)
        pNPs_list.append(pNPs)
        pNNs_list.append(pNNs)
        blockweights_list.append(blockweights)
    
    if verbose:
        print(f"blockweights_list = {blockweights_list}")
        print("Total blockweight for each ensemble:")
        for i, blockweights in enumerate(blockweights_list):
            print(f"Path ensemble {i}: {sum(blockweights)}")
    
    return pPP, pPN, pNP, pNN, pPP_err, pPN_err, pNP_err, pNN_err, pPPs_list, pPNs_list, pNPs_list, pNNs_list, blockweights_list

def get_shortcross_probabilities_blockavg(pe, Nblocks, inzero=False, verbose=True):
    """
    Calculate short crossing probabilities with block averaging for a given path ensemble.

    Parameters
    ----------
    pe : :py:class:`.PathEnsemble`
        The path ensemble object.
    Nblocks : int
        Number of blocks for block averaging.
    inzero : bool, optional
        If True, includes zero ensemble paths. Default is False.
    verbose : bool, optional
        If True, prints detailed information. Default is True.

    Returns
    -------
    tuple
        Short crossing probabilities with errors (pPP, pPN, pNP, pNN, pPP_err, pPN_err, pNP_err, pNN_err, pPPs, pPNs, pNPs, pNNs, blockweights).
    """
    pPP, pPN, pNP, pNN = [], [], [], []
    pPP_err, pPN_err, pNP_err, pNN_err = [], [], [], []
    
    # Get the LMR masks
    masks, masknames = get_lmr_masks(pe)
    # Get weights of the paths
    w, _ = get_weights(pe.flags, ACCFLAGS, REJFLAGS, verbose=False)
    # Get acceptance mask
    flagmask = get_flag_mask(pe, "ACC")
    # Get load mask
    load_mask = get_generation_mask(pe, "ld")
    
    if verbose:
        print(f"Ensemble {pe.name} has {len(w)} paths")
        print(f"Total weight of the ensemble: {np.sum(w)}")
        print(f"Total amount of the accepted paths: {np.sum(flagmask)}")
        print(f"Amount of loaded masks is {np.sum(load_mask)}")

    # Create masks that partition the paths into Nblocks blocks of equal size
    L = int(np.floor(len(flagmask) / Nblocks))
    blockmasks = [np.zeros_like(flagmask, dtype=bool) for _ in range(Nblocks)]
    for i in range(Nblocks):
        blockmasks[i][i * L:(i + 1) * L] = True

    # Calculate short crossing probabilities for each block
    wRMRs, wRMLs, wLMRs, wLMLs = [], [], [], []
    if not inzero:
        wRs, wLs = [], []
    else:
        wRs, wLs, wLSLs, wLSRs = [], [], [], []
    
    blockweights = []
    for blockmask in blockmasks:
        blockweight = np.sum(select_with_masks(w, [blockmask, ~load_mask]))
        blockweights.append(blockweight)
        wRMR = np.sum(select_with_masks(w, [masks[masknames.index("RMR")], flagmask, ~load_mask, blockmask]))
        wRML = np.sum(select_with_masks(w, [masks[masknames.index("RML")], flagmask, ~load_mask, blockmask]))
        wLMR = np.sum(select_with_masks(w, [masks[masknames.index("LMR")], flagmask, ~load_mask, blockmask]))
        wLML = np.sum(select_with_masks(w, [masks[masknames.index("LML")], flagmask, ~load_mask, blockmask]))
        wRMRs.append(wRMR)
        wRMLs.append(wRML)
        wLMRs.append(wLMR)
        wLMLs.append(wLML)

        if not inzero:
            wR = wRMR + wRML
            wL = wLMR + wLML
            wRs.append(wR)
            wLs.append(wL)
        else:
            wLSL = np.sum(select_with_masks(w, [masks[masknames.index("L*L")], flagmask, ~load_mask, blockmask]))
            wRSR = np.sum(select_with_masks(w, [masks[masknames.index("R*R")], flagmask, ~load_mask, blockmask]))
            wR = wRMR + wRML + wRSR
            wL = wLMR + wLML + wLSL
            wRs.append(wR)
            wLs.append(wL)
            wLSLs.append(wLSL)
            wLSRs.append(wRSR)

        #print all the weights:
        if verbose:
            print("wRMR = {}".format(wRMR))
            print("wRML = {}".format(wRML))
            print("wLMR = {}".format(wLMR))
            print("wLML = {}".format(wLML))
            if not inzero:
                print("wR = {}".format(wR))
                print("wL = {}".format(wL))
            else:
                print("wLSL = {}".format(wLSL))
                print("wRSR = {}".format(wRSR))
                print("wR = {}".format(wR))
                print("wL = {}".format(wL))

    # Calculate the shortcrossing probabilities for the ensemble
    pPPs = np.array(wRMRs)/np.array(wRs)
    pPNs = np.array(wRMLs)/np.array(wRs)
    pNPs = np.array(wLMRs)/np.array(wLs)
    pNNs = np.array(wLMLs)/np.array(wLs)

    # Calculate the average and the error of the shortcrossing probabilities
    pPP = np.average(pPPs,weights=blockweights)
    pPN = np.average(pPNs,weights=blockweights)
    pNP = np.average(pNPs,weights=blockweights)
    pNN = np.average(pNNs,weights=blockweights)
    pPP_err = np.sqrt(np.cov(pPPs, aweights=blockweights)) 
    pPN_err = np.sqrt(np.cov(pPNs, aweights=blockweights)) 
    pNP_err = np.sqrt(np.cov(pNPs, aweights=blockweights)) 
    pNN_err = np.sqrt(np.cov(pNNs, aweights=blockweights)) 
    print("pPPs, pPNs, pNPs, pNNs shapes: {}, {}, {}, {}".format(
        np.shape(pPPs), np.shape(pPNs), np.shape(pNPs), np.shape(pNNs)))
    print("pPP_err, pPN_err, pNP_err, pNN_err shapes: {}, {}, {}, {}".format(
        np.shape(pPP_err), np.shape(pPN_err), np.shape(pNP_err), 
        np.shape(pNN_err))) 
    return pPP, pPN, pNP, pNN, pPP_err, pPN_err, pNP_err, pNN_err, \
        pPPs, pPNs, pNPs, pNNs, blockweights


def get_longcross_probabilities_blockavg(pPPs, pPNs, pNPs, pNNs, pPP_err, 
                                         pPN_err, pNP_err, pNN_err, 
                                         blockweights_list):
    """
    Calculate long crossing probabilities with block averaging.

    Parameters
    ----------
    pPPs, pPNs, pNPs, pNNs : list of list of float
        Short crossing probabilities for each block.
    pPP_err, pPN_err, pNP_err, pNN_err : list of float
        Errors in short crossing probabilities.
    blockweights_list : list of list of float
        Weights for each block.

    Returns
    -------
    tuple
        Long crossing probabilities with errors (P_plus, P_min, P_plus_err, P_min_err, P_plus_list, P_min_list).
    """
    pPPs = np.array(pPPs).T
    pPNs = np.array(pPNs).T
    pNPs = np.array(pNPs).T
    pNNs = np.array(pNNs).T
    pPP_errs = np.array(pPP_err).T
    pPN_errs = np.array(pPN_err).T
    pNP_errs = np.array(pNP_err).T
    pNN_errs = np.array(pNN_err).T
    
    P_plus_list = []
    P_min_list = []
    
    for pPP, pPN, pNP, pNN in zip(pPPs, pPNs, pNPs, pNNs):
        pPP = pPP[2:]
        pPN = pPN[2:]
        pNP = pNP[2:]
        pNN = pNN[2:]
        
        P_plus = [1]
        P_min = [1]
        
        for i in range(len(pPP)):
            P_plus.append((pNP[i] * P_plus[i]) / (pNP[i] + pNN[i] * P_min[i]))
            P_min.append((pPN[i] * P_min[i]) / (pNP[i] + pNN[i] * P_min[i]))
        
        P_plus_list.append(P_plus)
        P_min_list.append(P_min)
    
    # Calculate the average and the error of the longcross probabilities
    P_plus = np.average(P_plus_list, axis=0)
    P_min = np.average(P_min_list, axis=0)
    P_plus_err = np.std(P_plus_list, axis=0)
    P_min_err = np.std(P_min_list, axis=0)

    return P_plus, P_min, P_plus_err, P_min_err, P_plus_list, P_min_list

def get_TIS_cross_from_PPTIS_cross_blockavg(P_plus_list, pNPs):
    """
    Calculate TIS crossing probabilities with block averaging from PPTIS crossing probabilities.

    Parameters
    ----------
    P_plus_list : list of list of float
        Long crossing probabilities for each block.
    pNPs : list of list of float
        Short crossing probabilities for each block.

    Returns
    -------
    tuple
        TIS crossing probabilities with errors (P_A, P_A_err, P_A_rel_err, P_A_list).
    """
    P_A_list = []
    P_plus_list = np.array(P_plus_list).T
    pNPs = np.array(pNPs)
    
    for P_plus in P_plus_list:
        P_A = [pNP1 * P_plus[i] for i, pNP1 in enumerate(pNPs[1, :])]
        P_A_list.append(P_A)
    
    P_A = np.mean(P_A_list, axis=1)
    P_A_err = np.std(P_A_list, axis=1)
    P_A_rel_err = np.array(P_A_err) / np.array(P_A)
    
    return np.array(P_A), np.array(P_A_err), P_A_rel_err, np.array(P_A_list)

def extract_Pcross_from_retis_html_report(html_report_file):
    """
    Extract crossing probabilities from a RETIS HTML report.

    Parameters
    ----------
    html_report_file : str
        Path to the HTML report file.

    Returns
    -------
    tuple
        Crossing probabilities, their errors, and relative errors (P_cross, P_cross_err, P_cross_relerr).
    """
    with open(html_report_file, 'r') as f:
        lines = f.readlines()
    
    for i, line in enumerate(lines):
        if "<p>The calculated crossing probabilities are:</p>" in line:
            break
    
    P_cross = []
    P_cross_err = []
    P_cross_relerr = []
    
    for j in range(i + 1, len(lines)):
        if ("<tr><td>[" in lines[j]) and ("^+]</td>" in lines[j]):
            P_cross.append(float(lines[j + 1].split(">")[1].split("<")[0]))
            P_cross_err.append(float(lines[j + 2].split(">")[1].split("<")[0]))
            P_cross_relerr.append(float(lines[j + 3].split(">")[1].split("<")[0]))
        if "</table>" in lines[j]:
            break
    
    return P_cross, P_cross_err, P_cross_relerr

def compare_cross_probabilities_blockavg(pPP, pPN, pNP, pNN, P_plus, P_min, P_A, pPP_err, pPN_err, pNP_err, pNN_err, P_plus_err, P_min_err, P_A_err, P_cross, P_cross_err, P_cross_relerr, Nblocks=10):
    """
    Compare RETIS and (RE)PPTIS crossing probabilities with block averaging.

    Parameters
    ----------
    pPP, pPN, pNP, pNN : list of float
        Short crossing probabilities.
    P_plus, P_min, P_A : list of float
        Long crossing probabilities.
    pPP_err, pPN_err, pNP_err, pNN_err : list of float
        Errors in short crossing probabilities.
    P_plus_err, P_min_err, P_A_err : list of float
        Errors in long crossing probabilities.
    P_cross : list of float
        RETIS crossing probabilities.
    P_cross_err : list of float
        Errors in RETIS crossing probabilities.
    P_cross_relerr : list of float
        Relative errors in RETIS crossing probabilities.
    Nblocks : int, optional
        Number of blocks for block averaging. Default is 10.
    """
    import matplotlib.pyplot as plt
    
    # Print a table comparing the short crossing probabilities
    print("Comparison of short crossing probabilities:")
    print("----------------------------------------------")
    print("   \t\t\tREPPTIS\t\t              \t       \t\t  RETIS    \t              ")
    print("--------------------------------------------------------------------------------")
    print("pNP\t\t\tpNP_err\t\tpNP_relerr [%]\t\tP_cross\t\tP_cross_err\t\tP_cross_relerr [%]")
    print("--------------------------------------------------------------------------------")
    for i in range(len(P_cross)):
        print("{:.8e}\t\t{:.8e}\t\t{:.4e}\t\t{:.8e}\t\t{:.8e}\t\t{:.4e}".format(
            pNP[i + 1], pNP_err[i + 1] / np.sqrt(Nblocks), 
            (pNP_err[i + 1] / np.sqrt(Nblocks) / pNP[i + 1]) * 100 if pNP_err[i + 1] != 0 else 0, 
            P_cross[i], P_cross_err[i], P_cross_relerr[i]))
    print("--------------------------------------------------------------------------------")
    print("")

    # Calculate P_A according to RETIS: P_A_RETIS[i] = prod of P_cross up to i
    P_A_RETIS = []
    for i in range(len(P_cross)):
        if i == 0:
            P_A_RETIS.append(P_cross[0])
        else:
            P_A_RETIS.append(P_A_RETIS[i - 1] * P_cross[i])

    # Calculate P_A_RETIS_error[i], which is the error of P_A_RETIS[i]
    P_A_RETIS_error = []
    for i in range(len(P_cross)):
        if i == 0:
            P_A_RETIS_error.append(P_cross_err[0])
        else:
            P_A_RETIS_error.append(P_A_RETIS_error[i - 1] * P_cross[i] + P_A_RETIS[i - 1] * P_cross_err[i])
    
    # Now make a table comparing P_A and P_A_RETIS, and their errors
    print("")
    print("Comparison of P_A_REPPTIS and P_A_RETIS:")
    print("----------------------------------------------")
    for i in range(len(P_cross)):
        print("PPTIS: P({}|0) = {:.8e} +/- {:.8e} ({:.4e}%)\nRETIS: P({}|0) = {:.8e} +/- {:.8e} ({:.4e}%)".format(
            i + 1, P_A[i], P_A_err[i] / np.sqrt(Nblocks), 
            (P_A_err[i] / np.sqrt(Nblocks) / P_A[i]) * 100 if P_A_err[i] != 0 else 0, 
            i + 1, P_A_RETIS[i], P_A_RETIS_error[i], 
            (P_A_RETIS_error[i] / P_A_RETIS[i]) * 100 if P_A_RETIS_error[i] != 0 else 0))
        print("----------------------------------------------")
    print("")

    # Plot P_A_REPPTIS and P_A_RETIS, and save to a PNG file
    fig, ax = plt.subplots()
    ax.errorbar(range(1, len(P_A) + 1), P_A, yerr=P_A_err / np.sqrt(Nblocks), fmt='o', label=r"$p_0^{\pm}P^{+}_{j}$")
    ax.errorbar(range(1, len(P_A_RETIS) + 1), P_A_RETIS, yerr=P_A_RETIS_error, fmt='o', label=r'$P_A(\lambda_{j}|\lambda_{0})$')
    ax.set_xlabel('interface')
    ax.set_ylabel('crossing probability')
    ax.set_title(f"Comparison of long crossing probabilities in RETIS and REPPTIS. Nblocks = {Nblocks}")
    ax.legend()
    fig.tight_layout()
    fig.savefig('P_cross_compared.png')

    # Plot P_A and P_A_RETIS on a logarithmic scale
    fig, ax = plt.subplots()
    ax.errorbar(range(1, len(P_A) + 1), P_A, yerr=P_A_err / np.sqrt(Nblocks), fmt='o', label=r"$p_0^{\pm}P^{+}_{j}$")
    ax.errorbar(range(1, len(P_A_RETIS) + 1), P_A_RETIS, yerr=P_A_RETIS_error, marker='o', label=r'$P_A(\lambda_{j}|\lambda_{0})$')
    ax.set_xlabel('interface')
    ax.set_ylabel('crossing probability')
    ax.set_yscale('log', nonpositive='clip')
    ax.set_title(f"Comparison of long crossing probabilities in RETIS and REPPTIS. Nblocks = {Nblocks}")
    ax.legend()
    fig.tight_layout()
    fig.savefig('P_cross_compared_LOG.png')

    fig, ax = plt.subplots()        
    ax.errorbar(range(1, len(P_cross) + 1), P_cross, yerr=P_cross_err / np.sqrt(Nblocks), color="red", marker='o', linestyle='', capsize=5, capthick=1, elinewidth=1, ecolor='black', barsabove=True, label=r"$P_A(\lambda_{i+1}|\lambda_i}$")
    for i, (pc, pce, pcre) in enumerate(zip(P_cross, P_cross_err, P_cross_relerr)):
        ax.text(i + 1.15, pc, "{:.2f}".format(pc) + r"$\pm$" + "{:.2f}%".format(pcre / np.sqrt(Nblocks)))
    ax.errorbar(range(1, len(pNP[1:]) + 1), pNP[1:], yerr=pNP_err[1:], marker='o', color="blue", linestyle='', capsize=5, capthick=1, elinewidth=1, ecolor='black', barsabove=True, label=r"$p_i^{\pm}$")
    for i, (pc, pce, pcre) in enumerate(zip(pNP[1:], pNP_err[1:], (np.array(pNP_err[1:]) / np.array(pNP[1:])) * 100 if pNP[1:] != 0 else 0)):
        ax.text(i + 1.15, pc, "{:.2f}".format(pc) + r"$\pm$" + "{:.2f}%".format(pcre))
    ax.legend()    
    ax.set_xlabel('interface')
    ax.set_ylabel('crossing probability')
    ax.set_title(f"Short crossing probabilities in REPPTIS and RETIS. Nblocks = {Nblocks}")
    fig.tight_layout()
    fig.savefig('shortcross_compared.png')

    # Save all the data to a pickle file
    import pickle
    with open('Pcross_data.pkl', 'wb') as f:
        pickle.dump([P_A, P_A_err, P_A_RETIS, P_A_RETIS_error, P_cross, P_cross_err, P_cross_relerr, pNP, pNP_err, Nblocks], f)

def compare_cross_probabilities(pPP, pPN, pNP, pNN, P_plus, P_min, P_A, P_cross, P_cross_err, P_cross_relerr):
    """
    Compare RETIS and (RE)PPTIS crossing probabilities.

    Parameters
    ----------
    pPP, pPN, pNP, pNN : list of float
        Short crossing probabilities.
    P_plus, P_min, P_A : list of float
        Long crossing probabilities.
    P_cross : list of float
        RETIS crossing probabilities.
    P_cross_err : list of float
        Errors in RETIS crossing probabilities.
    P_cross_relerr : list of float
        Relative errors in RETIS crossing probabilities.
    """
    import matplotlib.pyplot as plt
    
    print("")
    print("Comparison of long and TIS crossing probabilities:")
    print("----------------------------------------------")
    print("interf\tpNP\t\tP_cross\t\tP_cross_err\tP_cross_relerr")
    for i, (pnp, pa, pc, pce, pcre) in enumerate(zip(pNP[1:], P_A, P_cross, P_cross_err, P_cross_relerr)):
        print("{}:\t {:.10f}\t{:.10f}\t{:.10f}\t{:.10f}".format(i + 1, pnp, pc, pce, pcre))
    print("----------------------------------------------")

    # Calculate P_A according to RETIS: P_A_RETIS[i] = prod of P_cross up to i
    P_A_RETIS = []
    for i in range(len(P_cross)):
        if i == 0:
            P_A_RETIS.append(P_cross[0])
        else:
            P_A_RETIS.append(P_A_RETIS[i - 1] * P_cross[i])

    # Calculate P_A_RETIS_error[i], which is the error of P_A_RETIS[i]
    P_A_RETIS_error = []
    for i in range(len(P_cross)):
        if i == 0:
            P_A_RETIS_error.append(P_cross_err[0])
        else:
            P_A_RETIS_error.append(P_A_RETIS_error[i - 1] * P_cross[i] + P_A_RETIS[i - 1] * P_cross_err[i])
    
    # Now make a table comparing P_A and P_A_RETIS
    print("")
    print("Comparison of P_A_REPPTIS and P_A_RETIS:")
    print("----------------------------------------------")
    print("interf\tP_A_REPPTIS\tP_A_RETIS")
    for i, (pa, par) in enumerate(zip(P_A, P_A_RETIS)):
        print("{}:\t {:.10f}\t{:.10f}".format(i + 1, pa, par))
    print("----------------------------------------------")

    # Plot P_A_REPPTIS and P_A_RETIS, and save to a PNG file
    fig, ax = plt.subplots()
    ax.plot(range(1, len(P_A) + 1), P_A, marker='o', label=r"$p_0^{\pm}P^{+}_{j}$")
    ax.errorbar(range(1, len(P_A_RETIS) + 1), P_A_RETIS, yerr=P_A_RETIS_error, marker='o', label=r'$P_A(\lambda_{j}|\lambda_{0})$')
    ax.set_xlabel('interface')
    ax.set_ylabel('crossing probability')
    ax.set_title("Comparison of long crossing probs in RETIS and REPPTIS")
    ax.legend()
    fig.tight_layout()
    fig.savefig('Pcross_compared.png')

    # Plot P_A and P_A_RETIS on a logarithmic scale
    fig, ax = plt.subplots()
    ax.plot(range(1, len(P_A) + 1), P_A, marker='o', label=r"$p_0^{\pm}P^{+}_{j}$")
    ax.errorbar(range(1, len(P_A_RETIS) + 1), P_A_RETIS, yerr=P_A_RETIS_error, marker='o', label=r'$P_A(\lambda_{j}|\lambda_{0})$')
    ax.set_xlabel('interface')
    ax.set_ylabel('crossing probability')
    ax.set_yscale('log', nonpositive='clip')
    ax.set_title("Comparison of long crossing probs in RETIS and REPPTIS")
    ax.legend()
    fig.tight_layout()
    fig.savefig('Pcross_compared_LOG.png')


    fig, ax = plt.subplots()        
    ax.errorbar(range(1, len(P_cross) + 1), P_cross, yerr=P_cross_err, marker='o', linestyle='', capsize=5, capthick=1, elinewidth=1, ecolor='black', barsabove=True)
    for i, (pc, pce, pcre) in enumerate(zip(P_cross, P_cross_err, P_cross_relerr)):
        ax.text(i + 1.15, pc, "{:.2f}".format(pc) + r"$\pm$" + "{:.2f}".format(pcre))
    ax.set_xlabel('interface')
    ax.set_ylabel('crossing probability')
    ax.set_title("Crossing probabilities in RETIS")
    fig.tight_layout()
    fig.savefig('Pcross_error.png')

    # fig,ax=plt.subplots()
    # ax.errorbar(range(1,len(P_cross)+1),P_cross,yerr=P_cross_err,marker='o')
    # ax.set_xlabel('interface')
    # ax.set_ylabel('crossing probability')
    # ax.set_title('RETIS short crossing probabilities')
    # fig.tight_layout()
    # fig.savefig('Pcross_error.png')

def create_order_distributions(pathensembles, orderparameters, nbins = 50, 
                               verbose = True, flag = "ACC"):
    """
    Creates order parameter distributions for the given path ensembles.
    """
    for pe, op in zip(pathensembles,orderparameters):
        create_order_distribution(pe, op, nbins = nbins, verbose = verbose, 
                                  flag = flag)
 

def create_order_distribution(pe,op,nbins=50,verbose=True,flag="ACC"):
    """
    Plots the distribution of orderparameters for each path mask in the 
    ensemble (accepted paths)
    """
    import matplotlib.pyplot as plt
    # get weights of the paths
    w, _ = get_weights(pe.flags, ACCFLAGS, REJFLAGS)
    # Get load mask
    loadmask = get_generation_mask(pe, "ld")
    # get acc mask
    accmask = get_flag_mask(pe,"ACC")
    if flag == "ACC":
        flagmask = accmask
    elif flag == "REJ":
        flagmask = ~accmask
    else:
        flagmask = get_flag_mask(pe, flag)

    # get the lmr masks
    masks, masknames = get_lmr_masks(pe)
    # Strip the orderparameters of their start and endpoints (not part of ens)
    stripped_op_list = strip_endpoints(op)
    stripped_op = np.array(stripped_op_list,object)
    # Create the distributions
    fig,ax=plt.subplots(nrows = 4, ncols = 4, figsize = (10,10))
    i = 0
    for mask, maskname in zip(masks, masknames):
        axi = ax[i//4,i%4]
        # Select the paths with the mask
        # print("stripped_op.shape = ", stripped_op.shape)
        # print("flagmask.shape = ", flagmask.shape)
        # print("mask.shape = ", mask.shape)
        # print("loadmask.shape = ", loadmask.shape)

        # If you get errors with the shape, check whether your simulation
        # actually finished. It's likely that ensembles up to j were updated
        # in a specific cycle, while the ensembles from j+1 on were not updated 
        # (because the simulation was [forcefully] stopped before the cycle was
        #  finished).
        # TODO: add a check for this, and print a warning if this is the case.
        # It can only be the op that is 1 bigger ...: 

        if len(stripped_op) == len(flagmask) + 1:
            print(' '.join(["WARNING: orderparam is 1 longer than flagmask.\n",
            "This is most likely because the simul was stopped before the\n",
            "cycle was finished. The last order parameter will be ignored."]))
            stripped_op = stripped_op[:-1]
        elif len(stripped_op) != len(flagmask):
            raise Exception("Order parameter and flagmask have different lens.")

        mask_o = select_with_masks(stripped_op, [flagmask,mask,~loadmask])
        mask_w = select_with_masks(w, [flagmask,mask,~loadmask])
        # Flatten
        mask_o, mask_w = get_flat_list_and_weights(mask_o, mask_w)
        # Plot
        if flag == "ACC": # use weights
            axi.hist(mask_o, weights = mask_w, bins = nbins)
        else: # don't use weights
            axi.hist(mask_o, bins = nbins)
        axi.set_title(maskname)
        i += 1
    fig.suptitle("Ensemble {} with intfs {}.\n These are the {} paths.".format(
        pe.name, (pe.interfaces)[0], flag))
    fig.tight_layout()
    fig.savefig("{}_{}_order_distributions.png".format(pe.name, flag))

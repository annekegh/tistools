from .pathlengths import *
from json import load
import numpy as np
from .reading import * 


### CROSSING PROBABILITIES ###
def get_pptis_shortcross_probabilities(pe, inzero=False, lambda_minone=True, 
                                       verbose=False):
    """
    We denote P_a^b by pab, where a is either +(P) or -(N).
    Thus we have 
        pPP: the weight of RMR paths in the ensemble divided by total weight
        pPN: the weight of RML paths in the ensemble divided by total weight
        pNP: the weight of LMR paths in the ensemble divided by total weight
        pNN: the weight of LML paths in the ensemble divided by total weight
    """
    # get the lmr masks
    masks, masknames = get_lmr_masks(pe)
    # get weights of the paths
    w, _ = get_weights(pe.flags, ACCFLAGS, REJFLAGS, verbose = False)
    # get acc mask
    flagmask = get_flag_mask(pe, "ACC")
    # get load mask
    load_mask = get_generation_mask(pe, "ld")
    if verbose:
        print("Ensemble {} has {} paths".format(pe.name, len(w)))
        print("Total weight of the ensemble: {}".format(np.sum(w)))
        print("Total amount of the accepted paths: {}".format(np.sum(flagmask)))
        print("Amount of loaded masks is {}".format(np.sum(load_mask)))
    # get the weights of the different paths
    wRMR = np.sum(select_with_masks(w, [masks[masknames.index("RMR")], 
                                        flagmask, ~load_mask]))
    wRML = np.sum(select_with_masks(w, [masks[masknames.index("RML")], 
                                        flagmask, ~load_mask]))
    wLMR = np.sum(select_with_masks(w, [masks[masknames.index("LMR")], 
                                        flagmask, ~load_mask]))
    wLML = np.sum(select_with_masks(w, [masks[masknames.index("LML")], 
                                        flagmask, ~load_mask]))
    
    if verbose:
        print("weights of the different paths:")
        print("wRMR = {}, wRML = {}, wLMR = {}, wLML = {}".format(wRMR, wRML, 
                                                                  wLMR, wLML))
        print("sum of weights = {}".format(wRMR+wRML+wLMR+wLML))

    # For info: if you have lambda_minone = True, then the 000 pathensemble will
    # have RMR, RML, LMR, LML, L*L, R*R paths. M is just put in the middle of 
    # lambda_minone and lambda_zero. L thus refers to lambda_minone and R to 
    # lambda_zero. I don't know the correct relation with the cross 
    # rate/probability yet. TODO: check this. At the end of the day, we only 
    # want to know pNP for the 000 path ensemble...

    if not inzero:
        # For info: only the 001 ensemble will have no RMR paths, as 
        # L = M = lambda_zero and R = lambda_one. All larger ensemble folders 
        # should have RMR paths. 

        # Calculate the weights of probabilities starting from left and right
        wR = wRMR + wRML
        wL = wLMR + wLML
        # Calculate the probabilities
        pPP = wRMR/wR
        pPN = wRML/wR
        pNP = wLMR/wL
        pNN = wLML/wL
        return pPP, pPN, pNP, pNN
    if inzero:
        # In PPTIS, we also get accepted paths that are of the type 
        # L*L and R*R in the 000 ensemble
        wLSL = np.sum(select_with_masks(w, [masks[masknames.index("L*L")], 
                                            flagmask, ~load_mask]))
        wRSR = np.sum(select_with_masks(w, [masks[masknames.index("R*R")], 
                                            flagmask, ~load_mask]))
        if verbose:
            print("Extra weights in zero ensemble:")
            print("wLSL = {}, wRSR = {}".format(wLSL, wRSR))
            print("sum of weights = {}".format(wRMR+wRML+wLMR+wLML+wLSL+wRSR))
        # Calculate the weights of probabilities starting from left and right
        wR = wRMR + wRML + wRSR
        wL = wLMR + wLML + wLSL
        # Calculate the probabilities
        pPP = wRMR/wR
        pPN = wRML/wR
        pNP = wLMR/wL
        pNN = wLML/wL
        return pPP, pPN, pNP, pNN
    
def get_all_shortcross_probabilities(pathensembles, verbose = True):
    """
    Returns the shortcrossing probabilities for all path ensembles.
    """
    pPP = []
    pPN = []
    pNP = []
    pNN = []
    for i,pe in enumerate(pathensembles):
        if i == 0:
            inzero = True
        else:
            inzero = False
        pPP_pe, pPN_pe, pNP_pe, pNN_pe = \
            get_pptis_shortcross_probabilities(pe,inzero=inzero, 
                                               verbose = verbose)
        pPP.append(pPP_pe)
        pPN.append(pPN_pe)
        pNP.append(pNP_pe)
        pNN.append(pNN_pe)
    return pPP, pPN, pNP, pNN

def get_longcross_probabilities(pPP, pPN, pNP, pNN):
    """
        The crossing probality of an ensemble is given by recursive formulas:
        P_plus[j] = (pNP[j-1]*P_plus[j-1]) / (pNP[j-1]+pNN[j-1]*P_min[j-1])
        P_min[j] = (pPN[j-1]P_min[j-1])/(pNP[j-1]+pNN[j-1]*P_min[j-1])
        where the sum is over j = 1, ..., N and where P_plus[1] = P_min[1] = 1.

        This does not yet include error propagation.
    """
    # Discard the first two elements of the pPP, pPN, pNP, pNN lists, as 
    # the longcross_probabilities only depend on the shortcross probabilities 
    # starting from the second ensemble.
    pPP = pPP[2:]
    pPN = pPN[2:]
    pNP = pNP[2:]
    pNN = pNN[2:]
    # Now we can calculate the longcross probabilities
    P_plus = [1]
    P_min = [1]
    print("len(pPP) = {}".format(len(pPP)))
    for i in range(len(pPP)):
        P_plus.append((pNP[i]*P_plus[i])/(pNP[i]+pNN[i]*P_min[i]))
        P_min.append((pPN[i]*P_min[i])/(pNP[i]+pNN[i]*P_min[i]))
    print("P_plus = {}".format(P_plus))
    return P_plus, P_min

def get_TIS_cross_from_PPTIS_cross(P_plus, pNP):
    """
    The TIS cross probability P_A[j] = pNP[0]*P_plus[j]
    """
    P_A = []
    for i in range(len(P_plus)):
        P_A.append(pNP[1]*P_plus[i])
    return P_A

def calculate_cross_probabilities(pathensembles, verbose=True):
    """
    Calculates and returns the TIS and PPTIS crossing probabilities for the 
    given path ensembles. For each path ensemble, print the shortcrossing 
    probabilities and the TIS and PPTIS crossing probabilities.
    """
    pPP, pPN, pNP, pNN = get_all_shortcross_probabilities(pathensembles, verbose=True)
    P_plus, P_min = get_longcross_probabilities(pPP, pPN, pNP, pNN)
    P_A = get_TIS_cross_from_PPTIS_cross(P_plus,pNP)
    for i, pe in enumerate(pathensembles):
        pe_LMR_values = (pe.interfaces)[0]
        pe_LMR_strings = (pe.interfaces)[1]
        print("##############################################")
        print("Path ensemble: {}".format(pathensembles[i].name))
        print("----------------------------------------------")
        print("Interfaces: {}".format(pe_LMR_values))
        print("Interfaces: {}".format(pe_LMR_strings))
        print("----------------------------------------------")
        print("pPP = {}".format(pPP[i]))
        print("pPN = {}".format(pPN[i]))
        print("pNP = {}".format(pNP[i]))
        print("pNN = {}".format(pNN[i]))
        print("----------------------------------------------")
        print("##############################################")

    print("")
    print("Long crossing probabilities:")
    print("----------------------------------------------")
    for i, (pp, pm, pa) in enumerate(zip(P_plus, P_min, P_A)):
        print("P{}_plus = {}".format(i+1,pp))
        print("P{}_min = {}".format(i+1,pm))
        print("P{}_A = {}".format(i+1,pa))
        print("----------------------------------------------")
    return pPP, pPN, pNP, pNN, P_plus, P_min, P_A

### BLOCKAVERAGING ALTERNATIVES TO THE FUNCTIONS ABOVE ###
### AKA: GET AN ERROR BAR ON THE CROSSING PROBABILITIES ###
def calculate_cross_probabilities_blockavg(pathensembles, Nblocks, verbose=True):
    """
    Calculates and returns the TIS and PPTIS crossing probabilities for the 
    given path ensembles. For each pathensemble, an error is calculated by 
    blockaveraging the short crossing probabilities.
    """
    # Get the short crossing probabilities (PPTIS)
    pPP, pPN, pNP, pNN, pPP_err, pPN_err, pNP_err, pNN_err, \
        pPPs, pPNs, pNPs, pNNs, blockweights_list = \
            get_all_shortcross_probabilities_blockavg(pathensembles, 
                                                      Nblocks, verbose=True)
    # Then we get the longcrossing probabilities (PPTIS)
    P_plus, P_min, P_plus_err, P_min_err, P_plus_list, _ = \
        get_longcross_probabilities_blockavg(pPPs, pPNs, pNPs, pNNs,pPP_err, 
                                             pPN_err, pNP_err, pNN_err, 
                                             blockweights_list)
    # Then we get the RETIS crossing probabilities, as predicted by the 
    # PPTIS crossing probabilities
    P_A, P_A_err, P_A_rel_err, _ = \
        get_TIS_cross_from_PPTIS_cross_blockavg(P_plus_list,pNPs) 

    for i, pe in enumerate(pathensembles):
        pe_LMR_values = (pe.interfaces)[0]
        pe_LMR_strings = (pe.interfaces)[1]
        print("##############################################")
        print("Path ensemble: {}".format(pathensembles[i].name))
        print("----------------------------------------------")
        print("Interfaces: {}".format(pe_LMR_values))
        print("Interfaces: {}".format(pe_LMR_strings))
        print("----------------------------------------------")
        print("pPP = {} +/- {} ({}%)".format(pPP[i], pPP_err[i], 
                                             (pPP_err[i]/pPP[i]*100) if \
                                                pPP[i] != 0 else 0))
        print("pPN = {} +/- {} ({}%)".format(pPN[i], pPN_err[i], 
                                             (pPN_err[i]/pPN[i]*100) if \
                                                pPN[i] != 0 else 0))
        print("pNP = {} +/- {} ({}%)".format(pNP[i], pNP_err[i], 
                                             (pNP_err[i]/pNP[i]*100) if \
                                                pNP[i] != 0 else 0))
        print("pNN = {} +/- {} ({}%)".format(pNN[i], pNN_err[i], 
                                             (pNN_err[i]/pNN[i]*100) if \
                                                pNN[i] != 0 else 0))
        print("----------------------------------------------")
        print("##############################################")

    print("")
    print("Long crossing probabilities PPTIS:")
    print("----------------------------------------------")
    for i, (pp, pm, pa, pp_err, pm_err, pa_err) in \
        enumerate(zip(P_plus, P_min, P_A, P_plus_err, P_min_err, P_A_err)):
        print("P{}_plus = {} +/- {} ({}%)".format(i+1, pp, pp_err, 
                                                  (pp_err/pp*100)) if \
                                                    pp != 0 else 0)
        print("P{}_min = {} +/- {} ({}%)".format(i+1,pm, pm_err, 
                                                 (pm_err/pm*100)) if \
                                                    pm != 0 else 0)
        print("P{}_A = {} +/- {} ({}%)".format(i+1,pa, pa_err, 
                                               (pa_err/pa*100)) if \
                                                pa != 0 else 0)
        print("----------------------------------------------")
    return pPP, pPN, pNP, pNN, P_plus, P_min, P_A, pPP_err, pPN_err, \
        pNP_err, pNN_err, P_plus_err, P_min_err, P_A_err


def get_all_shortcross_probabilities_blockavg(pathensembles, Nblocks, 
                                              verbose=True):
    """
    Calculates and returns the shortcrossing probabilities for the given path 
    ensembles. For each pathensemble, an error is calculated by blockaveraging 
    the short crossing probabilities.
    """
    pPP = []
    pPN = []
    pNP = []
    pNN = []
    pPP_err = []
    pPN_err = []
    pNP_err = []
    pNN_err = []
    pPPs_list = []
    pPNs_list = []
    pNPs_list = []
    pNNs_list = []
    blockweights_list = []
    for i,pe in enumerate(pathensembles):
        if i == 0:
            inzero = True
        else:
            inzero = False
        pPP_pe, pPN_pe, pNP_pe, pNN_pe, pPP_pe_err, pPN_pe_err, pNP_pe_err, \
            pNN_pe_err, pPPs, pPNs, pNPs, pNNs, blockweights = \
                get_shortcross_probabilities_blockavg(pe, Nblocks, 
                                                      inzero=inzero, 
                                                      verbose=verbose)
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
    print("blockweights_list = {}".format(blockweights_list))
    print("Total blockweight for each ensemble:")
    for i, blockweights in enumerate(blockweights_list):
        print("Path ensemble {}: {}".format(i, sum(blockweights)))
    return pPP, pPN, pNP, pNN, pPP_err, pPN_err, pNP_err, pNN_err, pPPs_list, \
        pPNs_list, pNPs_list, pNNs_list, blockweights_list

def get_shortcross_probabilities_blockavg(pe, Nblocks, inzero=False, 
                                          verbose=True):
    """
    Calculates and returns the shortcrossing probabilities for the given path 
    ensemble. For the pathensemble, an error is calculated by blockaveraging. 
    This is done by taking a block of the pathensemble and calculating the 
    shortcrossing probabilities for that block. Then the average is taken over 
    all blocks, and an error is calculated by taking the standard deviation over
    all blocks, dividing by the square root of the number of blocks.
    """
    pPP = []
    pPN = []
    pNP = []
    pNN = []
    pPP_err = []
    pPN_err = []
    pNP_err = []
    pNN_err = []
    # get the lmr masks
    masks, masknames = get_lmr_masks(pe)
    # get the weights of the paths
    w, _ = get_weights(pe.flags, ACCFLAGS, REJFLAGS, verbose = False)
    # get acc mask
    flagmask = get_flag_mask(pe, "ACC")
    # get load mask
    load_mask = get_generation_mask(pe, "ld")
    if verbose:
        print("Ensemble {} has {} paths".format(pe.name, len(w)))
        print("Total weight of the ensemble: {}".format(np.sum(w)))
        print("Total amount of the accepted paths: {}".format(np.sum(flagmask)))
        print("Amount of loaded masks is {}".format(np.sum(load_mask)))

    # Create masks that partition the paths into Nblocks blocks of equal size
    L = int(np.floor(len(flagmask)/Nblocks))
    blockmasks = []
    for i in range(Nblocks):
        blockmask = np.zeros_like(flagmask, dtype=bool)
        blockmask[i*L:(i+1)*L] = True
        blockmasks.append(blockmask)

    # Calculate the shortcrossing probabilities for each block
    wRMRs, wRMLs, wLMRs, wLMLs = [], [], [], []
    if not inzero:
        wRs, wLs = [], []
    else:
        wRs, wLs, wLSLs, wLSRs = [], [], [], []
    blockweights = []
    for blockmask in blockmasks:
        blockweight = np.sum(select_with_masks(w, [blockmask, ~load_mask]))
        blockweights.append(blockweight)
        wRMR = np.sum(select_with_masks(w, [masks[masknames.index("RMR")], 
                                            flagmask, ~load_mask, blockmask]))
        wRML = np.sum(select_with_masks(w, [masks[masknames.index("RML")], 
                                            flagmask, ~load_mask, blockmask]))
        wLMR = np.sum(select_with_masks(w, [masks[masknames.index("LMR")], 
                                            flagmask, ~load_mask, blockmask]))
        wLML = np.sum(select_with_masks(w, [masks[masknames.index("LML")], 
                                            flagmask, ~load_mask, blockmask]))
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
            wLSL = np.sum(select_with_masks(w, 
                                            [masks[masknames.index("L*L")], 
                                             flagmask, ~load_mask, blockmask]))
            wRSR = np.sum(select_with_masks(w, 
                                            [masks[masknames.index("R*R")], 
                                             flagmask, ~load_mask, blockmask]))
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
    The same thing is done as get_longcross_probabilities, but we keep track 
    of the errors as well now. Actually we don't. Err will just be calculated
    by averaging the loncross probabilities of the blocks...
    """
    pPPs = np.array(pPPs).T
    pPNs = np.array(pPNs).T
    pNPs = np.array(pNPs).T
    pNNs = np.array(pNNs).T
    pPP_errs = np.array(pPP_err).T
    pPN_errs = np.array(pPN_err).T
    pNP_errs = np.array(pNP_err).T
    pNN_errs = np.array(pNN_err).T
    print("pPPs.shape = {}".format(pPPs.shape))
    P_plus_list = []
    P_min_list = []
    P_plus_err_list = []
    P_min_err_list = []
    for pPP, pPN, pNP, pNN in zip(pPPs, pPNs, pNPs, pNNs):
        pPP = pPP[2:]
        pPN = pPN[2:]
        pNP = pNP[2:]
        pNN = pNN[2:]
        pPP_errs = pPP_errs[2:]
        pPN_errs = pPN_errs[2:]
        pNP_errs = pNP_errs[2:]
        pNN_errs = pNN_errs[2:]
        # Now we can calculate the longcross probabilities
        P_plus = [1]
        P_min = [1]
        P_plus_err = [0]
        P_min_err = [0]
        print("len(pPP) = {}".format(len(pPP)))
        for i in range(len(pPP)):
            P_plus.append((pNP[i]*P_plus[i])/(pNP[i]+pNN[i]*P_min[i]))
            P_min.append((pPN[i]*P_min[i])/(pNP[i]+pNN[i]*P_min[i]))
            #P_plus_err.append(np.sqrt((P_plus[i]/(pNP[i]+pNN[i]*P_min[i]))**2 * pNP_errs[i]**2 +pNP[i]/(pNP[i]+pNN[i]*P_min[i])**2 * P_plus_err[i]**2 + 
            #                          (pNP[i]*P_plus[i]/(pNP[i]+pNN[i]*P_min[i])**2)**2 * pNN_errs[i]**2 + (pNP[i]*P_plus[i]/(pNP[i]+pNN[i]*P_min[i])**2)**2 * P_min_err[i]**2 + 
            #                          (P_plus[i]*pNP[i]/(pNP[i]+pNN[i]*P_min[i])**2)**2 * pPN_errs[i]**2))
        P_plus_list.append(P_plus)
        P_min_list.append(P_min)
    # Print P_plus_list and P_min_list
    for i in range(len(P_plus_list)):
        print("P_plus_list[{}] = {}".format(i, P_plus_list[i]))
    for i in range(len(P_min_list)):
        print("P_min_list[{}] = {}".format(i, P_min_list[i]))
    

    # Calculate the average and the error of the longcross probabilities
    # Weights cannot be extracted from blockweight_list anymor
    # e, because different ensemble members have different weights in the same 
    # block.. So error should be propagated from the shortcrossing probabilities
    P_plus = np.average(P_plus_list, axis = 0)
    P_min = np.average(P_min_list, axis = 0)
    P_plus_err = np.std(P_plus_list, axis = 0)
    P_min_err = np.std(P_min_list, axis = 0)

    print("P_plus = {}".format(P_plus))

    return P_plus, P_min, P_plus_err, P_min_err, P_plus_list, P_min_list

def get_TIS_cross_from_PPTIS_cross_blockavg(P_plus_list, pNPs):
    P_A_list = []
    P_plus_list = np.array(P_plus_list).T
    pNPs = np.array(pNPs)
    # Print the pNPs nicely formatted with tabs
    for i in range(len(pNPs)):
        print("pNPs[{}] = {}".format(i, pNPs[i]))
    # Print the P_plus_list nicely formatted with tabs
    for i in range(len(P_plus_list)):
        print("P_plus_list[{}] = {}".format(i, P_plus_list[i]))

    print("P_plus_list.shape = {}".format(P_plus_list.shape))
    print("pNPs.shape = {}".format(pNPs.shape))
    for P_plus in P_plus_list:
        P_A = []
        for i, pNP1 in zip(np.arange(len(P_plus)), pNPs[1,:]):
            print("pNP[1] = {}".format(pNP1))
            print("P_plus[i] = {}".format(P_plus[i]))
            P_A.append(pNP1*P_plus[i])
        P_A_list.append(P_A)
    # Print P_A_list, nicely formatted with tabs
    for i in range(len(P_A_list)):
        print("P_A_list[{}] = {}".format(i, P_A_list[i]))
    
    P_A = np.mean(P_A_list, axis = 1)
    print("P_A: {}".format(P_A))
    P_A_err = np.std(P_A_list, axis = 1)
    P_A_rel_err = np.array(P_A_err)/np.array(P_A)
    return np.array(P_A), np.array(P_A_err), P_A_rel_err, np.array(P_A_list)


def extract_Pcross_from_retis_html_report(html_report_file):
    """
    In the table following the line 
    "<p>The calculated crossing probabilities are:</p>"
    in the html report file, the crossing probabilities are given. 
    Crossing probabilities are given right after "<tr><td>[0^+]</td>" lines in 
    the mentioned table, after which the error and relative 
    error are given.

    """
    with open(html_report_file, 'r') as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if "<p>The calculated crossing probabilities are:</p>" in line:
            break
    P_cross = []
    P_cross_err = []
    P_cross_relerr = []
    for j in range(i+1,len(lines)):
        if ("<tr><td>[" in lines[j]) and ("^+]</td>" in lines[j]):
            P_cross.append(float(lines[j+1].split(">")[1].split("<")[0]))
            P_cross_err.append(float(lines[j+2].split(">")[1].split("<")[0]))
            P_cross_relerr.append(float(lines[j+3].split(">")[1].split("<")[0]))
        if "</table>" in lines[j]:
            break
    return P_cross, P_cross_err, P_cross_relerr

def compare_cross_probabilities_blockavg(pPP, pPN, pNP, pNN, P_plus, P_min, 
                                         P_A, pPP_err, pPN_err, pNP_err, 
                                         pNN_err, P_plus_err, P_min_err, 
                                         P_A_err, P_cross, P_cross_err, 
                                         P_cross_relerr, Nblocks=10):
    """
    Comparison of RETIS and (RE)PPTIS results. 
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
            pNP[i+1], pNP_err[i+1]/np.sqrt(Nblocks), 
            (pNP_err[i+1]/np.sqrt(Nblocks)/pNP[i+1])*100 if \
                pNP_err[i+1] != 0 else 0, P_cross[i], P_cross_err[i], 
                P_cross_relerr[i]))
    print("--------------------------------------------------------------------------------")
    print("")

    # Calculate P_A according to RETIS: P_A_RETIS[i] = prod of P_cross up to i
    P_A_RETIS = []
    for i in range(len(P_cross)):
        if i == 0:
            P_A_RETIS.append(P_cross[0])
        else:
            P_A_RETIS.append(P_A_RETIS[i-1]*P_cross[i])

    # Calculate P_A_RETIS_error[i], which is the error of P_A_RETIS[i]
    P_A_RETIS_error = []
    for i in range(len(P_cross)):
        if i == 0:
            P_A_RETIS_error.append(P_cross_err[0])
        else:
            P_A_RETIS_error.append(P_A_RETIS_error[i-1]*P_cross[i] + \
                                   P_A_RETIS[i-1]*P_cross_err[i])
    
    # Now make a table comparing P_A and P_A_RETIS, and their errors
    print("")
    print("Comparison of P_A_REPPTIS and P_A_RETIS:")
    print("----------------------------------------------")
    for i in range(len(P_cross)):
        print("PPTIS: P({}|0) = {:.8e} +/- {:.8e} ({:.4e}%)\nRETIS: P({}|0) = {:.8e} +/- {:.8e} ({:.4e}%)".format(i+1, 
            P_A[i], P_A_err[i]/np.sqrt(Nblocks), (P_A_err[i]/np.sqrt(Nblocks)/P_A[i])*100 if P_A_err[i] != 0 else 0, i+1, P_A_RETIS[i], 
            P_A_RETIS_error[i], (P_A_RETIS_error[i]/P_A_RETIS[i])*100 if P_A_RETIS_error[i] != 0 else 0))
        print("----------------------------------------------")
    print("")


    # Plot P_A_REPPTIS and P_A_RETIS, and save to a PNG file, 
    fig,ax=plt.subplots()
    ax.errorbar(range(1,len(P_A)+1), P_A, yerr=P_A_err/np.sqrt(Nblocks), 
                fmt='o', label=r"$p_0^{\pm}P^{+}_{j}$")
    ax.errorbar(range(1,len(P_A_RETIS)+1), P_A_RETIS, yerr=P_A_RETIS_error, 
                fmt='o', label=r'$P_A(\lambda_{j}|\lambda_{0})$')
    ax.set_xlabel('interface')
    ax.set_ylabel('crossing probability')
    ax.set_title("Comparison of long crossing probabilities in RETIS" + \
                 " and REPPTIS. Nblocks = {}".format(Nblocks))
    ax.legend()
    fig.tight_layout()
    fig.savefig('P_cross_compared.png')

    # Plot P_A and P_A_RETIS on a logarithmic scale
    fig,ax=plt.subplots()
    #ax.plot(range(1,len(P_A)+1),P_A,marker='o',label=r"$p_0^{\pm}P^{+}_{j}$")
    ax.errorbar(range(1,len(P_A)+1), P_A, yerr=P_A_err/np.sqrt(Nblocks), 
                fmt='o', label=r"$p_0^{\pm}P^{+}_{j}$")
    ax.errorbar(range(1,len(P_A_RETIS)+1),P_A_RETIS,yerr=P_A_RETIS_error, 
                marker='o',label=r'$P_A(\lambda_{j}|\lambda_{0})$')
    ax.set_xlabel('interface')
    ax.set_ylabel('crossing probability')
    ax.set_yscale('log',nonpositive='clip')
    ax.set_title("Comparison of long crossing probabilities in RETIS " + \
                 "and REPPTIS. Nblocks = {}".format(Nblocks))
    ax.legend()
    fig.tight_layout()
    fig.savefig('P_cross_compared_LOG.png')


    fig,ax=plt.subplots()        
    ax.errorbar(range(1,len(P_cross)+1), P_cross, 
                yerr=P_cross_err/np.sqrt(Nblocks), color="red", marker='o',
                linestyle='', capsize=5,capthick=1,elinewidth=1,ecolor='black', 
                barsabove=True,label=r"$P_A(\lambda_{i+1}|\lambda_i}$")
    for i, (pc, pce, pcre) in enumerate(zip(P_cross, P_cross_err, 
                                            P_cross_relerr)):
        ax.text(i+1.15,pc,"{:.2f}".format(pc)+r"$\pm$"+"{:.2f}%".format(
            pcre/np.sqrt(Nblocks)))
    ax.errorbar(range(1,len(pNP[1:])+1), pNP[1:], yerr=pNP_err[1:], marker='o', 
                color="blue",linestyle='', capsize=5,capthick=1, elinewidth=1, 
                ecolor='black',barsabove=True,label=r"$p_i^{\pm}$")
    for i, (pc, pce, pcre) in \
        enumerate(zip(pNP[1:], pNP_err[1:], 
                      (np.array(pNP_err[1:])/np.array(pNP[1:]))*100 if \
                        pNP[1:] != 0 else 0)):
        ax.text(i+1.15,pc,"{:.2f}".format(pc)+r"$\pm$"+"{:.2f}%".format(pcre))
    ax.legend()    
    ax.set_xlabel('interface')
    ax.set_ylabel('crossing probability')
    ax.set_title("Short crossing probabilities in REPPTIS and " + \
                 "RETIS. Nblocks = {}".format(Nblocks))
    fig.tight_layout()
    fig.savefig('shortcross_compared.png')

    # fig,ax=plt.subplots()
    # ax.errorbar(range(1,len(P_cross)+1),P_cross,yerr=P_cross_err,marker='o')
    # ax.set_xlabel('interface')
    # ax.set_ylabel('crossing probability')
    # ax.set_title('RETIS short crossing probabilities')
    # fig.tight_layout()
    # fig.savefig('Pcross_error.png')

    # Now save all the data to a pickle file. 
    # P_A, P_A_err, P_A_RETIS, P_A_RETIS_error, P_cross, 
    # P_cross_err, P_cross_relerr, pNP, pNP_err, pNP_relerr, Nblocks
    import pickle
    with open('Pcross_data.pkl', 'wb') as f:
        pickle.dump([P_A, P_A_err, P_A_RETIS, P_A_RETIS_error, P_cross, 
                     P_cross_err, P_cross_relerr, 
        pNP, pNP_err, Nblocks], f)


def compare_cross_probabilities(pPP, pPN, pNP, pNN, P_plus, P_min, P_A, P_cross,
                                P_cross_err, P_cross_relerr):
    """
    Compare RETIS and (RE)PPTIS
    """
    import matplotlib.pyplot as plt
    print("")
    print("Comparison of long and TIS crossing probabilities:")
    print("----------------------------------------------")
    print("interf\tpNP\t\tP_cross\t\tP_cross_err\tP_cross_relerr")
    for i, (pnp, pa, pc, pce, pcre) in enumerate(zip(pNP[1:], P_A, P_cross, 
                                                     P_cross_err, 
                                                     P_cross_relerr)):
        print("{}:\t {:.10f}\t{:.10f}\t{:.10f}\t{:.10f}".format(i+1,pnp,pc,
                                                                pce,pcre))
    print("----------------------------------------------")

    # Calculate P_A according to RETIS: P_A_RETIS[i] = prod of P_cross up to i
    P_A_RETIS = []
    for i in range(len(P_cross)):
        if i == 0:
            P_A_RETIS.append(P_cross[0])
        else:
            P_A_RETIS.append(P_A_RETIS[i-1]*P_cross[i])

    # Calculate P_A_RETIS_error[i], which is the error of P_A_RETIS[i]
    P_A_RETIS_error = []
    for i in range(len(P_cross)):
        if i == 0:
            P_A_RETIS_error.append(P_cross_err[0])
        else:
            P_A_RETIS_error.append(P_A_RETIS_error[i-1]*P_cross[i] + \
                                   P_A_RETIS[i-1]*P_cross_err[i])
    
    # Now make a table comparing P_A and P_A_RETIS
    print("")
    print("Comparison of P_A_REPPTIS and P_A_RETIS:")
    print("----------------------------------------------")
    print("interf\tP_A_REPPTIS\tP_A_RETIS")
    for i, (pa, par) in enumerate(zip(P_A, P_A_RETIS)):
        print("{}:\t {:.10f}\t{:.10f}".format(i+1,pa,par))
    print("----------------------------------------------")

    # Plot P_A_REPPTIS and P_A_RETIS, and save to a PNG file, 
    # with the name "Pcross_compared.png", and with nice labels
    fig,ax=plt.subplots()
    ax.plot(range(1,len(P_A)+1),P_A,marker='o',label=r"$p_0^{\pm}P^{+}_{j}$")
    ax.errorbar(range(1,len(P_A_RETIS)+1),P_A_RETIS,yerr=P_A_RETIS_error,
                marker='o',label=r'$P_A(\lambda_{j}|\lambda_{0})$')
    ax.set_xlabel('interface')
    ax.set_ylabel('crossing probability')
    ax.set_title("Comparison of long crossing probs in RETIS and REPPTIS")
    ax.legend()
    fig.tight_layout()
    fig.savefig('Pcross_compared.png')

    # Plot P_A and P_A_RETIS on a logarithmic scale, 
    # where we also plot the error bars for P_A_RETIS
    fig,ax=plt.subplots()
    ax.plot(range(1,len(P_A)+1),P_A,marker='o',label=r"$p_0^{\pm}P^{+}_{j}$")
    ax.errorbar(range(1,len(P_A_RETIS)+1),P_A_RETIS,yerr=P_A_RETIS_error, 
                marker='o',label=r'$P_A(\lambda_{j}|\lambda_{0})$')
    ax.set_xlabel('interface')
    ax.set_ylabel('crossing probability')
    ax.set_yscale('log',nonpositive='clip')
    ax.set_title("Comparison of long crossing probs in RETIS and REPPTIS")
    ax.legend()
    fig.tight_layout()
    fig.savefig('Pcross_compared_LOG.png')


    fig,ax=plt.subplots()        
    ax.errorbar(range(1,len(P_cross)+1),P_cross,yerr=P_cross_err,marker='o',
                linestyle='', capsize=5,capthick=1,elinewidth=1, 
                ecolor='black',barsabove=True)
    for i, (pc, pce, pcre) in enumerate(zip(P_cross, 
                                            P_cross_err, P_cross_relerr)):
        ax.text(i+1.15,pc,"{:.2f}".format(pc)+r"$\pm$"+"{:.2f}".format(pcre))
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

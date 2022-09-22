from json import load
import numpy as np
from .reading import *

# Hard-coded rejection flags found in output files
ACCFLAGS,REJFLAGS = set_flags_ACC_REJ() 

def get_lmr_masks(pe, masktype="all"):
    """
    Returns boolean arrays
    """

    # Path type masks
    rml = pe.lmrs == "RML"
    rmr = pe.lmrs == "RMR"
    lmr = pe.lmrs == "LMR"
    lml = pe.lmrs == "LML"
    sss = pe.lmrs == "***"
    ssl = pe.lmrs == "**L"
    ssr = pe.lmrs == "**R"
    sms = pe.lmrs == "*M*"
    sml = pe.lmrs == "*ML"
    smr = pe.lmrs == "*MR"
    lss = pe.lmrs == "L**"
    lsl = pe.lmrs == "L*L"
    lms = pe.lmrs == "LM*"
    rss = pe.lmrs == "R**"
    rsr = pe.lmrs == "R*R"
    rms = pe.lmrs == "RM*"

    if masktype == "all":
        return [rml,rmr,lmr,lml,sss,ssl,ssr,sms,sml,smr,lss,lsl,lms,
                rss,rsr,rms], \
                ["RML","RMR","LMR","LML","***","**L","**R","*M*","*ML",
                "*MR","L**","L*L","LM*","R**","R*R","RM*"]
    elif masktype == "rml":
        return rml
    elif masktype == "rmr":
        return rmr
    elif masktype == "lmr":
        return lmr
    elif masktype == "lml":
        return lml
    elif masktype == "sss":
        return sss
    elif masktype == "ssl":
        return ssl
    elif masktype == "ssr":
        return ssr
    elif masktype == "sms":
        return sms
    elif masktype == "sml":
        return sml
    elif masktype == "smr":
        return smr
    elif masktype == "lss":
        return lss
    elif masktype == "lsl":
        return lsl
    elif masktype == "lms":
        return lms
    elif masktype == "rss":
        return rss
    elif masktype == "rsr":
        return rsr
    elif masktype == "rms":
        return rms
    else:
        raise ValueError("Invalid masktype")


def get_acc_mask(pe):
    """
    Returns boolean array
    """
    acc = pe.flags == "ACC"
    return acc

def get_flag_mask(pe, status):
    """
    Returns boolean array
    """
    flagmask = pe.flags == status 
    return flagmask

def get_generation_mask(pe, generation):
    """
    Returns boolean array
    """
    genmask = pe.generation == generation
    return genmask

def select_with_masks(A, masks):
    """
    Returns the elements of the array A that are True in all of 
    the masks, where each mask is a similarly sized boolean array.
    
    A: np.array
    masks: list of masks, where each mask is a boolean array 
            with the same shape as A
    """
    # first check whether masks are for the same object array, 
    # meaning they must have the same size as  A
    for mask in masks:
        assert mask.shape == A.shape
    # Start with full True array, then take unions with masks
    union_mask = np.full(A.shape, True)
    for mask in masks:
        union_mask = union_mask & mask
    return A[union_mask]

def plot_pathlength_distributions_separately(pe, nbins = 50, save=True, dpi=500,\
     do_pdf = False, fn = "", status = "ACC", force_bins = False, xmin = 0):
    """
    Plots the pathlength distributions for the given path ensemble.
    """
    # get pathlengths
    pl = pe.lengths
    # get weights of the paths
    w, ncycle_true = get_weights(pe.flags, ACCFLAGS, REJFLAGS)
    # get acc mask
    assert (status == "ACC") or (status == "REJ") or (status in REJFLAGS)
    if status == "ACC":
        flag_mask = get_acc_mask(pe)
    elif status == "REJ":
        flag_mask = ~get_acc_mask(pe)
    else:
        flag_mask = get_flag_mask(pe, status)
    ncycle_flag = np.sum(flag_mask)
    # get the lmr masks
    masks, mask_names = get_lmr_masks(pe)
    # get the load mask
    load_mask = get_generation_mask(pe, "ld")
    # Plot the pathlength distributions for the different masks
    for mask, maskname in zip(masks,mask_names):
        pl_mask = select_with_masks(pl, [mask, flag_mask, ~load_mask])
        w_mask = select_with_masks(w, [mask, flag_mask, ~load_mask])
    
        if pl_mask.tolist(): # If there are any paths in the mask, we check for better binsize
            if nbins > np.max(pl_mask):
                if force_bins == False:
                    print('\n'.join(("The amount of bins is larger than the maximum pathlength in the mask.",\
                    "Setting nbins to {}".format(np.max(pl_mask)))))
                    nbins_use = np.max(pl_mask)
                else:
                    print('\n'.join(("The amount of bins is larger than the maximum pathlength in the mask.",\
                    "set force_bins=False to allow for variable bin sizes")))
                    nbins_use = nbins
            else:
                nbins_use = nbins
        else:
            nbins_use = nbins
            # If we want to plot rejected paths, we give all paths unit weight...
        if status != "ACC":
            w_mask = np.ones_like(w_mask)
        plot_pathlength_distribution(pl_mask, w_mask, ncycle_flag, maskname, \
            name=pe.name, nbins=nbins_use, save=save, dpi=dpi, do_pdf=do_pdf, \
            fn=fn, status = status, xmin = xmin)

def plot_pathlength_distribution(pl, w, ncycle, maskname, nbins=50, \
    name="", save=True, dpi=500, do_pdf=False, fn="", status = "ACC", xmin = 0):
    import matplotlib.pyplot as plt
    ncycle_mask = len(w)
    hist, bin_centers = get_pathlength_distribution(pl, w, nbins)
    fig,ax = plt.subplots()
    ax.bar(bin_centers, hist, width=bin_centers[-1]/nbins)
    ax.set_xlabel("Pathlength ({} bins, bwidth = {})".format(nbins, \
        np.round(bin_centers[-1]/nbins,2)))
    ax.set_ylabel("Counts")
    ax.set_title("Pathlength distribution for {} ensemble\n({} of {} paths with status {}) [w = {}]"\
        .format(name, ncycle_mask, ncycle, maskname,str(np.sum(hist))))
    ax.set_xlim(left=xmin)
    fig.tight_layout()
    if save:
        fig.savefig(fn+"pathlength_distribution_{}_{}_{}.png".format(name, maskname, status), dpi=dpi)
        if do_pdf:
            fig.savefig(fn+"pathlength_distribution_{}_{}_{}.pdf".format(name, maskname, status))
    else:
        fig.show()

def get_pathlength_distribution(pl, w, nbins=50):
    pl = np.array([el-2 for el in pl])
    hist, bin_edges = np.histogram(pl, bins=nbins, weights=w)
    bin_centers = (bin_edges[1:] + bin_edges[:-1])/2
    return hist, bin_centers

def plot_pathlength_distributions_together(pe, nbins = 50, save=True, dpi=500, do_pdf=False, fn="",\
     status = "ACC", force_bins=False, xmin = 0):
    """
    Plots the pathlength distributions for the given path ensemble.
    """
    import matplotlib.pyplot as plt
    # We don't want load paths:
    load_mask = get_generation_mask(pe, "ld")
    # get pathlengths
    pl = pe.lengths
    # get weights of the paths
    w, ncycle_true = get_weights(pe.flags, ACCFLAGS, REJFLAGS)
    # get acc mask
    assert (status == "ACC") or (status == "REJ") or (status in REJFLAGS)
    if status == "ACC":
        flagmask = get_acc_mask(pe)
    if status == "REJ":
        flagmask = ~get_acc_mask(pe)
    else: 
        flagmask = get_flag_mask(pe, status)
    ncycle_flag = np.sum(flagmask)
    # get the lmr masks
    masks, mask_names = get_lmr_masks(pe)
    # Plot the pathlength distributions for the different masks
    fig,ax = plt.subplots(nrows=4, ncols=4, figsize=(12,12))
    i = 0
    for mask, maskname in zip(masks,mask_names):
        pl_mask = select_with_masks(pl, [mask, flagmask, ~load_mask])
        w_mask = select_with_masks(w, [mask, flagmask, ~load_mask])
        if pl_mask.tolist(): # If there are any paths in the mask, we check for better binsize
            if nbins > np.max(pl_mask):
                if force_bins == False:
                    print('\n'.join(("The amount of bins is larger than the maximum pathlength in the mask.",\
                    "Setting nbins to {}".format(np.max(pl_mask)))))
                    nbins_use = np.max(pl_mask)
                else:
                    print('\n'.join(("The amount of bins is larger than the maximum pathlength in the mask.",\
                    "set force_bins=False to allow for variable bin sizes")))
                    nbins_use = nbins
            else: nbins_use = nbins
        else: nbins_use = nbins
        # If we want to plot rejected paths, we give all paths unit weight...
        if status != "ACC":
            w_mask = np.ones_like(w_mask)
        ncycle_mask = len(w_mask)
        hist, bin_centers = get_pathlength_distribution(pl_mask, w_mask, nbins_use)
        ax[i//4,i%4].bar(bin_centers, hist, width=bin_centers[-1]/nbins_use)
        ax[i//4,i%4].set_xlabel("Pathlength ({} bins, bwidth = {})".format(nbins_use,\
             np.round(bin_centers[-1]/nbins_use, 2)))
        ax[i//4,i%4].set_ylabel("Counts")
        ax[i//4,i%4].set_title("{} of {} paths\n status {} [w = {}]"\
            .format(ncycle_mask, ncycle_flag, maskname,str(np.sum(hist))))
        ax[i//4,i%4].set_xlim(left=xmin)
        i += 1
    if status == "ACC":
        fig.suptitle("Pathlength distributions for {} ensemble, with {} {} paths of {} total."\
            .format(pe.name, ncycle_flag, status, ncycle_true),fontsize=16)
    else:
        fig.suptitle("Pathlength distributions for {} paths, with {} {} paths of {} total."\
            .format(pe.name, ncycle_flag, status, ncycle_true),fontsize=16)
    fig.tight_layout()
    if save:
        fig.savefig(fn+"{}_{}_pathlen_distrib.png".format(pe.name, status), dpi=dpi)
        if do_pdf:
            fig.savefig(fn+"{}_{}_pathlen_distrib.pdf".format(pe.name, status))
    else:
        fig.show()


def create_pathlength_distributions(pathensembles, nbins=50, save=True, dpi=500, do_pdf=False, \
    fn="",plot_separately=False, status = "ACC", force_bins=False, xmin=0):
    """
    Creates the pathlength distributions for the given path ensembles.
    """
    print(pathensembles)
    for pe in pathensembles:
        if plot_separately:
            plot_pathlength_distributions_separately(pe, nbins=nbins, save=save, dpi=dpi,\
                 do_pdf=do_pdf, fn=fn, status=status, force_bins=force_bins, xmin=xmin)
        plot_pathlength_distributions_together(pe, nbins=nbins, save=save, dpi=dpi,\
             do_pdf=do_pdf, fn=fn, status=status, force_bins=force_bins, xmin=xmin)

def get_pptis_shortcross_probabilities(pe,inzero=False,lambda_minone=True,verbose=False):
    """
    We denote P_a^b by pab, where a is either +(P) or -(N).
    Thus we have 
        pPP which is the weight of RMR paths in the ensemble divided by total weight
        pPN which is the weight of RML paths in the ensemble divided by total weight
        pNP which is the weight of LMR paths in the ensemble divided by total weight
        pNN which is the weight of LML paths in the ensemble divided by total weight
    """
    # get the lmr masks
    masks, masknames = get_lmr_masks(pe)
    # get weights of the paths
    w, _ = get_weights(pe.flags, ACCFLAGS, REJFLAGS, verbose = False)
    # get acc mask
    flagmask = get_acc_mask(pe)
    # get load mask
    load_mask = get_generation_mask(pe, "ld")
    if verbose:
        print("Ensemble {} has {} paths".format(pe.name, len(w)))
        print("The total weight of the ensemble is {}".format(np.sum(w)))
        print("The total amount of the accepted paths is {}".format(np.sum(flagmask)))
        print("Amount of loaded masks is {}".format(np.sum(load_mask)))
    # get the weights of the different paths
    wRMR = np.sum(select_with_masks(w, [masks[masknames.index("RMR")], flagmask, ~load_mask]))
    wRML = np.sum(select_with_masks(w, [masks[masknames.index("RML")], flagmask, ~load_mask]))
    wLMR = np.sum(select_with_masks(w, [masks[masknames.index("LMR")], flagmask, ~load_mask]))
    wLML = np.sum(select_with_masks(w, [masks[masknames.index("LML")], flagmask, ~load_mask]))
    
    if verbose:
        print("weights of the different paths:")
        print("wRMR = {}, wRML = {}, wLMR = {}, wLML = {}".format(wRMR, wRML, wLMR, wLML))
        print("sum of weights = {}".format(wRMR+wRML+wLMR+wLML))

    # For info: if you have lambda_minone = True, then the 000 pathensemble will have RMR, RML,
    # LMR, LML, L*L, R*R paths. M is just put in the middle of lambda_minone and lambda_zero. 
    # L thus refers to lambda_minone and R to lambda_zero. I don't know the correct relation with 
    # the cross rate/probability yet. TODO: check this. At the end of the day, we only want to 
    # know pNP for the 000 path ensemble...

    if not inzero:
        # For info: only the 001 ensemble will have no RMR paths, as L = M = lambda_zero and
        # R = lambda_one. All larger ensemble folders should have RMR paths. 
        # Now we can calculate the weights of probabilities starting from the left and right
        wR = wRMR + wRML
        wL = wLMR + wLML
        # Now we can calculate the probabilities
        pPP = wRMR/wR
        pPN = wRML/wR
        pNP = wLMR/wL
        pNN = wLML/wL
        return pPP, pPN, pNP, pNN
    if inzero:
        # In PPTIS, we also get accepted paths that are of the type 
        # L*L and R*R in the 000 ensemble
        wLSL = np.sum(select_with_masks(w, [masks[masknames.index("L*L")], flagmask, ~load_mask]))
        wRSR = np.sum(select_with_masks(w, [masks[masknames.index("R*R")], flagmask, ~load_mask]))
        if verbose:
            print("Extra weights in zero ensemble:")
            print("wLSL = {}, wRSR = {}".format(wLSL, wRSR))
            print("sum of weights = {}".format(wRMR+wRML+wLMR+wLML+wLSL+wRSR))
        # Now we can calculate the weights of probabilities starting from the left and right
        wR = wRMR + wRML + wRSR
        wL = wLMR + wLML + wLSL
        # Now we can calculate the probabilities
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
        pPP_pe, pPN_pe, pNP_pe, pNN_pe = get_pptis_shortcross_probabilities(pe,inzero=inzero, verbose = verbose)
        pPP.append(pPP_pe)
        pPN.append(pPN_pe)
        pNP.append(pNP_pe)
        pNN.append(pNN_pe)
    return pPP, pPN, pNP, pNN

def get_longcross_probabilities(pPP, pPN, pNP, pNN):
    """
    The crossing probality of an ensemble is given by the following recursive formulas:
        P_plus[j] = (pNP[j-1]*P_plus[j-1]) / (pNP[j-1]+pNN[j-1]*P_min[j-1])
        P_min[j] = (pPN[j-1]P_min[j-1])/(pNP[j-1]+pNN[j-1]*P_min[j-1])
        where the sum is over j = 1, ..., N and where P_plus[1] = 1 and P_min[1] = 1.
    """
    # First we discard the first two elements of the pPP, pPN, pNP, pNN lists, as these are
    # as the longcross_probabilities only depend on the shortcross probabilities starting from
    # the second ensemble (002 has the LMR, LML, RMR, and RML paths  with M = lambda_one, so
    # from this we can get get P_plus[2] = P[_0^2|^1_0]). And P_plus[1] = 1 and P_min[1] = 1 
    # are the initial conditions.
    pPP = pPP[2:]
    pPN = pPN[2:]
    pNP = pNP[2:]
    pNN = pNN[2:]
    # Now we can calculate the longcross probabilities
    P_plus = [1]
    P_min = [1]
    for i in range(len(pPP)):
        P_plus.append((pNP[i]*P_plus[i])/(pNP[i]+pNN[i]*P_min[i]))
        P_min.append((pPN[i]*P_min[i])/(pNP[i]+pNN[i]*P_min[i]))

    return P_plus, P_min

    # N = len(pPP)
    # P_plus = np.zeros(N)
    # P_min = np.zeros(N)
    # P_plus[1] = 1
    # P_min[1] = 1
    # for j in range(2,N):
    #     # P_plus[j] = (pNP[j-1]*P_plus[j-1]) / (pNP[j-1]+pNN[j-1]*P_min[j-1])
    #     # P_min[j] = (pPN[j-1]*P_min[j-1])/(pNP[j-1]+pNN[j-1]*P_min[j-1])
    #     P_plus[j] = (pNP[j]*P_plus[j-1]) / (pNP[j]+pNN[j]*P_min[j-1])
    #     P_min[j] = (pPN[j]*P_min[j-1])/(pNP[j]+pNN[j]*P_min[j-1])
    #     print()
    #return P_plus, P_min

    # P_plus = []
    # P_min = []
    # P_plus.append(pPP[0])
    # P_min.append(pNN[0])
    # for i in range(1,len(pPP)):
    #     P_plus.append((pNP[i-1]*P_plus[i-1])/(pNP[i-1]+pNN[i-1]*P_min[i-1]))
    #     P_min.append((pPN[i-1]*P_min[i-1])/(pNP[i-1]+pNN[i-1]*P_min[i-1]))
    # return P_plus, P_min

def get_TIS_cross_from_PPTIS_cross(P_plus,pNP):
    """
    The TIS cross probability P_A[j] is given by the following formula:
    P_A[j] = pNP[0]*P_plus[j]
    """
    P_A = []
    for i in range(len(P_plus)):
        P_A.append(pNP[1]*P_plus[i])
    return P_A

def calculate_cross_probabilities(pathensembles, verbose=True):
    """
    Calculates and returns the TIS and PPTIS crossing probabilities for the given path ensembles.
    For each path ensemble, print the shortcrossing probabilities and the TIS and PPTIS crossing probabilities.
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


def extract_Pcross_from_retis_html_report(html_report_file):
    """
    In the table following the line "<p>The calculated crossing probabilities are:</p>"
    in the html report file, the crossing probabilities are given. Note that the crossing probabilities are 
    given right after "<tr><td>[0^+]</td>" lines in the mentioned table, after which the error and relative 
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

def compare_cross_probabilities(pPP, pPN, pNP, pNN, P_plus, P_min, P_A, P_cross, P_cross_err, P_cross_relerr):
    """
    First, print a table with pNP, P_cross, P_cross_err and P_cross_relerr. 
    """
    import matplotlib.pyplot as plt
    print("")
    print("Comparison of long and TIS crossing probabilities:")
    print("----------------------------------------------")
    print("interf\tpNP\t\tP_cross\t\tP_cross_err\tP_cross_relerr")
    for i, (pnp, pa, pc, pce, pcre) in enumerate(zip(pNP[1:], P_A, P_cross, P_cross_err, P_cross_relerr)):
        print("{}:\t {:.10f}\t{:.10f}\t{:.10f}\t{:.10f}".format(i+1,pnp,pc,pce,pcre))
    print("----------------------------------------------")

    # Second, calculate P_A according to RETIS, whhere P_A_RETIS[i] = product of P_cross from 0 to i
    P_A_RETIS = []
    for i in range(len(P_cross)):
        if i == 0:
            P_A_RETIS.append(P_cross[0])
        else:
            P_A_RETIS.append(P_A_RETIS[i-1]*P_cross[i])

    # Second bis, we calculate P_A_RETIS_error[i], which is the error of P_A_RETIS[i]
    P_A_RETIS_error = []
    for i in range(len(P_cross)):
        if i == 0:
            P_A_RETIS_error.append(P_cross_err[0])
        else:
            P_A_RETIS_error.append(P_A_RETIS_error[i-1]*P_cross[i]+P_A_RETIS[i-1]*P_cross_err[i])
    
    # Now make a table comparing P_A and P_A_RETIS
    print("")
    print("Comparison of P_A_REPPTIS and P_A_RETIS:")
    print("----------------------------------------------")
    print("interf\tP_A_REPPTIS\tP_A_RETIS")
    for i, (pa, par) in enumerate(zip(P_A, P_A_RETIS)):
        print("{}:\t {:.10f}\t{:.10f}".format(i+1,pa,par))
    print("----------------------------------------------")

    # Third, plot P_A_REPPTIS and P_A_RETIS, and save to a PNG file, 
    # with the name "Pcross_compared.png", and with nice labels
    fig,ax=plt.subplots()
    ax.plot(range(1,len(P_A)+1),P_A,marker='o',label=r"$p_0^{\pm}P^{+}_{j}$")
    ax.errorbar(range(1,len(P_A_RETIS)+1),P_A_RETIS,yerr=P_A_RETIS_error,marker='o',label=r'$P_A(\lambda_{j}|\lambda_{0})$')
    #ax.plot(range(1,len(P_A_RETIS)+1),P_A_RETIS,marker='o',label=r'$P_A(\lambda_{j}|\lambda_{0})$')
    ax.set_xlabel('interface')
    ax.set_ylabel('crossing probability')
    ax.set_title("Comparison of long crossing probabilities in RETIS and REPPTIS")
    ax.legend()
    fig.tight_layout()
    fig.savefig('Pcross_compared.png')

    # Fourth, we plot P_A and P_A_RETIS on a logarithmic scale, where we also plot the error bars for P_A_RETIS
    fig,ax=plt.subplots()
    ax.plot(range(1,len(P_A)+1),P_A,marker='o',label=r"$p_0^{\pm}P^{+}_{j}$")
    ax.errorbar(range(1,len(P_A_RETIS)+1),P_A_RETIS,yerr=P_A_RETIS_error,marker='o',label=r'$P_A(\lambda_{j}|\lambda_{0})$')
    #ax.plot(range(1,len(P_A_RETIS)+1),P_A_RETIS,marker='o',label=r'$P_A(\lambda_{j}|\lambda_{0})$')
    ax.set_xlabel('interface')
    ax.set_ylabel('crossing probability')
    ax.set_yscale('log',nonpositive='clip')
    ax.set_title("Comparison of long crossing probabilities in RETIS and REPPTIS")
    ax.legend()
    fig.tight_layout()
    fig.savefig('Pcross_compared_LOG.png')


    fig,ax=plt.subplots()        
    ax.errorbar(range(1,len(P_cross)+1),P_cross,yerr=P_cross_err,marker='o',linestyle='',
    capsize=5,capthick=1,elinewidth=1,ecolor='black',barsabove=True)
    for i, (pc, pce, pcre) in enumerate(zip(P_cross, P_cross_err, P_cross_relerr)):
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


def create_order_distributions(pathensembles, orderparameters, nbins = 50, verbose = True, flag = "ACC"):
    """
    Creates order parameter distributions for the given path ensembles.
    """
    for pe, op in zip(pathensembles,orderparameters):
        create_order_distribution(pe, op, nbins = nbins, verbose = verbose, flag = flag)
 

def create_order_distribution(pe,op,nbins=50,verbose=True,flag="ACC"):
    """
    Plots the distribution of orderparameters for each path mask in the ensemble (accepted paths)
    """
    import matplotlib.pyplot as plt
    # get weights of the paths
    w, _ = get_weights(pe.flags, ACCFLAGS, REJFLAGS)
    # Get load mask
    loadmask = get_generation_mask(pe, "ld")
    # get acc mask
    accmask = get_acc_mask(pe)
    if flag == "ACC":
        flagmask = accmask
    elif flag == "REJ":
        flagmask = ~accmask
    else:
        flagmask = get_flag_mask(pe, flag)

    # get the lmr masks
    masks, masknames = get_lmr_masks(pe)
    # Strip the orderparameters of their start and endpoints (not part of ensemble)
    stripped_op_list = strip_endpoints(op)
    stripped_op = np.array(stripped_op_list,object)
    # Create the distributions
    fig,ax=plt.subplots(nrows = 4, ncols = 4, figsize = (10,10))
    i = 0
    for mask, maskname in zip(masks, masknames):
        axi = ax[i//4,i%4]
        # Select the paths with the mask
        # print the shape of stripped_op, flagmask, mask, loadmask, and w
        # and name it
        # print("stripped_op.shape = ", stripped_op.shape)
        # print("flagmask.shape = ", flagmask.shape)
        # print("mask.shape = ", mask.shape)
        # print("loadmask.shape = ", loadmask.shape)

        # If you get errors with the shape, check whether your simulation
        # actually finished. It's most likely that ensembles up to j were updated
        # in a specific cycle, while the ensembles from j+1 on were not updated 
        # (because the simulation was [forcefully] stopped before the cycle was finished).
        # TODO: add a check for this, and print a warning if this is the case.
        # It can only be the op that is 1 bigger ...: 

        if len(stripped_op) == len(flagmask) + 1:
            print(' '.join(["WARNING: The order parameter is 1 longer than the flagmask.\n",
            "This is most likely because the simulation was stopped before the\n",
            "cycle was finished. The last order parameter will be ignored."]))
            stripped_op = stripped_op[:-1]
        elif len(stripped_op) != len(flagmask):
            raise Exception("The order parameter and the flagmask have different lengths.")

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
    fig.suptitle("Ensemble {} with interfaces {}.\n These are the {} paths.".format(pe.name, (pe.interfaces)[0], flag))
    fig.tight_layout()
    fig.savefig("{}_{}_order_distributions.png".format(pe.name, flag))



# Not really used anymore
def calculate_crossing_rates(pathensembles, verbose = True):
    """
    Calculates the crossing rates for the given path ensembles.
    The cross rate is the amount of LMR paths divided by the amount of accepted paths,
    using the correct weights for the paths.
    """
    cross_rates = []
    for pe in pathensembles:
        cross_rates.append(get_cross_rate(pe, verbose = verbose))

def get_cross_rate(pe, verbose = True):
    """
    Calculates the crossing rate for the given path ensemble.
    The cross rate is the amount of LMR paths divided by the amount of accepted paths,
    using the correct weights for the paths. The weight of the accepted paths is of course
    just the weight of the ensemble.
    """
    # get weights of the paths
    w, _ = get_weights(pe.flags, ACCFLAGS, REJFLAGS)
    print(np.sum(w))
    # get acc mask
    flagmask = get_acc_mask(pe)
    # get load mask
    loadmask = get_generation_mask(pe, "ld")
    # get the lmr masks
    masks, _ = get_lmr_masks(pe)
    # Calculate the crossing rate
    lmr_mask = masks[2]
    lml_mask = masks[3]
    lmr_acc_mask = select_with_masks(lmr_mask, [flagmask, ~loadmask])
    lml_acc_mask = select_with_masks(lml_mask, [flagmask, ~loadmask])
    w_lmr = select_with_masks(w, [flagmask, lmr_mask, ~loadmask])
    w_lml = select_with_masks(w, [flagmask, lml_mask, ~loadmask])
    cross_rate = np.sum(w_lmr)/(np.sum(w_lmr)+np.sum(w_lml))
    
    if verbose: 
        print("Crossing rate for {} is {}".format(pe.name, cross_rate))
        # and print amount of lmr and lml paths, weighted and unweighted
        print("Amount of ACC LMR paths: {} (weight: {})".format(np.sum(lmr_acc_mask), np.sum(w_lmr)))
        print("Amount of ACC LML paths: {} (weight: {})".format(np.sum(lml_acc_mask), np.sum(w_lml)))
        # Print cross ratio without weights
        print("Crossing rate without weights for {} is {}".format(pe.name, 
            np.sum(lmr_mask)/np.sum(flagmask)))
    return cross_rate
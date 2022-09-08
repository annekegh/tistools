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
    # Plot the pathlength distributions for the different masks
    for mask, maskname in zip(masks,mask_names):
        pl_mask = select_with_masks(pl, [mask, flag_mask])
        w_mask = select_with_masks(w, [mask, flag_mask])
    
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
    hist, bin_edges = np.histogram(pl, bins=nbins, weights=w)
    bin_centers = (bin_edges[1:] + bin_edges[:-1])/2
    return hist, bin_centers

def plot_pathlength_distributions_together(pe, nbins = 50, save=True, dpi=500, do_pdf=False, fn="",\
     status = "ACC", force_bins=False, xmin = 0):
    """
    Plots the pathlength distributions for the given path ensemble.
    """
    import matplotlib.pyplot as plt
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
        pl_mask = select_with_masks(pl, [mask, flagmask])
        w_mask = select_with_masks(w, [mask, flagmask])
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
        fig.savefig(fn+"pathlength_distribution_{}_{}.png".format(pe.name, status), dpi=dpi)
        if do_pdf:
            fig.savefig(fn+"pathlength_distribution_{}_{}.pdf".format(pe.name, status))
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
    
def calculate_crossing_rates(pathensembles, verbose = True):
    """
    Calculates the crossing rates for the given path ensembles.
    The cross rate is the amount of LMR paths divided by the amount of (LML+LMR) paths,
    using the correct weights for the paths.
    """
    cross_rates = []
    for pe in pathensembles:
        cross_rates.append(get_cross_rate(pe, verbose = verbose))


def get_cross_rate(pe, verbose = True):
    """
    Calculates the crossing rate for the given path ensemble.
    The cross rate is the amount of LMR paths divided by the amount of (LML+LMR) paths,
    using the correct weights for the paths.
    """
    # get weights of the paths
    w, _ = get_weights(pe.flags, ACCFLAGS, REJFLAGS)
    # get acc mask
    flagmask = get_acc_mask(pe)
    # get the lmr masks
    masks, _ = get_lmr_masks(pe)
    # Calculate the crossing rate
    lmr_mask = masks[2]
    lml_mask = masks[3]
    lmr_acc_mask = select_with_masks(lmr_mask, [flagmask])
    lml_acc_mask = select_with_masks(lml_mask, [flagmask])
    w_lmr = select_with_masks(w, [flagmask, lmr_mask])
    w_lml = select_with_masks(w, [flagmask, lml_mask])
    cross_rate = np.sum(w_lmr)/(np.sum(w_lmr)+np.sum(w_lml))
    if verbose: 
        print("Crossing rate for {} is {}".format(pe.name, cross_rate))
        # and print amount of lmr and lml paths, weighted and unweighted
        print("Amount of ACC LMR paths: {} (weight: {})".format(np.sum(lmr_acc_mask), np.sum(w_lmr)))
        print("Amount of ACC LML paths: {} (weight: {})".format(np.sum(lml_acc_mask), np.sum(w_lml)))
        # Print cross ratio without weights
        print("Crossing rate without weights for {} is {}".format(pe.name, 
            np.sum(lmr_mask)/(np.sum(lmr_acc_mask)+np.sum(lml_acc_mask))))
    return cross_rate

# def get_cross_rate(pe):
#     """
#     Calculates the crossing rate for the given path ensemble.
#     The cross rate is the amount of LMR paths divided by the amount of (LML+LMR) paths
#     """
#     masks, mask_names = get_lmr_masks(pe)
#     mask_lml = masks[2]
#     mask_lmr = masks[3]
#     ncycle_lml = np.sum(mask_lml)
#     ncycle_lmr = np.sum(mask_lmr)
#     cross_rate = ncycle_lmr/(ncycle_lml+ncycle_lmr)
#     print("Crossing rate (lmr/(lml+lmr)) for {} is {}".format(pe.name, cross_rate))
#     return cross_rate

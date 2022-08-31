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
     do_pdf = False, fn = ""):
    """
    Plots the pathlength distributions for the given path ensemble.
    """
    # get pathlengths
    pl = pe.lengths
    # get weights of the paths
    w, ncycle_true = get_weights(pe.flags, ACCFLAGS, REJFLAGS)
    # get acc mask
    accmask = get_acc_mask(pe)
    ncycle_acc = np.sum(accmask)
    # get the lmr masks
    masks, mask_names = get_lmr_masks(pe)
    # Plot the pathlength distributions for the different masks
    for mask, maskname in zip(masks,mask_names):
        pl_mask = select_with_masks(pl, [mask, accmask])
        w_mask = select_with_masks(w, [mask, accmask])
        plot_pathlength_distribution(pl_mask, w_mask, ncycle_acc, maskname, \
            name=pe.name, nbins=nbins, save=save, dpi=dpi, do_pdf=do_pdf, fn=fn)

def plot_pathlength_distribution(pl, w, ncycle, maskname, nbins=50, \
    name="", save=True, dpi=500, do_pdf=False, fn=""):
    import matplotlib.pyplot as plt
    ncycle_mask = len(w)
    hist, bin_centers = get_pathlength_distribution(pl, w, nbins)
    fig,ax = plt.subplots()
    ax.plot(bin_centers, hist)
    ax.set_xlabel("Pathlength")
    ax.set_ylabel("Counts")
    ax.set_title("Pathlength distribution for {} ensemble ({} of {} paths with mask {})"\
        .format(name, ncycle_mask, ncycle, maskname))
    fig.tight_layout()
    if save:
        fig.savefig(fn+"pathlength_distribution_{}_{}.png".format(name, maskname), dpi=dpi)
        if do_pdf:
            fig.savefig(fn+"pathlength_distribution_{}_{}.pdf".format(name, maskname))
    else:
        fig.show()

def get_pathlength_distribution(pl, w, nbins=50):
    hist, bin_edges = np.histogram(pl, bins=nbins, weights=w)
    bin_centers = (bin_edges[1:] + bin_edges[:-1])/2
    return hist, bin_centers

def plot_pathlength_distributions_together(pe, nbins = 50, save=True, dpi=500, do_pdf=False, fn=""):
    """
    Plots the pathlength distributions for the given path ensemble.
    """
    import matplotlib.pyplot as plt
    # get pathlengths
    pl = pe.lengths
    # get weights of the paths
    w, ncycle_true = get_weights(pe.flags, ACCFLAGS, REJFLAGS)
    # get acc mask
    accmask = get_acc_mask(pe)
    ncycle_acc = np.sum(accmask)
    # get the lmr masks
    masks, mask_names = get_lmr_masks(pe)
    # Plot the pathlength distributions for the different masks
    fig,ax = plt.subplots(nrows=4, ncols=4, figsize=(10,10))
    i = 0
    for mask, maskname in zip(masks,mask_names):
        pl_mask = select_with_masks(pl, [mask, accmask])
        w_mask = select_with_masks(w, [mask, accmask])
        ncycle_mask = len(w_mask)
        hist, bin_centers = get_pathlength_distribution(pl_mask, w_mask, nbins)
        ax[i//4,i%4].plot(bin_centers, hist)
        ax[i//4,i%4].set_xlabel("Pathlength")
        ax[i//4,i%4].set_ylabel("Counts")
        ax[i//4,i%4].set_title("{} of {} paths\n with mask {}"\
            .format(ncycle_mask, ncycle_acc, maskname))
        i+=1
    fig.suptitle("Pathlength distributions for {} ensemble, with {} ACC paths of {} total."\
        .format(pe.name, ncycle_acc, ncycle_true))
    fig.tight_layout()
    if save:
        fig.savefig(fn+"pathlength_distribution_{}.png".format(pe.name), dpi=dpi)
        if do_pdf:
            fig.savefig(fn+"pathlength_distribution_{}.pdf".format(pe.name))
    else:
        fig.show()


def create_pathlength_distributions(pathensembles, nbins=50, save=True, dpi=500, do_pdf=False, \
    fn="",plot_separately=False):
    """
    Creates the pathlength distributions for the given path ensembles.
    """
    print(pathensembles)
    for pe in pathensembles:
        if plot_separately:
            plot_pathlength_distributions_separately(pe, nbins=nbins, save=save, dpi=dpi, do_pdf=do_pdf, fn=fn)
        plot_pathlength_distributions_together(pe, nbins=nbins, save=save, dpi=dpi, do_pdf=do_pdf, fn=fn)
    

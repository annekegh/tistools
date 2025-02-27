from json import load
import numpy as np
from .reading import *

# Hard-coded rejection flags found in output files
ACCFLAGS,REJFLAGS = set_flags_ACC_REJ() 

def get_lmr_masks(pe, masktype="all"):
    """
    Return one (or all) boolean array(s) of the paths, based on whether or not
    path i is of type masktype.

    Parameters
    ----------
    pe : object
        Path ensemble object containing the lmrs attribute.
    masktype : str, optional
        Type of mask to return. If "all", returns all masks.

    Returns
    -------
    tuple or np.ndarray
        If masktype is "all", returns a tuple of (masks, types).
        Otherwise, returns the mask for the specified type.
    """
    # Define the types of paths
    types = ["RML", "RMR", "LMR", "LML", "***", "**L", "**R", "*M*",
             "*ML", "*MR", "L**", "L*L", "LM*", "R**", "R*R", "RM*"]

    # Obtain the boolean masks for each path type
    masks = [pe.lmrs == t for t in types]

    # Create a dictionary of types with their corresponding masks
    masks = dict(zip(types, masks))

    if masktype == "all":
        return masks, types
    else:
        return masks[masktype], masktype

def get_flag_mask(pe, status):
    """
    Return a boolean array indicating the paths with the specified status.

    Parameters
    ----------
    pe : object
        Path ensemble object containing the flags attribute.
    status : str
        Status to filter by (e.g., "ACC" for accepted, "REJ" for rejected).

    Returns
    -------
    np.ndarray
        Boolean array indicating the paths with the specified status.
    """
    if status == "REJ":
        return ~get_flag_mask(pe, "ACC")
    else:
        return pe.flags == status

def get_generation_mask(pe, generation):
    """
    Return a boolean array indicating the paths with the specified generation.

    Parameters
    ----------
    pe : object
        Path ensemble object containing the generation attribute.
    generation : str
        Generation type to filter by (e.g., "ld" for load).

    Returns
    -------
    np.ndarray
        Boolean array indicating the paths with the specified generation.
    """
    return pe.generation == generation

def select_with_masks(A, masks):
    """
    Return the elements of the array A that are True in all of the masks.

    Parameters
    ----------
    A : np.ndarray
        Array to filter.
    masks : list of np.ndarray
        List of boolean arrays (masks) with the same shape as A.

    Returns
    -------
    np.ndarray
        Elements of A that are True in all of the masks.

    Raises
    ------
    ValueError
        If any mask does not have the same shape as A.
    """
    # Check whether masks are for the same object array, meaning they must have the same shape as A
    for mask in masks:
        if mask.shape != A.shape:
            raise ValueError(f"Mask shape {mask.shape} does not match the shape of A {A.shape}.")

    # Use the masks to select the elements of A
    union_mask = np.all(masks, axis=0).astype(bool)
    return A[union_mask]


def plot_pathlength_distributions_separately(pe, nbins=50, save=True, dpi=500,
                                             do_pdf=False, fn="", status="ACC",
                                             force_bins=False, xmin=0):
    """
    Filer and plot the pathlength distributions for the given path ensemble separately.

    This function generates and saves individual pathlength distribution plots for each mask type
    within the given path ensemble, given as a :py:class:`.PathEnsemble` object. It allows for filtering by 
    path status (e.g., accepted or rejected) and can handle variable bin sizes for the histograms.

    Parameters
    ----------
    pe : object
        :py:class:`.PathEnsemble` object containing the lengths attribute.
    nbins : int, optional
        Number of bins for the histogram.
    save : bool, optional
        Whether to save the plot.
    dpi : int, optional
        Dots per inch for the saved plot.
    do_pdf : bool, optional
        Whether to save the plot as a PDF.
    fn : str, optional
        Filename for the saved plot.
    status : str, optional
        Status to filter by (e.g., "ACC" for accepted, "REJ" for rejected).
    force_bins : bool, optional
        Whether to force the number of bins.
    xmin : int, optional
        Minimum x-axis value for the plot.
    """
    # Get pathlengths
    pl = pe.lengths
    # Get weights of the paths
    w, _ = get_weights(pe.flags, ACCFLAGS, REJFLAGS)
    # Get flag masks
    assert (status == "ACC") or (status == "REJ") or (status in REJFLAGS)
    flag_mask = get_flag_mask(pe, status)
    # Get the number of paths with the given flag
    ncycle_flag = np.sum(flag_mask)
    # Get the lmr masks
    masks, mask_names = get_lmr_masks(pe)
    # Get the load mask
    load_mask = get_generation_mask(pe, "ld")

    # Plot the pathlength distributions for the different masks
    for mask, maskname in zip(masks, mask_names):
        pl_mask = select_with_masks(pl, [mask, flag_mask, ~load_mask])
        w_mask = select_with_masks(w, [mask, flag_mask, ~load_mask])

        # Check for better binsize if there are any paths in the mask
        if pl_mask.tolist():
            if nbins > np.max(pl_mask):
                if not force_bins:
                    print('\n'.join(("The number of bins is larger than the",
                                     "maximum pathlength in the mask.",
                                     f"Setting nbins to {np.max(pl_mask)}")))
                    nbins = np.max(pl_mask)
                else:
                    print('\n'.join(("The number of bins is larger than the",
                                     "maximum pathlength in the mask.",
                                     "Set force_bins=False to allow for",
                                     "variable bin sizes")))

        # If plotting rejected paths, give all paths unit weight
        if status != "ACC":
            w_mask = np.ones_like(w_mask)

        plot_pathlength_distribution(pl_mask, w_mask, ncycle_flag, maskname,
                                     name=pe.name, nbins=nbins, save=save,
                                     dpi=dpi, do_pdf=do_pdf, fn=fn, status=status,
                                     xmin=xmin)

def plot_pathlength_distribution(pl, w, ncycle, maskname, nbins=50, 
                                 name="", save=True, dpi=500, do_pdf=False, 
                                 fn="", status="ACC", xmin=0):
    """
    Plot the pathlength distribution for the given path ensemble.

    This function generates and saves a pathlength distribution plot for a specific mask type
    within the path ensemble. It allows for filtering by path status (e.g., accepted or rejected)
    and can handle variable bin sizes for the histogram.

    Parameters
    ----------
    pl : np.ndarray
        Array of pathlengths.
    w : np.ndarray
        Array of weights.
    ncycle : int
        Total number of cycles.
    maskname : str
        Name of the mask.
    nbins : int, optional
        Number of bins for the histogram.
    name : str, optional
        Name of the path ensemble.
    save : bool, optional
        Whether to save the plot.
    dpi : int, optional
        Dots per inch for the saved plot.
    do_pdf : bool, optional
        Whether to save the plot as a PDF.
    fn : str, optional
        Filename for the saved plot.
    status : str, optional
        Status to filter by (e.g., "ACC" for accepted, "REJ" for rejected).
    xmin : int, optional
        Minimum x-axis value for the plot.
    """
    import matplotlib.pyplot as plt

    ncycle_mask = len(w)  # Number of paths having this mask (not the weight)
    hist, bin_centers = get_pathlength_distribution(pl, w, nbins)
    fig, ax = plt.subplots()
    ax.bar(bin_centers, hist, width=bin_centers[-1] / nbins)
    ax.set_xlabel(f"Pathlength ({nbins} bins, bwidth = {np.round(bin_centers[-1] / nbins, 2)})")
    ax.set_ylabel("Counts")
    ax.set_title(f"Pathlength distribution for {name} ensemble\n"
                 f"({ncycle_mask} of {ncycle} paths with status {maskname}) [w = {np.sum(hist)}]")
    ax.set_xlim(left=xmin)
    fig.tight_layout()

    if save:
        fig.savefig(fn + f"pathlen_distrib_{name}_{maskname}_{status}.png", dpi=dpi)
        if do_pdf:
            fig.savefig(fn + f"pathlen_distrib_{name}_{maskname}_{status}.pdf")
    else:
        fig.show()

def get_pathlength_distribution(pl, w, nbins=50):
    """
    Return the pathlength distribution for the given path ensemble.

    This function calculates the histogram of pathlengths for the path ensemble, weighted by the
    provided weights. It also adjusts the pathlengths to exclude the start and end points, which
    are not part of the path ensemble.

    Parameters
    ----------
    pl : np.ndarray
        Array of pathlengths.
    w : np.ndarray
        Array of weights.
    nbins : int, optional
        Number of bins for the histogram.

    Returns
    -------
    tuple
        hist : np.ndarray
            Histogram values.
        bin_centers : np.ndarray
            Bin center values.
    """
    # Subtract two from the pathlengths to get rid of the start and endpoint,
    # which are not part of the path ensemble
    pl = np.array([el - 2 for el in pl])
    hist, bin_edges = np.histogram(pl, bins=nbins, weights=w)
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
    return hist, bin_centers


def plot_pathlength_distributions_together(pe, nbins=50, save=True, dpi=500, 
                                           do_pdf=False, fn="", status="ACC", 
                                           force_bins=False, xmin=0):
    """
    Plot the pathlength distributions for the given path ensemble all together.

    This function generates a grid of subplots, each showing the pathlength distribution
    for a specific mask type within the path ensemble. It allows for filtering by path status
    (e.g., accepted or rejected) and can handle variable bin sizes for the histograms.

    Parameters
    ----------
    pe : object
        :py:class:`.PathEnsemble` object containing the lengths attribute.
    nbins : int, optional
        Number of bins for the histogram.
    save : bool, optional
        Whether to save the plot.
    dpi : int, optional
        Dots per inch for the saved plot.
    do_pdf : bool, optional
        Whether to save the plot as a PDF.
    fn : str, optional
        Filename for the saved plot.
    status : str, optional
        Status to filter by (e.g., "ACC" for accepted, "REJ" for rejected).
    force_bins : bool, optional
        Whether to force the number of bins.
    xmin : int, optional
        Minimum x-axis value for the plot.
    """
    import matplotlib.pyplot as plt

    # We don't want load paths
    load_mask = get_generation_mask(pe, "ld")
    # Get pathlengths
    pl = pe.lengths
    # Get weights of the paths
    w, ncycle_true = get_weights(pe.flags, ACCFLAGS, REJFLAGS)
    # Get flag mask
    assert (status == "ACC") or (status == "REJ") or (status in REJFLAGS)
    flagmask = get_flag_mask(pe, status)
    # Get the number of paths with the given flag
    ncycle_flag = np.sum(flagmask)
    # Get the lmr masks
    masks, mask_names = get_lmr_masks(pe)

    # Plot the pathlength distributions for the different masks
    fig, ax = plt.subplots(nrows=4, ncols=4, figsize=(12, 12))
    i = 0
    for mask, maskname in zip(masks, mask_names):
        pl_mask = select_with_masks(pl, [mask, flagmask, ~load_mask])
        w_mask = select_with_masks(w, [mask, flagmask, ~load_mask])

        # Check for better binsize if there are any paths in the mask
        if pl_mask.tolist():
            if nbins > np.max(pl_mask):
                if not force_bins:
                    print('\n'.join(("The number of bins is larger than the",
                                     "maximum pathlength in the mask.",
                                     f"Setting nbins to {np.max(pl_mask)}")))
                    nbins = np.max(pl_mask)
                else:
                    print('\n'.join(("The number of bins is larger than the",
                                     "maximum pathlength in the mask.",
                                     "Set force_bins=False to allow for",
                                     "variable bin sizes")))

        # If plotting rejected paths, give all paths unit weight
        if status != "ACC":
            w_mask = np.ones_like(w_mask)

        ncycle_mask = len(w_mask)  # Number of paths having this mask
        hist, bin_centers = get_pathlength_distribution(pl_mask, w_mask, nbins)
        ax[i // 4, i % 4].bar(bin_centers, hist, width=bin_centers[-1] / nbins)
        ax[i // 4, i % 4].set_xlabel(f"Pathlength ({nbins} bins, bwidth = {np.round(bin_centers[-1] / nbins, 2)})")
        ax[i // 4, i % 4].set_ylabel("Counts")
        ax[i // 4, i % 4].set_title(f"{ncycle_mask} of {ncycle_flag} paths\n status {maskname} [w = {np.sum(hist)}]")
        ax[i // 4, i % 4].set_xlim(left=xmin)
        i += 1

    if status == "ACC":
        fig.suptitle(f"Pathlength distributions for {pe.name} ensemble, with {ncycle_flag} {status} paths of {ncycle_true} total.", fontsize=16)
    else:
        fig.suptitle(f"Pathlength distributions for {pe.name} paths, with {ncycle_flag} {status} paths of {ncycle_true} total.", fontsize=16)
    fig.tight_layout()
    if save:
        fig.savefig(fn + f"{pe.name}_{status}_pathlen_distrib.png", dpi=dpi)
        if do_pdf:
            fig.savefig(fn + f"{pe.name}_{status}_pathlen_distrib.pdf")
    else:
        fig.show()

def create_pathlength_distributions(pathensembles, nbins=50, save=True, dpi=500,
                                    do_pdf=False, fn="", plot_separately=False, 
                                    status="ACC", force_bins=False, xmin=0):
    """
    Create the pathlength distributions for the given path ensembles.

    This function generates and saves pathlength distribution plots for multiple path ensembles.
    It can plot the distributions separately for each ensemble or together in a grid of subplots.
    The function allows for filtering by path status (e.g., accepted or rejected) and can handle
    variable bin sizes for the histograms.

    Parameters
    ----------
    pathensembles : list of objects
        List of :py:class:`.PathEnsemble` objects.
    nbins : int, optional
        Number of bins for the histogram.
    save : bool, optional
        Whether to save the plot.
    dpi : int, optional
        Dots per inch for the saved plot.
    do_pdf : bool, optional
        Whether to save the plot as a PDF.
    fn : str, optional
        Filename for the saved plot.
    plot_separately : bool, optional
        Whether to plot the distributions separately.
    status : str, optional
        Status to filter by (e.g., "ACC" for accepted, "REJ" for rejected).
    force_bins : bool, optional
        Whether to force the number of bins.
    xmin : int, optional
        Minimum x-axis value for the plot.
    """
    print(pathensembles)
    for pe in pathensembles:
        if plot_separately:
            plot_pathlength_distributions_separately(pe, nbins=nbins, save=save,
                                                     dpi=dpi, do_pdf=do_pdf, fn=fn,
                                                     status=status, force_bins=force_bins,
                                                     xmin=xmin)
        plot_pathlength_distributions_together(pe, nbins=nbins, save=save, 
                                               dpi=dpi, do_pdf=do_pdf, fn=fn,
                                               status=status, force_bins=force_bins,
                                               xmin=xmin)
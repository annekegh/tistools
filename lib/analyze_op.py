"""functions to analyze order parameter and path ensembles

AG, Nov 26, 2019
AG, adapted May, 2020
"""

import numpy as np
import matplotlib.pyplot as plt
from .reading import *

# Hard-coded rejection flags found in output files
ACCFLAGS,REJFLAGS = set_flags_ACC_REJ() 


#---------------------
# TRAJECTORIES
#---------------------
def create_figure_trajs(figbasename, data_op, lengths, accepted, first, acc=True, interfaces=None, generation=None):
    """
    Creates and saves figures of trajectory segments from order parameter data.

    This function generates and saves figures of the first few trajectory segments from the 
    order parameter data. It plots the order parameters and optionally the interfaces and 
    generation information.

    Parameters
    ----------
    figbasename : str
        The base name for the output figure files (without extension).
    data_op : np.ndarray
        The order parameter data, typically the ops attribute of an OrderParameter object.
    lengths : list or np.ndarray
        A list or array of lengths (in time steps) for each trajectory.
    accepted : list or np.ndarray
        A list or array indicating whether each trajectory is accepted (True) or rejected (False).
    first : int
        The number of trajectories to plot, starting from index 0.
    acc : bool, optional
        If True, only accepted trajectories are considered. If False, all trajectories are considered.
        The default is True.
    interfaces : list or np.ndarray, optional
        A list or array of interface values to plot as horizontal lines. The default is None.
    generation : list or np.ndarray, optional
        A list or array of generation information for each trajectory. The default is None.

    Returns
    -------
    None
    """
    ntraj, indices = get_ntraj(accepted, acc=acc)
    maxtraj = min(first, ntraj)
    ncol = int(np.ceil(np.sqrt(maxtraj)))

    fig, axes = plt.subplots(ncol, ncol, figsize=(ncol * 3, ncol * 3))  # in inches
    axes = axes.flatten()

    for count in range(maxtraj):
        index = indices[count]
        da = select_traj(data_op, lengths, index)

        ax = axes[count]
        if interfaces is not None:
            # Plot interfaces
            for val in interfaces:
                ax.plot([0, len(da) - 1], [val, val], color='grey', linestyle='--', linewidth=0.5)

        # Plot order parameter
        ax.plot(da[:, 0], linewidth=1, marker='o', markersize=3, label='Order Parameter')
        # Plot other particles, these are in col=1...
        for k in range(1, da.shape[1]):
            ax.plot(da[:, k], label=f'Particle {k}')

        # Add title
        if generation is None:
            ax.set_title(f"Cycle {index}")
        else:
            ax.set_title(f"Cycle {index} {generation[index]}")

        ax.set_xlabel('Time Step')
        ax.set_ylabel('Order Parameter')
        ax.legend()

    # Remove empty subplots
    for count in range(maxtraj, len(axes)):
        fig.delaxes(axes[count])

    plt.tight_layout()
    plt.savefig(f"{figbasename}.png")
    plt.savefig(f"{figbasename}.pdf")
    plt.close()

def make_plot_trajs(figbasename, folder, interfaces, first, acc=True):
    """
    Generates and saves figures of trajectory segments from order parameter data in a specified folder.

    This function reads the order and path ensemble data from the specified folder, determines 
    the accepted trajectories, and creates figures of the first few trajectory segments. It plots 
    the order parameters and optionally the interfaces and generation information.

    Parameters
    ----------
    figbasename : str
        The base name for the output figure files (without extension).
    folder : str
        The simulation folder containing the order.txt and pathensemble.txt files. 
        Example: 'somedir/000', 'somedir/001', 'somedir/002', etc.
    interfaces : list or np.ndarray
        A list or array of interface values to plot as horizontal lines.
        Example: [lambda0], [lambda0, lambda1], etc.
    first : int
        The number of first trajectories to plot.
    acc : bool, optional
        If True, only accepted trajectories are considered. If False, all trajectories are considered.
        The default is True.

    Returns
    -------
    None
    """
    op, _ = get_data_ensemble_consistent(folder)

    accepted = [(acc == 'ACC') for acc in op.flags]

    create_figure_trajs(figbasename, op.ops, op.lengths, accepted, first,
                        acc=acc, interfaces=interfaces, generation=op.generation)


def analyze_lengths(simul,ens,interfaces,above=0,skip=0):

    op, _ = get_data_ensemble_consistent(simul,ens,interfaces,)

    weights, ncycle_true = get_weights(op.flags)

    # hard coded !!!! TODO
    dtc = 0.1   # time between gromacs values is 100*0.001 ps
    figname = "lengths_histogram.%s.%s"%(simul,ens)
    if above > 0:
        figname += ".above%i"%above
    if skip > 0:
        figname += ".skip%i"%skip
    figname += ".png"
    histogram_lengths(figname,op.lengths,weights,dtc,above=above,skip=skip)


def weighted_avg_and_std(values, weights):
    """
    Calculate the weighted average and standard deviation.

    This function computes the weighted average and standard deviation of the given values 
    using the provided weights. It ensures numerical precision and efficiency.

    Parameters
    ----------
    values : np.ndarray
        An array of values for which the weighted average and standard deviation are to be calculated.
    weights : np.ndarray
        An array of weights corresponding to the values. Must have the same shape as values.

    Returns
    -------
    tuple
        A tuple containing:
        - average (float): The weighted average of the values.
        - std_dev (float): The weighted standard deviation of the values.

    Example
    -------
    >>> values = np.array([1, 2, 3, 4])
    >>> weights = np.array([0.1, 0.2, 0.3, 0.4])
    >>> avg, std_dev = weighted_avg_and_std(values, weights)
    >>> print(avg, std_dev)
    3.0 1.118033988749895
    """
    average = np.average(values, weights=weights)
    # Fast and numerically precise calculation of variance
    variance = np.average((values - average) ** 2, weights=weights)
    return average, np.sqrt(variance)

def plot_histogram(figname, hist, bin_mids, mean, std):
    """
    Plots and saves a histogram with the weighted average and standard deviation.

    Parameters
    ----------
    figname : str
        The base name for the output figure files (without extension).
    hist : np.ndarray
        The histogram values.
    bin_mids : np.ndarray
        The midpoints of the histogram bins.
    mean : float
        The weighted average of the values.
    std : float
        The weighted standard deviation of the values.

    Returns
    -------
    None
    """
    plt.figure()
    plt.plot(bin_mids, hist, label='Histogram')
    plt.xlabel("Time (ps)")
    plt.ylabel("Count")
    plt.title(f"Mean: {mean:.2f} ± {std:.2f} ps, Bin width: {bin_mids[1] - bin_mids[0]:.1f} ps")
    plt.axvline(mean, color='red', linewidth=2, label='Mean')
    plt.axvline(mean - std, color='grey', linestyle='--', linewidth=2, label='Std Dev')
    plt.axvline(mean + std, color='grey', linestyle='--', linewidth=2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{figname}.png")
    plt.savefig(f"{figname}.pdf")
    plt.close()

def histogram_lengths(figname, retislengths, retisweights, timestep, writeout_freq, above=0, skip=0):
    """
    Generates and saves a histogram of trajectory lengths with weighted average and standard deviation.

    This function creates a histogram of the trajectory lengths, calculates the weighted average 
    and standard deviation, and plots the results. It allows for skipping initial trajectories 
    and filtering trajectories above a certain length.

    Parameters
    ----------
    figname : str
        The base name for the output figure files (without extension).
    retislengths : np.ndarray
        An array of trajectory lengths.
    retisweights : np.ndarray
        An array of weights corresponding to the trajectory lengths.
    timestep : float
        The simulation timestep in physical units (e.g., ps).
    writeout_freq : int
        The frequency of writing out data (e.g., every nth step).
    above : int, optional
        The minimum length of trajectories to include in the histogram. The default is 0.
    skip : int, optional
        The number of initial trajectories to skip. The default is 0.

    Returns
    -------
    None

    Example
    -------
    >>> lengths = np.array([10, 20, 30, 40, 50])
    >>> weights = np.array([1, 2, 3, 4, 5])
    >>> histogram_lengths('histogram', lengths, weights, timestep=0.1, writeout_freq=10, above=0, skip=0)
    """
    if len(retislengths) != len(retisweights):
        raise ValueError("Lengths and weights must have the same length.")
    if not (0 <= skip < len(retislengths)):
        raise ValueError("Skip value must be within the range of lengths.")

    # Compute dtc
    dtc = timestep * writeout_freq

    # Skip initial trajectories
    lengths = retislengths[skip:]
    weights = retisweights[skip:]

    # Delete first and last point of each trajectory
    lengths = np.array([l - 2 for l in lengths])

    # Define bins for the histogram
    bins = np.linspace(0, 600, 61)

    if above == 0:
        hist, bin_edges = np.histogram(lengths * dtc, bins=bins, weights=weights)
        bin_mids = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        mean, std = weighted_avg_and_std(lengths * dtc, weights)
    else:
        weights2 = weights * (lengths > above)
        hist, bin_edges = np.histogram(lengths * dtc, bins=bins, weights=weights2)
        bin_mids = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        mean, std = weighted_avg_and_std(lengths * dtc, weights2)

    plot_histogram(figname, hist, bin_mids, mean, std)

#-------------------------------------------------
# Investigate efficiency of sampling
#-------------------------------------------------

def decay_path(figbasename, list_ensembles, dt):
    """
    Investigates the decay of the first path (ld, from load) and generates decay plots.

    This function reads the path ensemble data from the specified list of ensembles, 
    calculates the decay matrix, and plots the decay and new path numbers. It also 
    counts the phase points and prints the total count in physical units.

    Parameters
    ----------
    figbasename : str
        The base name for the output figure files (without extension).
    list_ensembles : list of str
        A list of folder names containing the pathensemble.txt files.
    dt : float
        The time conversion factor to convert lengths to physical units (e.g., ps).

    Returns
    -------
    None
    """
    nens = len(list_ensembles)
    all_data_path = []
    for ens in list_ensembles:
        fn_path = f"{ens}/pathensemble.txt"
        pe = read_pathensemble(fn_path)
        all_data_path.append(pe)

    # Determine size
    ncycles = [pe.ncycle for pe in all_data_path]
    max_cycles = max(ncycles)
    print("ncycles:   ", ncycles)
    print("max_cycles:", max_cycles)
    # Or index of cycles
    ncycles = [pe.cyclenumbers[-1] for pe in all_data_path]
    max_cycles = max(ncycles)
    print("ncycles:   ", ncycles)
    print("max_cycles:", max_cycles)

    # Cycle numbers
    cycle_numbers = [pe.cyclenumbers.tolist() for pe in all_data_path]

    # Fill in the decay matrix
    decay = np.zeros((max_cycles, nens))
    factor = 0.8
    decay[0, :] = 1.

    for i in range(1, max_cycles):
        for j, pe in enumerate(all_data_path):
            # Just copy previous
            decay[i, j] = decay[i - 1, j]
            # Is this index present in ensemble j?
            try:
                index = cycle_numbers[j].index(i)
                # Different cases
                if pe.flags[index] == 'ACC':
                    if pe.generation[index] == 'ld':
                        decay[i, j] = 1.
                    elif pe.generation[index] == 'sh':
                        decay[i, j] = decay[i - 1, j] * factor
                    elif pe.generation[index] in ['00', 'tr']:
                        decay[i, j] = decay[i - 1, j]
                    elif pe.generation[index] == 's+':
                        decay[i, j] = decay[i - 1, j + 1]
                    elif pe.generation[index] == 's-':
                        decay[i, j] = decay[i - 1, j - 1]
                    else:
                        print("Unknown generation type:", pe.generation[index])
            except ValueError:
                pass

    print("decay:", decay.shape)
    plot_decay(figbasename, decay, list_ensembles)
    list_newpathnumbers = [pe.newpathnumbers for pe in all_data_path]
    plot_newpathnumbers(f"{figbasename}.newpaths.png", list_newpathnumbers)

    print("Counting phase points")
    totcount = 0
    for i, pe in enumerate(all_data_path):
        count = compute_phasepoints(pe.lengths, pe.flags, pe.generation, f"{i:03d}")
        totcount += count
        print(f"count {i:03d} {count}")

    print(f"ASSUME dt={dt:.6f} ps")
    print(f"totcount = {totcount} = {totcount * dt:.1f} ps = {totcount * dt / 1000:.4f} ns")
        

def plot_decay(figbasename, decay, list_ensembles):
    """
    Plots the decay of the load path and saves the figures.

    This function generates and saves plots of the decay of the load path for the given 
    ensembles. It creates a zoomed-in plot for the first 150 cycles and a full plot for 
    all cycles.

    Parameters
    ----------
    figbasename : str
        The base name for the output figure files (without extension).
    decay : np.ndarray
        The decay matrix with shape (max_cycles, nens).
    list_ensembles : list of str
        A list of ensemble names for the legend.

    Returns
    -------
    None

    Example
    -------
    >>> decay = np.random.rand(200, 3)
    >>> plot_decay('decay_plot', decay, ['ensemble1', 'ensemble2', 'ensemble3'])
    """
    plt.figure(figsize=(10, 6))
    plt.plot(decay[:150, :])
    plt.xlabel("Cycle")
    plt.ylabel("Initial Condition Survival")
    plt.title("Decay of Load Path (Zoomed)")
    plt.legend(list_ensembles, loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{figbasename}.zoom.png")
    plt.clf()

    plt.figure(figsize=(10, 6))
    plt.plot(decay)
    plt.xlabel("Cycle")
    plt.ylabel("Initial Condition Survival")
    plt.title("Decay of Load Path")
    plt.legend(list_ensembles, loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{figbasename}.png")
    plt.close()

def plot_newpathnumbers(figname, list_newpathnumbers):
    """
    Investigates and plots the number of new paths generated over cycles.

    This function generates and saves a plot showing the number of new paths generated 
    for each ensemble over the cycles.

    Parameters
    ----------
    figname : str
        The name of the output figure file (with extension).
    list_newpathnumbers : list of np.ndarray
        A list of arrays, each containing the number of new paths generated per cycle for an ensemble.

    Returns
    -------
    None
    """
    plt.figure(figsize=(10, 6))
    for i, newpathnumbers in enumerate(list_newpathnumbers):
        plt.plot(newpathnumbers, label=f'Ensemble {i}')
    plt.xlabel("Cycle")
    plt.ylabel("Number of New Paths")
    plt.title("Number of New Paths Generated Over Cycles")
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(figname)
    plt.close()

#-------------------------------------------------
# Compute how many phase points were computed
#-------------------------------------------------

def compute_phasepoints(lengths, flags, generation, ensemble):
    """
    Computes the number of phase points that were calculated in this ensemble.

    This function calculates the number of phase points based on the lengths, flags, 
    and generation types of trajectories in a given ensemble.

    Parameters
    ----------
    lengths : list or np.ndarray
        A list or array of lengths (in time steps) for each trajectory.
    flags : list or np.ndarray
        A list or array of flags indicating the status of each trajectory.
    generation : list or np.ndarray
        A list or array of generation types for each trajectory.
    ensemble : int
        The ensemble index (e.g., 0, 1, 2, etc.).

    Returns
    -------
    int
        The total number of phase points calculated in this ensemble.

    Example
    -------
    >>> lengths = [10, 20, 30]
    >>> flags = ['ACC', 'ACC', 'ACC']
    >>> generation = ['sh', 'tr', 's+']
    >>> ensemble = 0
    >>> count = compute_phasepoints(lengths, flags, generation, ensemble)
    >>> print(count)
    37
    """
    ncycle = len(lengths)
    if ncycle != len(flags) or ncycle != len(generation):
        raise ValueError("Lengths, flags, and generation arrays must have the same length.")

    count = 0
    for i in range(ncycle):
        if generation[i] in ["sh", "ki"]:
            # One point is recycled, i.e., the shooting point
            count += lengths[i] - 1
        elif generation[i] in ["tr", "00"]:
            pass  # Do not add anything
        elif generation[i] in ["s+", "s-"]:
            if ensemble < 2:
                # Two points are recycled, 1 before and 1 after the interface
                count += lengths[i] - 2
            else:
                pass  # Do not add anything
    return count


###############
# PERMEABILITY
###############
#-------------------------------------------------
# Computation of \xi
# The Big Reweight
#-------------------------------------------------
def calc_xi(lmrs, weights):
    """
    Computes the factor xi for ensemble [0-'].

    xi is calculated as the ratio of the total number of paths to the number of paths that end at R in [0-'].

    Parameters
    ----------
    lmrs : np.ndarray
        An array of path codes indicating the transitions (e.g., 'LML', 'RML', etc.).
    weights : np.ndarray
        An array of weights corresponding to the paths.

    Returns
    -------
    float
        The computed factor xi.
    """
    print("Calculating xi...")
    print(f"Number of lmrs: {len(lmrs)}")
    print(f"Total weights: {np.sum(weights)}")

    # make sure lmrs is an array
    lmrs = np.array(lmrs)

    n_lml = np.sum((lmrs == "LML") * weights)
    n_rml = np.sum((lmrs == "RML") * weights)
    n_lmr = np.sum((lmrs == "LMR") * weights)
    n_rmr = np.sum((lmrs == "RMR") * weights)
    n_lstarl = np.sum((lmrs == "L*L") * weights)
    n_rstarl = np.sum((lmrs == "R*L") * weights)
    n_lstarr = np.sum((lmrs == "L*R") * weights)
    n_rstarr = np.sum((lmrs == "R*R") * weights)
    n_lstarstar = np.sum((lmrs == "L**") * weights)

    n_ends_l = n_lml + n_rml + n_lstarl + n_rstarl + n_lstarstar
    n_ends_r = n_lmr + n_rmr + n_lstarr + n_rstarr

    n_all = np.sum(weights)

    # Skip the load path if it is not in the specified codes
    if weights[0] > 0 and lmrs[0] not in ["LML", "RML", "RMR", "LMR", "L*L", "R*L", "L*R", "R*R"]:
        n_all = np.sum(weights[1:])  # Skip this first path

    if n_all != n_ends_r + n_ends_l:
        raise ValueError("The total number of paths does not match the sum of paths ending at L and R.")

    if n_ends_r > 0:
        print("Big reweighting required!")
        xi = n_all / n_ends_r
    else:
        print("No reweighting required, because n_ends_r <= 0")
        xi = 1

    print(f"Computed xi: {xi}")
    return xi

def print_lmr(lmrs, weights):
    """
    Prints the codes of ensemble, such as LML and RML, along with their counts and weighted counts.

    This function prints the count and weighted count of each path code in the ensemble.

    Parameters
    ----------
    lmrs : np.ndarray
        An array of path codes indicating the transitions (e.g., 'LML', 'RML', etc.).
    weights : np.ndarray
        An array of weights corresponding to the paths.

    Returns
    -------
    None

    Example
    -------
    >>> lmrs = np.array(['LML', 'RML', 'L*L', 'R*R'])
    >>> weights = np.array([1, 2, 3, 4])
    >>> print_lmr(lmrs, weights)
    Counting paths in ensemble
    Code   Count  Weighted Count
    LML   1     1.00
    RML   1     2.00
    L*L   1     3.00
    R*R   1     4.00
    """
    print("Counting paths in ensemble")
    print(f"{'Code':<6} {'Count':<6} {'Weighted Count':<15}")
    codes = ["LML", "LMR", "L*R", "L*L", "RMR", "RML", "R*R", "R*L", "L**", "R**", "**L", "**R", "RM*", "LM*", "*MR", "*ML", "*M*"]
    for code in codes:
        n = np.sum(lmrs == code)
        nw = np.sum((lmrs == code) * weights)
        print(f"{code:<6} {n:<6} {nw:<15.2f}")

def print_concentration_lambda0(ncycle, trajs, cut, dlambda, dt, w_all, xi):
    """
    Compute density at interface lambda0.

    This function calculates and prints various metrics related to the density at a specified
    interface (lambda0) for permeability calculations.

    Parameters
    ----------
    ncycle : int
        Total number of cycles.
    trajs : np.ndarray
        Array of trajectory data.
    cut : float
        Interface position (lambda0).
    dlambda : float
        Bin width for histogram.
    dt : float
        Time step size.
    w_all : np.ndarray
        Weights for each trajectory.
    xi : float
        Scaling factor for the results.

    Returns
    -------
    nL : float
        Count in the histogram bin to the left of the interface.
    """
    
    print("Calculating density at interface lambda0...\n")
    print("Parameters:")
    print(f"  Time step (dt): {dt}")
    print(f"  Bin width (dlambda): {dlambda}\n")

    print("Getting values around the interface...")

    # Histogram bins to the left of the cut
    bins_left = np.arange(cut - 30 * dlambda, cut + dlambda / 10., dlambda)
    histL, edgesL = np.histogram(trajs, bins=bins_left, weights=w_all)
    nL = histL[-1]
    print(f"  Bins at cut (lambda = {cut}):")
    print(f"    Left of cut: {histL[-2]}, {histL[-1]}")

    # Histogram bins to the right of the cut
    bins_right = np.arange(cut, cut + 30 * dlambda + dlambda / 10., dlambda)
    histR, edgesR = np.histogram(trajs, bins=bins_right, weights=w_all)
    nR = histR[0]
    print(f"    Right of cut: {histR[0]}, {histR[1]}\n")

    print("Computing metrics for left (L) and right (R) of the interface...")

    # Metrics for the left side
    rhoL = nL / dlambda / ncycle
    tauL = nL * dt / dlambda / ncycle
    PL = 1. / tauL

    # Metrics for the right side
    rhoR = nR / dlambda / ncycle
    tauR = nR * dt / dlambda / ncycle
    PR = 1. / tauR

    print("** Left histogram **")
    print(f"  L n            : {nL}")
    print(f"  L n/ncycle     : {nL / ncycle}")
    print(f"  L rho          : {rhoL}")
    print(f"  L tau/Dz       : {tauL}")
    print(f"  L P/Pcross     : {PL}")
    print("** Left histogram (scaled by xi) **")
    print(f"  L n        (xi): {nL * xi}")
    print(f"  L n/ncycle (xi): {nL / ncycle * xi}")
    print(f"  L rho      (xi): {rhoL * xi}")
    print(f"  L tau/Dz   (xi): {tauL * xi}")
    print(f"  L P/Pcross (xi): {PL / xi}\n")

    print("** Right histogram **")
    print(f"  R n            : {nR}")
    print(f"  R n/ncycle     : {nR / ncycle}")
    print(f"  R rho          : {rhoR}")
    print(f"  R tau/Dz       : {tauR}")
    print(f"  R P/Pcross     : {PR}")
    print("." * 10)

    return nL
    

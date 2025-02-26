import numpy as np
from .reading import *
from .analyze_op import *
import matplotlib.pyplot as plt

ACCFLAGS,REJFLAGS = set_flags_ACC_REJ() # Hard-coded rejection flags found in output files


# first crossing point distribution

def get_first_crossing_point(traj, lambdacross, lmr):
    """
    Get the first crossing point of a trajectory.

    Parameters
    ----------
    traj : np.ndarray
        Trajectory data.
    lambdacross : float
        Crossing lambda value.
    lmr : str
        Left or right crossing indicator.

    Returns
    -------
    tuple
        Index of the first crossing point and the crossing value.
    """
    if lmr.startswith("L"):
        # Detect when the trajectory crosses to the right of the interface
        a = np.where(traj >= lambdacross)[0]
    elif lmr.startswith("R"):
        # Detect when the trajectory crosses to the left of the interface
        a = np.where(traj <= lambdacross)[0]
    else:
        return -1, None

    if len(a) == 0:
        return -1, None
    else:
        cross_index = a[0]
        crossing = traj[cross_index]
        return cross_index, crossing

def get_first_crossing_distr(flags, weights, lens, w_all, ncycle, ncycle_true, trajs, lmrs, lambdacross):
    """
    Get the distribution of first crossing points.

    Parameters
    ----------
    flags : np.ndarray
        Array of flags for each trajectory.
    weights : np.ndarray
        Array of weights for each trajectory.
    lens : np.ndarray
        Array of lengths for each trajectory.
    w_all : np.ndarray
        Array of weights for all trajectories.
    ncycle : int
        Total number of cycles.
    ncycle_true : int
        True number of cycles.
    trajs : np.ndarray
        Array of trajectory data.
    lmrs : np.ndarray
        Array of left or right crossing indicators.
    lambdacross : float
        Crossing lambda value.

    Returns
    -------
    np.ndarray
        Array of first crossing indices.
    """
    assert len(lens) == ncycle
    assert len(lmrs) == ncycle

    cross_indices = []
    count = 0
    for i in range(ncycle):
        assert lens[i] > 0
        traj = trajs[count:count + lens[i]]
        count += lens[i]
        if weights[i] > 0:
            cross_index, crossing = get_first_crossing_point(traj, lambdacross, lmrs[i])
            cross_indices.append(cross_index)
        else:
            cross_indices.append(-1)

    assert count == len(trajs)
    return np.array(cross_indices)

def create_distrib_first_crossing(folders, interfaces_input, outputfile, do_pdf, dt, offset):
    """
    Create figure of distributions of first crossing points.

    Parameters
    ----------
    folders : list of str
        List of folder paths.
    interfaces_input : list of float
        List of interface positions.
    outputfile : str
        Output filename for the figures.
    do_pdf : bool
        Whether to save figures as PDF.
    dt : float
        Time step size.
    offset : float
        Offset to apply to interfaces.
    """
    extensions = ["png"]
    if do_pdf:
        extensions.append("pdf")

    interfaces = [interf - offset for interf in interfaces_input] if offset != 0. else interfaces_input

    plt.figure(1, figsize=(10, 6))
    if len(folders) > 2:
        plt.figure(2, figsize=(10, 6))

    for ifol, folder in enumerate(folders):
        fol = folder[-3:]
        print("-" * 20)
        print(f"Processing folder: {fol}")
        ofile = f"{folder}/order.txt"
        ostart = -1  # TODO: Verify if this should be changed
        op = read_order(ofile, ostart)
        flags = op.flags
        trajs = op.longtraj
        lens = op.lengths
        ncycle = op.ncycle
        weights, ncycle_true = get_weights(flags, ACCFLAGS, REJFLAGS)

        if offset != 0.:
            trajs = op.longtraj - offset

        print(f"Number of cycles: {ncycle}")
        w_all = np.array([weights[i] for i in range(ncycle) for k in range(lens[i])])
        print(f"One long trajectory: {len(w_all)}, {len(trajs)}")
        print("Statistics:")
        for flag in ACCFLAGS + REJFLAGS:
            print(f"  {flag}: {np.sum(flags == flag)}")
        print(f"Total flags: {len(flags)}")

        # Read path ensemble and calculate xi
        ofile = f"{folder}/pathensemble.txt"
        pe = read_pathensemble(ofile)
        lmrs = pe.lmrs
        flags1 = pe.flags
        print_lmr(lmrs, weights)
        xi = calc_xi(lmrs, weights)
        print(f"xi: {xi}")

        # Determine the crossing lambda value
        if ifol == 0:
            if len(interfaces) == len(folders):
                lambdacross = interfaces[0]
            elif len(interfaces) == len(folders) + 1:
                lambdacross = (interfaces[0] + interfaces[-1]) / 2.
            else:
                raise ValueError("Mismatch between number of interfaces and folders.")
        elif ifol == 1:
            lambdacross = interfaces[0]
        else:
            lambdacross = interfaces[ifol - 1]

        # Get the distribution of first crossing points
        cross_indices = get_first_crossing_distr(flags, weights, lens, w_all, ncycle, ncycle_true, trajs, lmrs, lambdacross)

        # Plot histogram of first crossing points
        bins = np.arange(-1, max(cross_indices) + 2) - 0.5
        hist, edges = np.histogram(cross_indices, bins=bins, weights=weights)
        centers = edges[:-1] + np.diff(edges) / 2.
        hist2, edges = np.histogram(lens, bins=bins, weights=weights)

        plt.figure(2)
        plt.bar(centers, hist, label="Crossing points")
        plt.plot(centers, hist2, label="Path lengths + 2", color='green')
        plt.xlim(xmin=-1)
        plt.xlabel(f"Phase point crossing lambda_i={lambdacross:.3f}")
        plt.ylabel("Histogram")
        plt.title(f"{fol}, ncycle={ncycle}, dt={dt:.3f}")
        plt.legend(loc='best')
        plt.savefig(f"crosshist.{fol}.png")
        plt.clf()

        # TODO: Optional - Remove first and last phase point of each path, which are not part of the ensemble
        # TODO: Optional - Compute time spent in the ensemble

        if ifol != 0:
            # Plot normalized histogram
            bins = np.arange(-1, 51) - 0.5
            hist, edges = np.histogram(cross_indices, bins=bins, weights=weights)
            centers = edges[:-1] + np.diff(edges) / 2.

            plt.figure(1)
            plt.plot(centers, hist / float(np.sum(hist)) + 0.2 * ifol, label=fol)

    plt.figure(1)
    plt.xlim(xmin=-1)
    plt.xlabel("Phase point crossing lambda_i")
    plt.ylabel("Normalized histogram")
    plt.legend()
    plt.title(f"Normalized, ncycle={ncycle}, dt={dt:.3f}")
    plt.tight_layout()
    plt.savefig("crosshist.all.png")


import numpy as np
from .reading import *
from .analyze_op import *
import matplotlib.pyplot as plt

ACCFLAGS,REJFLAGS = set_flags_ACC_REJ() # Hard-coded rejection flags found in output files


def get_bins(folders, interfaces, dlambda, lmin=None, lmax=None):
    """
    Generate bins for the lambda interval.

    Parameters
    ----------
    folders : list of str
        List of folder paths.
    interfaces : list of float
        List of interface positions.
    dlambda : float
        Bin width.
    lmin : float, optional
        Minimum lambda value. If not provided, it is calculated based on the interfaces.
    lmax : float, optional
        Maximum lambda value. If not provided, it is calculated based on the interfaces.

    Returns
    -------
    np.ndarray
        Array of bin edges.
    """
    if lmin is None:
        lm = interfaces[0] - 30 * dlambda if len(folders) >= len(interfaces) else interfaces[-1] - 10 * dlambda
    else:
        lm = lmin

    if lmax is None:
        lM = interfaces[len(folders) - 1] + 10 * dlambda
    else:
        lM = lmax

    bins = np.arange(lm, lM + dlambda / 10., dlambda)
    return bins

def create_distrib(folders, interfaces_input, outputfile, do_pdf, dt, dlambda, dlambda_conc, lmin, lmax, ymin, ymax, offset, box, op_index, op_weight, do_abs, do_time, do_density):
    """
    Create figure of distributions of order parameter.

    This function generates and saves figures showing the distributions of a specified order parameter
    across different path ensembles. It supports normalization by time or density and can handle
    periodic boundary conditions.

    Parameters
    ----------
    folders : list of str
        List of folder paths. E.g., ["somedir/000", "somedir/001", "somedir/002", ...].
    interfaces_input : list of float
        List of interface positions. E.g., [lambda0,lambda1,...,lambdaN], and if lambda_-1 exists: [lambda0,lambda1,...,lambdaN, lambda_-1].
    outputfile : str
        Output filename for the figures.
    do_pdf : bool
        Whether to save figures as PDF.
    dt : float
        Time step size.
    dlambda : float
        Bin width for histogram.
    dlambda_conc : float
        Bin width for concentration calculation.
    lmin : float
        Minimum lambda value.
    lmax : float
        Maximum lambda value.
    ymin : float
        Minimum y-axis value for plots.
    ymax : float
        Maximum y-axis value for plots.
    offset : float
        Offset to apply to interfaces.
    box : tuple of float
        Box dimensions for periodic boundary conditions.
    op_index : list of int
        List of selected order parameters.
    op_weight : list of float
        Weight for each of the selected order parameters.
    do_abs : bool
        Whether to take the absolute value of the order parameters before histogramming.
    do_time : bool
        Whether to normalize histogram by time.
    do_density : bool
        Whether to normalize histogram by density.
    """
    extensions = ["png"]
    if do_pdf:
        extensions.append("pdf")

    interfaces = [interf - offset for interf in interfaces_input] if offset != 0. else interfaces_input
    bins = get_bins(folders, interfaces, dlambda, lmin, lmax)

    plt.figure(1, figsize=(10, 6))
    if len(folders) > 2:
        plt.figure(2, figsize=(10, 6))

    for ifol, folder in enumerate(folders):
        fol = folder[-3:]
        print("-" * 20)
        print(f"Processing folder: {fol}")
        ofile = f"{folder}/order.txt"
        ostart = -1
        op = parse_order_file(ofile, ostart)
        flags = op.flags
        trajs = op.longtraj
        lens = op.lengths
        ncycle = op.ncycle
        weights, ncycle_true = get_weights(flags, ACCFLAGS, REJFLAGS)

        if offset != 0.:
            trajs = op.longtraj - offset

        print(f"Number of cycles: {ncycle}")
        assert len(op.ops) == len(op.longtraj)
        w_all = np.array([weights[i] for i in range(ncycle) for k in range(lens[i])])
        print(f"One long trajectory: {len(w_all)}, {len(trajs)}")
        print("Statistics:")
        for flag in ACCFLAGS + REJFLAGS:
            print(f"  {flag}: {np.sum(flags == flag)}")
        print(f"Total flags: {len(flags)}")

        xi = 1.
        if fol == "000":
            ofile = f"{folder}/pathensemble.txt"
            pe = read_pathensemble(ofile)
            lmrs = pe.lmrs
            flags1 = pe.flags

            # extract the corresponding cycles from the pathensemble.txt file in case the  order parameter is only stored for every nth trajectory
            flag1_ext = []
            lmrs_ext = []
            # calculate value of n
            n = int(len(flags1)-1)/(len(flags)-1)
            if n != 1:
                for j in range(0,len(flags)):
                    for k, (flag_pe, lmr) in enumerate(zip(flags1,lmrs)):
                        if k  == (j * n):
                            flag_pe = str(flag_pe)
                            flag1_ext.append(flag_pe)
                            lmr = str(lmr)
                            lmrs_ext.append(lmr)
                            break
                # overwrite flags and lmrs list from pathensemble
                flags1 = flag1_ext  
                lmrs = np.array(lmrs_ext)
        

            for i, (flag0, flag1) in enumerate(zip(flags, flags1)):
                if flag0 != flag1:
                    raise ValueError(f"Trajectory {i} flag mismatch: {flag0} vs {flag1}")
            print_lmr(lmrs, weights)
            xi = calc_xi(lmrs, weights)
            print(f"xi: {xi}")

        if fol in ["000", "001"]:
            cut = interfaces[0]
            nL = print_concentration_lambda0(ncycle, trajs, cut, dlambda_conc, dt, w_all, xi)

        # TODO
        # Optional:
        # remove first and last phase point of each path, which are not part of the ensemble
        # I think that it also works when the path length is 1
        # Also: compute time spent in the ensemble
        do_remove_endpoints = False
        if do_remove_endpoints:
            # reconstruct w_all
            time,w_all = compute_time_in_ensemble(w_all,lens,dt)
        else:
            # do not store w_all
            time_ens, _ = compute_time_in_ensemble(w_all,lens,dt)

        hist = np.zeros(len(bins) - 1)
        n_op = len(op_index)
        for i in range(n_op):
            if op_weight[i] != 0:
                offset1 = offset if op_index[i] == 0 else 0
                if box is None:
                    if do_abs:
                        histi, edges = np.histogram(abs(op.ops[:, op_index[i]] - offset1), bins=bins, weights=w_all * op_weight[i])
                    else:
                        histi, edges = np.histogram(op.ops[:, op_index[i]] - offset1, bins=bins, weights=w_all * op_weight[i])
                else:
                    low, high = box
                    L = high - low
                    if do_abs:
                        histi, edges = np.histogram(abs(op.ops[:, op_index[i]] - offset1 - np.floor(op.ops[:, op_index[i]] / L - offset1 / L) * L + low), bins=bins, weights=w_all * op_weight[i])
                    else:
                        histi, edges = np.histogram(op.ops[:, op_index[i]] - offset1 - np.floor(op.ops[:, op_index[i]] / L - offset1 / L) * L + low, bins=bins, weights=w_all * op_weight[i])
                hist += histi

        centers = edges[:-1] + np.diff(edges) / 2.

        # Normalization?
        #---------------
        # Several options...
        # Here: "time in interval dlambda per path" and "prob_dens"
        if do_time:
            hist = hist / ncycle * dt / dlambda
        elif do_density:
            time_ens, _ = compute_time_in_ensemble(w_all, lens, dt)
            hist = hist / ncycle * dt / dlambda / time_ens

        for k in [1, 2]:
            if k == 1 or (len(folders) > 2 and k == 2):
                plt.figure(k)
                for c in interfaces:
                    plt.plot([c, c], [0, max(hist)], "--", color='grey')
                if fol == "000":
                    plt.plot(centers, hist, marker='x', label=fol, color="green")
                    title = f"ncycle={ncycle} nL={nL}"
                    if xi != 1 and not do_density:
                        plt.plot(centers, hist * xi, marker='o', label=f"{fol}-xi", color="green", fillstyle='none')
                        title += f" xi={xi:.2f}"
                    plt.title(title)
                elif fol == "001":
                    plt.plot(centers, hist, marker='x', label=fol, color="orange")
                else:
                    if k == 1:
                        plt.plot(centers, hist, marker='x', label=fol)
                if ifol == 1:
                    plt.figure(2)
                    plt.legend()
                    plt.xlabel(f"Lambda (dlambda={dlambda:.3f})")
                    plt.xlim(bins[0], bins[-1])
                    if ymin is not None and ymax is not None:
                        plt.ylim(ymin, ymax)
                    plt.ylabel("Time spent per length" if do_time else "Probability density")
                    plt.tight_layout()
                    for ext in extensions:
                        plt.savefig(f"{outputfile}.time.01.{ext}" if do_time else f"{outputfile}.dens.01.{ext}", transparent=(ext == "pdf"))

        plt.figure(1)
        plt.legend()
        plt.xlabel(f"Lambda (dlambda={dlambda:.3f})")
        plt.xlim(bins[0], bins[-1])
        if ymin is not None and ymax is not None:
            plt.ylim(ymin, ymax)
        plt.ylabel("Time spent per length" if do_time else "Probability density")
        plt.tight_layout()
        for ext in extensions:
            plt.savefig(f"{outputfile}.time.{ext}" if do_time else f"{outputfile}.dens.{ext}", transparent=(ext == "pdf"))


def compute_time_in_ensemble(w_all, lens, dt):
    """
    Compute time spent in ensemble.

    This function calculates the total time spent in the ensemble by removing the first and last
    phase points of each path, which are not part of the ensemble. It also returns the modified
    weights array with the first and last points set to zero.

    Parameters
    ----------
    w_all : np.ndarray
        Array with weights of each order parameter path, for each stored phase point.
    lens : list of int
        Length of each path.
    dt : float
        Time step size.

    Returns
    -------
    tuple
        Total time spent in ensemble and modified weights array.
    """
    w_all2 = np.copy(w_all)
    ncycle = len(lens)
    count = 0
    for i in range(ncycle):
        assert lens[i] > 0
        w_all2[count] = 0
        w_all2[count + lens[i] - 1] = 0
        count += lens[i]
    time = np.sum(w_all2) * dt / ncycle
    print(f"Time spent per path: {time:.3f}")
    return time, w_all2

def make_histogram_op(figbasename, folder, interfaces, bins, skip=0):
    """
    Create and save histograms of order parameters.

    Parameters
    ----------
    figbasename : str
        Base name for the output figure files.
    folder : str
        Path to the folder containing the data.
    interfaces : list of float
        List of interface positions.
    bins : np.ndarray
        Array of bin edges.
    skip : int, optional
        Number of initial trajectories to skip.

    Notes
    -----
    This function generates histograms with and without the first and last points of each trajectory,
    as well as histograms excluding short trajectories.
    """
    op, _ = get_data_ensemble_consistent(folder)
    weights, _ = get_weights(op.flags, ACCFLAGS, REJFLAGS)
    if skip > 0:
        figbasename += f".skip{skip}"
    histogram_op(figbasename, interfaces, op.ops, op.lengths, weights, bins, skip=skip)

def histogram_op(figbasename, interfaces, data_op, length, weights, bins, skip=0):
    """
    Generate and save histograms of order parameters.

    Parameters
    ----------
    figbasename : str
        Base name for the output figure files.
    interfaces : list of float
        List of interface positions.
    data_op : np.ndarray
        Array of order parameter data.
    length : list of int
        Length of each trajectory.
    weights : np.ndarray
        Weights for each trajectory.
    bins : np.ndarray
        Array of bin edges.
    skip : int, optional
        Number of initial trajectories to skip.

    Notes
    -----
    This function generates histograms with and without the first and last points of each trajectory,
    as well as histograms excluding short trajectories.
    """
    ntraj = len(length)
    assert len(weights) == len(length)
    index = 0  # Column with the order parameter
    hist = np.zeros(len(bins) - 1)

    for i in range(skip, ntraj):
        if weights[i] > 0:
            histi, _ = np.histogram(select_traj(data_op, length, i)[:, index], bins=bins)
            hist += weights[i] * histi

    hist2 = np.zeros(len(bins) - 1)
    for i in range(skip, ntraj):
        if weights[i] > 0:
            histi, _ = np.histogram(select_traj(data_op, length, i)[1:-1, index], bins=bins)
            hist2 +=[i] * histi

    tooshort = 3
    hist3 = np.zeros(len(bins) - 1)
    for i in range(skip, ntraj):
        if weights[i] > 0:
            if length[i] > tooshort:
                histi, _ = np.histogram(select_traj(data_op, length, i)[:, index], bins=bins)
                hist3 += weights[i] * histi

    bin_mids = 0.5 * (bins[:-1] + bins[1:])

    plt.figure(figsize=(10, 6))
    plt.plot(bin_mids, hist, "x", label="With start/end points")
    plt.plot(bin_mids, hist2, "x", label="Without start/end points")
    plt.plot(bin_mids, hist3, "x", label="Without short trajectories")
    plt.xlabel("Lambda")
    plt.ylabel("Count")
    plt.title(f"Bin width = {bins[1] - bins[0]:.3f}")
    lambda0 = interfaces[0]
    plt.axvline(x=lambda0, color='grey', linewidth=2, label="Lambda0")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{figbasename}.png")

    plt.figure(figsize=(10, 6))
    plt.plot(bin_mids, -np.log(hist), "x", label="With start/end points")
    plt.plot(bin_mids, -np.log(hist2), "x", label="Without start/end points")
    plt.plot(bin_mids, -np.log(hist3), "x", label="Without short trajectories")
    plt.xlabel("Lambda")
    plt.ylabel("-ln(Count)")
    plt.axvline(x=lambda0, color='grey', linewidth=2, label="Lambda0")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{figbasename}.log.png")
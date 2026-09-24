#!/usr/bin/env python
"""
infretis_memory_analysis.py: crossing-probability profile and memory-effect
analysis for iSTAR/StapleTIS simulations run with infretis.

This is the infretis-data counterpart of `tistools-memory-analysis` (which
reads PyRETIS/toytis-style `logging.log` + per-ensemble `pathensemble.txt`
folders). Infretis instead writes a single `infretis_data.txt` path-data file
plus an `infretis.toml` config, so the reading and weight-matrix-construction
front end here is different; everything downstream (crossing-probability
profile, memory analysis, memory landscape) produces the same plots.

The infretis reading/weight logic is ported from
`inftools/examples/staple_test/infretis_istar_workflow.ipynb` and
`istar_memory_analysis.py` in that same folder. This file is self-contained
(only depends on `tistools`, numpy, matplotlib, seaborn and a TOML reader) so
it can be copied into any folder and pointed at an infretis simulation
directory.

Usage:
    python infretis_memory_analysis.py <simulation_dir> [options]

Example:
    python infretis_memory_analysis.py /path/to/infretis_sim --show
"""

import argparse
import re
import sys
# A sibling project (inftools) puts a directory containing its own empty
# `tistools` namespace package on sys.path; strip it so `import tistools`
# below resolves to the real, pip-installed (editable) tistools package
# instead of that empty shadow.
sys.path = [p for p in sys.path if 'inftools' not in p]
from pathlib import Path

import numpy as np

try:
    import tomllib
except ImportError:
    import tomli as tomllib


def parse_args():
    parser = argparse.ArgumentParser(
        description="Crossing-probability profile and memory-effect analysis for infretis iSTAR/StapleTIS simulations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "simdir",
        type=str,
        help="Path to the infretis simulation directory (containing infretis_data.txt and infretis.toml)",
    )
    parser.add_argument(
        "--data-file",
        type=str,
        default=None,
        help="Path to infretis_data.txt (default: <simdir>/infretis_data.txt)",
    )
    parser.add_argument(
        "--toml-file",
        type=str,
        default=None,
        help="Path to infretis.toml (default: <simdir>/infretis.toml)",
    )
    parser.add_argument(
        "--nskip",
        type=int,
        default=0,
        help="Number of initial path entries to skip (default: 0)",
    )
    parser.add_argument(
        "--tr",
        action="store_true",
        default=False,
        help="Apply time-reversal symmetrization to the weight matrices (default: off, matching the notebook's default)",
    )
    parser.add_argument(
        "--q-errors",
        type=str,
        default=None,
        help="Path to a block-error-analysis text file with errors for the q-matrix "
             "(e.g. block_error_analysis_qstaple_1.txt), used to draw error bars/bands "
             "on the memory-analysis and memory-landscape plots (default: none)",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=None,
        help="Directory to save plots in (default: <simdir>)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Also display the plots interactively (in addition to saving them)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# infretis reading and weight-matrix construction
# (ported from inftools/examples/staple_test/infretis_istar_workflow.ipynb)
# ---------------------------------------------------------------------------

def _parse_ptype_direction(ptype):
    """Direction of a path from its ptype string: 1 forward, -1 backward."""
    if ptype in ["RMR", "RML", "LMR", "LML", "L*L", "R*R"]:
        return 1
    match = re.match(r"^(\d+)([LR]M[LR])(\d+)$", ptype)
    if match:
        a, b = int(match.group(1)), int(match.group(3))
        return 1 if a <= b else -1
    return 1


def calculate_infretis_weights(data_file, toml_file, nskip=0):
    """
    Read infretis_data.txt + infretis.toml and compute per-path weights.

    Returns a dict with 'interfaces', 'path_data' (D), 'weights_matrix',
    'has_ptype' and 'lm1' (lambda_minus_one, if set in the toml).
    """
    with open(toml_file, "rb") as f:
        toml_config = tomllib.load(f)
    interfaces = toml_config["simulation"]["interfaces"]
    lm1 = toml_config.get("simulation", {}).get("tis_set", {}).get("lambda_minus_one", None)
    if lm1 is not None:
        print(f"read lm1 from toml: {lm1}")

    # infretis_data.txt has 2*n_interfaces + 5 columns (pnr, len, maxop, minop,
    # ptype, then path_f and path_w per interface). Some rows in practice have
    # extra trailing whitespace-separated junk, which makes a plain
    # np.loadtxt(..., dtype=str) (no usecols) see a ragged/inconsistent column
    # count and fail; pin usecols to the expected width to avoid that.
    n_cols = 2 * len(interfaces) + 5
    data = np.loadtxt(data_file, dtype=str, usecols=np.arange(n_cols))
    data = data[nskip:]

    has_ptype = False
    ptype_col = None
    for col in range(data.shape[1]):
        for val in data[:10, col]:
            if isinstance(val, str) and (
                re.match(r"\d+[LR]M[LR]\d+", val)
                or val in ["RMR", "RML", "LMR", "LML", "L*L", "R*R"]
            ):
                has_ptype = True
                ptype_col = col
                break
        if has_ptype:
            break

    if has_ptype:
        print(f"Found ptype information in column {ptype_col}")
        directions, start_interfaces, end_interfaces = [], [], []
        for row in data:
            ptype = row[ptype_col]
            direction = _parse_ptype_direction(ptype)
            directions.append(direction)
            if ptype in ["RMR", "RML", "LMR", "LML", "L*L", "R*R"]:
                start_interfaces.append(-1)
                end_interfaces.append(0)
            else:
                match = re.match(r"^(\d+)([LR]M[LR])(\d+)$", ptype)
                if match:
                    start_interfaces.append(int(match.group(1)))
                    end_interfaces.append(int(match.group(3)))
                else:
                    start_interfaces.append(0)
                    end_interfaces.append(0)
    else:
        print("No ptype information found, using standard format")
        directions = start_interfaces = end_interfaces = None

    # Column layout: pnr, len, maxop, minop, [ptype,] path_f x N, path_w x N
    data[data == "----"] = "0.0"
    non_zero_paths = np.full(len(data), True)

    D = {}
    D["pnr"] = data[non_zero_paths, 0:1].astype(int)
    D["len"] = data[non_zero_paths, 1:2].astype(int)
    D["maxop"] = data[non_zero_paths, 2:3].astype(float)
    D["minop"] = data[non_zero_paths, 3:4].astype(float)

    if has_ptype:
        D["ptype"] = data[non_zero_paths, ptype_col]
        D["direction"] = np.array([directions[i] for i in range(len(directions)) if non_zero_paths[i]])
        D["start_intf"] = np.array([start_interfaces[i] for i in range(len(start_interfaces)) if non_zero_paths[i]])
        D["end_intf"] = np.array([end_interfaces[i] for i in range(len(end_interfaces)) if non_zero_paths[i]])

    data_start_col = ptype_col + 1 if has_ptype else 4
    D["path_f"] = data[non_zero_paths, data_start_col: data_start_col + len(interfaces)].astype(float)
    D["path_w"] = data[non_zero_paths, data_start_col + len(interfaces): data_start_col + 2 * len(interfaces)].astype(float)

    w = D["path_f"] / D["path_w"]
    w[np.isnan(w)] = 0
    w = w / np.sum(w, axis=0) * np.sum(D["path_f"], axis=0)
    w[np.isnan(w)] = 0.0

    print(f"Processed {len(D['pnr'])} paths")
    print(f"Interfaces: {interfaces}")
    if has_ptype:
        print("Path type information detected:")
        print(f"  Forward paths (dir=1): {np.sum(D['direction'] == 1)}")
        print(f"  Backward paths (dir=-1): {np.sum(D['direction'] == -1)}")
        print(f"  Other paths (dir=0): {np.sum(D['direction'] == 0)}")

    return {
        "interfaces": interfaces,
        "path_data": D,
        "weights_matrix": w,
        "has_ptype": has_ptype,
        "lm1": lm1,
    }


def compute_weight_matrices_weights(weight_results, n_int=None, tr=False):
    """
    Build the {ensemble_i: 2D weight matrix[start_interface, end_interface]}
    dict ("w_path") from infretis path data, following istar_analysis.py's
    weight-counting rules for the iSTAR ensembles.
    """
    D = weight_results["path_data"]
    interfaces = weight_results["interfaces"] if n_int is None else weight_results["interfaces"][:n_int]
    has_ptype = weight_results.get("has_ptype", False)
    if not has_ptype or "start_intf" not in D:
        raise ValueError("Path data must contain ptype-derived direction info for istar_analysis-style computation.")

    n_interfaces = len(interfaces)
    n_ensembles = len(interfaces)
    n_paths = len(D["pnr"])

    weight_matrix_3d = {i: np.zeros((n_interfaces, n_interfaces)) for i in range(n_ensembles)}
    count_matrix_3d = {i: np.zeros((n_interfaces, n_interfaces)) for i in range(n_ensembles)}

    normalization_factor = np.nan_to_num(
        np.sum(D["path_f"], axis=0) / np.sum(np.nan_to_num(D["path_f"] / D["path_w"]), axis=0)
    )

    for path_idx in range(n_paths):
        start_intf = int(D["start_intf"][path_idx])
        end_intf = int(D["end_intf"][path_idx])
        direction = int(D["direction"][path_idx])

        if (
            ((start_intf < 0 or start_intf >= n_interfaces) and (end_intf < 0 or end_intf >= n_interfaces))
            or (start_intf >= n_interfaces - 1 and end_intf >= n_interfaces - 1 and n_int is not None)
        ):
            continue
        start_intf = min(start_intf, n_interfaces - 1)
        end_intf = min(end_intf, n_interfaces - 1)

        path_f_k = D["path_f"][path_idx, :]
        path_w_k = np.array([min(D["path_w"][path_idx, i], 1.0) for i in range(len(D["path_w"][path_idx, :]))])
        weight_k = np.nan_to_num(path_f_k / path_w_k) if np.sum(path_w_k) != 0 else np.zeros_like(path_f_k)
        weight_k *= normalization_factor

        if weight_k[0] == 0:
            for i in range(1, n_ensembles):
                if np.sum(path_w_k) == 0 or np.sum(path_f_k) == 0:
                    continue
                weight = weight_k[i]
                j, k = start_intf, end_intf
                should_count = False

                if j == k:
                    if j == 0:
                        should_count = True
                    elif j == len(interfaces) - 1:
                        k = len(interfaces) - 2
                        should_count = True
                elif j < k:
                    if j == 0 and k == 1:
                        should_count = True if i == 2 else (direction == 1)
                    elif j == len(interfaces) - 2 and k == len(interfaces) - 1:
                        should_count = True
                    elif i - 1 in [j, k] and 1 < i < len(interfaces):
                        should_count = True
                    else:
                        should_count = direction == 1
                else:
                    if j == 1 and k == 0:
                        should_count = True if i == 2 else (direction == -1)
                    elif j == len(interfaces) - 1 and k == len(interfaces) - 2:
                        should_count = True
                    elif i - 1 in [j, k] and 1 < i < len(interfaces):
                        should_count = True
                    else:
                        should_count = direction == -1

                if should_count:
                    weight_matrix_3d[i][j, k] += weight
                    count_matrix_3d[i][j, k] += 1
        else:
            weight_matrix_3d[0][0, 0] += weight_k[0]
            count_matrix_3d[0][0, 0] += 1

    if tr:
        for i in range(n_ensembles):
            weight_matrix_3d[i] = (weight_matrix_3d[i] + weight_matrix_3d[i].T) / 2.0
            count_matrix_3d[i] = (count_matrix_3d[i] + count_matrix_3d[i].T) / 2.0

    weight_matrix_2d = sum(weight_matrix_3d[i] for i in range(n_ensembles))

    return {
        "weight_matrix_3d": weight_matrix_3d,
        "count_matrix_3d": count_matrix_3d,
        "weight_matrix_2d": weight_matrix_2d,
        "interfaces": interfaces,
        "n_interfaces": n_interfaces,
        "n_ensembles": n_ensembles,
        "total_paths_processed": n_paths,
    }


def compute_plocs_efficient(weight_results, get_transition_probs_weights, construct_M_istar, global_pcross_msm_star):
    """
    Vectorized crossing-probability profile: ploc(lambda_i) for i = 0..N-1,
    recomputing the weight matrices for each growing subset of interfaces.
    """
    D = weight_results["path_data"]
    all_interfaces = weight_results["interfaces"]
    N = len(all_interfaces)

    path_f = D["path_f"]
    path_w_c = np.minimum(D["path_w"], 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(path_w_c != 0, path_f / path_w_c, 0.0)
    denom = np.sum(ratio, axis=0)
    numer = np.sum(path_f, axis=0)
    norm_factor = np.where(denom != 0, numer / denom, 0.0)
    weight_k = ratio * norm_factor[np.newaxis, :]

    j_raw = D["start_intf"].astype(int)
    k_raw = D["end_intf"].astype(int)

    is_ens0 = weight_k[:, 0] != 0
    not_ens0 = ~is_ens0
    ens0_weight = float(np.sum(weight_k[is_ens0, 0]))

    plocs = [1.0]
    for n_int in range(2, N + 1):
        L = n_int - 1

        j_out = (j_raw < 0) | (j_raw >= n_int)
        k_out = (k_raw < 0) | (k_raw >= n_int)
        skip = (j_out & k_out) | ((j_raw >= L) & (k_raw >= L))
        valid = ~skip

        j = np.clip(j_raw, 0, L)
        k = np.clip(k_raw, 0, L)

        wm3d = {ens: np.zeros((n_int, n_int)) for ens in range(n_int)}
        wm3d[0][0, 0] = ens0_weight

        for ens in range(1, n_int):
            w_ens = weight_k[:, ens]
            mask = valid & not_ens0 & (w_ens != 0)
            if not np.any(mask):
                continue
            jm, km, wm = j[mask], k[mask], w_ens[mask]

            self_mask = jm == km
            if np.any(self_mask):
                wm3d[ens][0, 0] += np.sum(wm[self_mask & (jm == 0)])
                if L > 0:
                    wm3d[ens][L, L - 1] += np.sum(wm[self_mask & (jm == L)])

            off = ~self_mask
            if np.any(off):
                np.add.at(wm3d[ens], (jm[off], km[off]), wm[off])

        p, _ = get_transition_probs_weights(wm3d)
        M = construct_M_istar(p, max(4, 2 * n_int), n_int)
        _, _, y1, _ = global_pcross_msm_star(M)
        plocs.append(float(y1[0][0]))
        print(f"  n_int={n_int:2d}  lambda={all_interfaces[L]:.4f}  ploc={plocs[-1]:.8f}")

    return plocs


# ---------------------------------------------------------------------------
# Memory analysis (ported from
# inftools/examples/staple_test/istar_memory_analysis.py — a PathEnsemble-free
# variant of tistools' own memory_analysis()/plot_memory_analysis())
# ---------------------------------------------------------------------------

def memory_analysis(w_path, tr=False):
    n_int = list(w_path.values())[0].shape[0]
    q_k = np.zeros([2, n_int - 1, n_int, n_int])
    for ens in range(1, n_int):
        if ens not in w_path:
            continue
        w_ens = w_path[ens].copy()
        if tr:
            w_ens += w_ens.T
        for i in range(w_ens.shape[0]):
            for k in range(w_ens.shape[0]):
                counts = np.zeros(2)
                if i == k:
                    if i == 0:
                        q_k[0][ens - 1][i][k] = 1
                    continue
                elif i == 0 and k == 1 and ens == 1:
                    q_k[0][ens - 1][i][k] = np.sum(w_ens[i][k:]) / np.sum(w_ens[i][k - 1:])
                    q_k[1][ens - 1][i][k] = np.sum(w_ens[i][k - 1:])
                    continue
                elif i < k:
                    if i <= ens <= k:
                        counts += [np.sum(w_ens[i][k:]), np.sum(w_ens[i][k - 1:])]
                elif i > k:
                    if k + 2 <= ens <= i + 1:
                        counts += [np.sum(w_ens[i][: k + 1]), np.sum(w_ens[i][: k + 2])]
                q_k[0][ens - 1][i][k] = counts[0] / counts[1] if counts[1] > 0 else np.nan
                q_k[1][ens - 1][i][k] = counts[1]

    q_tot = np.ones([2, n_int, n_int])
    for i in range(n_int):
        for k in range(n_int):
            counts = np.zeros(2)
            if i == k:
                q_tot[0][i][k] = 1 if i == 0 else 0
                continue
            elif i == 0 and k == 1:
                if 1 in w_path:
                    q_tot[0][i][k] = np.sum(w_path[1][i][k:]) / np.sum(w_path[1][i][k - 1:])
                    q_tot[1][i][k] = np.sum(w_path[1][i][k - 1:])
                continue
            elif i < k:
                for pe_i in range(i + 1, k + 1):
                    if pe_i > n_int - 1:
                        break
                    if pe_i in w_path:
                        counts += [np.sum(w_path[pe_i][i][k:]), np.sum(w_path[pe_i][i][k - 1:])]
            elif i > k:
                for pe_i in range(k + 2, i + 2):
                    if pe_i > n_int - 1:
                        break
                    if pe_i in w_path:
                        counts += [np.sum(w_path[pe_i][i][: k + 1]), np.sum(w_path[pe_i][i][: k + 2])]
            q_tot[0][i][k] = counts[0] / counts[1] if counts[1] > 0 else np.nan
            q_tot[1][i][k] = counts[1]

    return q_k, q_tot


def calculate_memory_effect_index(q_probs, q_weights, q_errors=None, min_samples=5, max_error=0.1):
    """
    Normalized memory effect index: how much the observed spread in q across
    starting interfaces exceeds the spread expected from binomial sampling
    noise alone, i.e. std(q) / sqrt(mean(q) * (1 - mean(q))). A value near
    100% means the variation looks like pure statistical noise (no memory
    effect); well above 100% indicates a real, non-Markovian effect.
    Matches tistools.calculate_memory_effect_index (lib/istar_analysis.py).
    """
    n_interfaces = q_probs.shape[0]
    forward_variation = np.full(n_interfaces, np.nan)
    forward_variation_error = np.full(n_interfaces, np.nan)
    forward_sample_sizes = np.zeros(n_interfaces, dtype=int)
    backward_variation = np.full(n_interfaces, np.nan)
    backward_variation_error = np.full(n_interfaces, np.nan)
    backward_sample_sizes = np.zeros(n_interfaces, dtype=int)

    for k in range(1, n_interfaces):
        q_values, weights, q_errors_k = [], [], []
        for i in range(max(1, k - 1)):
            if (not np.isnan(q_probs[i, k]) and q_weights[i, k] >= min_samples
                    and (q_errors is None or (not np.isnan(q_errors[i, k]) and q_errors[i, k] <= max_error))):
                q_values.append(q_probs[i, k])
                weights.append(q_weights[i, k])
                if q_errors is not None and not np.isnan(q_errors[i, k]):
                    q_errors_k.append(q_errors[i, k])
                else:
                    q_errors_k.append(np.sqrt(q_probs[i, k] * (1 - q_probs[i, k]) / q_weights[i, k]))
        if len(q_values) >= 2:
            q_values, weights, q_errors_arr = np.array(q_values), np.array(weights), np.array(q_errors_k)
            forward_sample_sizes[k] = np.sum(weights)
            n = len(q_values)
            mean_q = np.average(q_values, weights=weights)
            std_dev = np.sqrt(np.cov(q_values, aweights=weights))
            var_binomial = mean_q * (1 - mean_q)
            if var_binomial > 1e-12:
                denom = np.sqrt(var_binomial)
                forward_variation[k] = (std_dev / denom) * 100
                if std_dev > 0 and not np.any(np.isnan(q_errors_arr)):
                    term1 = (q_values - mean_q) / (denom * (n - 1) * std_dev)
                    term2 = (std_dev * (1 - 2 * mean_q)) / (2 * n * (denom ** 3))
                    partial_derivs = term1 - term2
                    forward_variation_error[k] = np.sqrt(np.sum((partial_derivs * q_errors_arr) ** 2)) * 100

    for k in range(n_interfaces - 1):
        q_values, weights, q_errors_k = [], [], []
        for i in range(k + 2, n_interfaces):
            if (not np.isnan(q_probs[i, k]) and q_weights[i, k] >= min_samples
                    and (q_errors is None or (not np.isnan(q_errors[i, k]) and q_errors[i, k] <= max_error))):
                q_values.append(q_probs[i, k])
                weights.append(q_weights[i, k])
                if q_errors is not None and not np.isnan(q_errors[i, k]):
                    q_errors_k.append(q_errors[i, k])
                else:
                    q_errors_k.append(np.sqrt(q_probs[i, k] * (1 - q_probs[i, k]) / q_weights[i, k]))
        if len(q_values) >= 2:
            q_values, weights, q_errors_arr = np.array(q_values), np.array(weights), np.array(q_errors_k)
            backward_sample_sizes[k] = np.sum(weights)
            n = len(q_values)
            mean_q = np.average(q_values, weights=weights)
            std_dev = np.sqrt(np.cov(q_values, aweights=weights))
            var_binomial = mean_q * (1 - mean_q)
            if var_binomial > 1e-12:
                denom = np.sqrt(var_binomial)
                backward_variation[k] = (std_dev / denom) * 100
                if std_dev > 0 and not np.any(np.isnan(q_errors_arr)):
                    term1 = (q_values - mean_q) / (denom * (n - 1) * std_dev)
                    term2 = (std_dev * (1 - 2 * mean_q)) / (2 * n * (denom ** 3))
                    partial_derivs = term1 - term2
                    backward_variation_error[k] = np.sqrt(np.sum((partial_derivs * q_errors_arr) ** 2)) * 100

    return {
        "forward_variation": forward_variation,
        "forward_variation_error": forward_variation_error,
        "backward_variation": backward_variation,
        "backward_variation_error": backward_variation_error,
        "forward_sample_sizes": forward_sample_sizes,
        "backward_sample_sizes": backward_sample_sizes,
    }


def calculate_memory_effect_index_corrected(q_probs, q_weights, q_errors=None, min_samples=5,
                                           n_eff=None, verbose=True):
    """
    Memory effect index, corrected for sampling noise::

        M_k = sqrt(max(0, s_k^2 - sigma_k^2)) / sqrt(q_mean (1 - q_mean))   (in percent)

    s_k^2 is the weighted variance of q(i,k) over starting turns i and
    sigma_k^2 the sampling-noise floor, so a purely statistical spread gives
    M_k = 0 rather than a spurious memory signal. Same normalisation as
    calculate_memory_effect_index above, which is what keeps the scale bounded
    and the threshold comparable between systems.

    The subtraction is in variance, so the floor must reach ~44% of the
    uncorrected index to change it by 10%; it mainly zeroes marginal regions
    rather than shifting the largest one.

    Mirrors tistools.calculate_memory_effect_index_corrected.
        """
    n_interfaces = q_probs.shape[0]
    out = {}
    for direction in ("forward", "backward"):
        eps = np.full(n_interfaces, np.nan)
        eps_error = np.full(n_interfaces, np.nan)
        floor = np.full(n_interfaces, np.nan)
        sizes = np.zeros(n_interfaces, dtype=int)

        k_range = range(1, n_interfaces) if direction == "forward" else range(n_interfaces - 1)
        for k in k_range:
            i_range = range(max(1, k - 1)) if direction == "forward" else range(k + 2, n_interfaces)
            q_values, weights, errs = [], [], []
            for i in i_range:
                if np.isnan(q_probs[i, k]) or q_weights[i, k] < min_samples:
                    continue
                if q_errors is not None:
                    if np.isnan(q_errors[i, k]):
                        continue
                    err = q_errors[i, k]
                else:
                    # The binomial fallback treats every MC step as an
                    # independent sample; n_eff (an effective-count array, or a
                    # scalar statistical inefficiency to divide by) corrects it.
                    if n_eff is None:
                        n_use = q_weights[i, k]
                    elif np.isscalar(n_eff):
                        n_use = q_weights[i, k] / max(float(n_eff), 1e-12)
                    else:
                        n_use = n_eff[i, k]
                    err = np.sqrt(q_probs[i, k] * (1 - q_probs[i, k]) / max(n_use, 1e-12))
                q_values.append(q_probs[i, k])
                weights.append(q_weights[i, k])
                errs.append(err)
            if len(q_values) < 2:
                continue

            q_values, weights, sigma = np.array(q_values), np.array(weights), np.array(errs)
            q_mean = np.average(q_values, weights=weights)
            var_binomial = q_mean * (1.0 - q_mean)
            if not np.isfinite(q_mean) or var_binomial <= 1e-12:
                continue
            denom = np.sqrt(var_binomial)
            var_obs = float(np.cov(q_values, aweights=weights))
            var_noise = float(np.average(sigma ** 2, weights=weights))
            s_corr = np.sqrt(max(var_obs - var_noise, 0.0))

            eps[k] = (s_corr / denom) * 100
            floor[k] = (np.sqrt(var_noise) / denom) * 100
            sizes[k] = int(np.sum(weights))
            if s_corr > 0 and not np.any(np.isnan(sigma)):
                n = len(q_values)
                term1 = (q_values - q_mean) / (denom * (n - 1) * s_corr)
                term2 = (s_corr * (1 - 2 * q_mean)) / (2 * n * (denom ** 3))
                partial_derivs = term1 - term2
                eps_error[k] = float(np.sqrt(np.sum((partial_derivs * sigma) ** 2))) * 100

        out[f"{direction}_variation"] = eps
        out[f"{direction}_variation_error"] = eps_error
        out[f"{direction}_floor"] = floor
        out[f"{direction}_sample_sizes"] = sizes
        out[f"{direction}_total"] = float(np.nansum(eps))

    if verbose:
        print(f"Memory index (noise-corrected, summed over interfaces): "
              f"forward {out['forward_total']:.1f}%, backward {out['backward_total']:.1f}%")
    return out


def estimate_free_energy_differences(interfaces, q_matrix, q_weights=None, min_samples=5):
    n_interfaces = len(interfaces)
    delta_G = np.full((n_interfaces, n_interfaces), np.nan)

    def is_valid_q(i, k):
        return not np.isnan(q_matrix[i, k]) and (q_weights is None or q_weights[i, k] >= min_samples) and abs(i - k) >= 2

    for i in range(n_interfaces - 1):
        forward_estimate = backward_estimate = None
        for start in range(i - 1, -1, -1):
            if is_valid_q(start, i + 1):
                q_fw = q_matrix[start, i + 1]
                dist_i_to_ip1 = interfaces[i + 1] - interfaces[i]
                dist_im1_to_i = interfaces[i] - interfaces[i - 1] if i > 0 else dist_i_to_ip1
                geo_q = dist_im1_to_i / (dist_i_to_ip1 + dist_im1_to_i) if (dist_i_to_ip1 + dist_im1_to_i) > 0 else 1.0
                if i > 0 and 0 < q_fw < 1 and 0 < geo_q < 1:
                    dG = -np.log(q_fw / (1 - q_fw)) - (-np.log(geo_q / (1 - geo_q)))
                elif 0 < q_fw < 1:
                    dG = -np.log(q_fw / (1 - q_fw))
                else:
                    dG = 0
                forward_estimate = np.nan_to_num(dG, posinf=6.0, neginf=-6.0)
                break
        for start in range(i + 2, n_interfaces):
            if is_valid_q(start, i):
                q_bw = q_matrix[start, i]
                if i + 1 < n_interfaces - 1:
                    dist_ip1_to_i = interfaces[i + 1] - interfaces[i]
                    dist_ip2_to_ip1 = interfaces[i + 2] - interfaces[i + 1]
                    geo_q = dist_ip2_to_ip1 / (dist_ip2_to_ip1 + dist_ip1_to_i) if (dist_ip2_to_ip1 + dist_ip1_to_i) > 0 else 1.0
                    if 0 < q_bw < 1 and 0 < geo_q < 1:
                        dG = np.log(q_bw / (1 - q_bw)) - np.log(geo_q / (1 - geo_q))
                    else:
                        dG = 0
                elif 0 < q_bw < 1:
                    dG = np.log(q_bw / (1 - q_bw))
                else:
                    dG = 0
                backward_estimate = np.nan_to_num(dG, posinf=6.0, neginf=-6.0)
                break
        if forward_estimate is not None and backward_estimate is not None:
            combined = (forward_estimate + backward_estimate) / 2.0
            delta_G[i, i + 1], delta_G[i + 1, i] = combined, -combined
        elif forward_estimate is not None:
            delta_G[i, i + 1], delta_G[i + 1, i] = forward_estimate, -forward_estimate
        elif backward_estimate is not None:
            delta_G[i, i + 1], delta_G[i + 1, i] = backward_estimate, -backward_estimate
    return delta_G


def calculate_diffusive_reference(interfaces, q_matrix, q_weights=None, min_samples=5):
    n_interfaces = len(interfaces)
    diffusive_q = np.full((n_interfaces, n_interfaces), np.nan)
    delta_G = estimate_free_energy_differences(interfaces, q_matrix, q_weights, min_samples)
    for i in range(n_interfaces):
        diffusive_q[i, i] = 0.0

    for k in range(1, n_interfaces):
        for i in range(k):
            if i == k - 1:
                diffusive_q[i, k] = 1.0
                continue
            ref_prob = 0.5
            if not np.isnan(delta_G[k - 1, k]):
                if k > 1:
                    d1 = interfaces[k] - interfaces[k - 1]
                    d2 = interfaces[k - 1] - interfaces[k - 2]
                    geo_q = d2 / (d1 + d2) if (d1 + d2) > 0 else 0.5
                    geo_q = max(0.001, min(0.999, geo_q))
                    ref_prob = 1.0 / (1.0 + np.exp(delta_G[k - 1, k]) * (1 - geo_q) / geo_q)
                else:
                    ref_prob = 1.0 / (1.0 + np.exp(delta_G[k - 1, k]))
            diffusive_q[i, k] = ref_prob

    for k in range(n_interfaces - 1):
        for i in range(k + 1, n_interfaces):
            if i == k + 1:
                diffusive_q[i, k] = 1.0
                continue
            ref_prob = 0.5
            if not np.isnan(delta_G[k, k + 1]):
                if k + 2 < n_interfaces:
                    d1 = interfaces[k + 1] - interfaces[k]
                    d2 = interfaces[k + 2] - interfaces[k + 1]
                    geo_q = d2 / (d1 + d2) if (d1 + d2) > 0 else 0.5
                    geo_q = max(0.001, min(0.999, geo_q))
                    ref_prob = 1.0 / (1.0 + np.exp(-delta_G[k, k + 1]) * (1 - geo_q) / geo_q)
                else:
                    ref_prob = 1.0 / (1.0 + np.exp(-delta_G[k, k + 1]))
            diffusive_q[i, k] = ref_prob

    return np.clip(diffusive_q, 0.0, 1.0)


def analyze_momentum_vs_free_energy(interfaces, q_matrix, q_weights=None, min_samples=5, momentum_threshold=0.2):
    n_interfaces = len(interfaces)
    delta_G = estimate_free_energy_differences(interfaces, q_matrix, q_weights, min_samples)
    diffusive_q = calculate_diffusive_reference(interfaces, q_matrix, q_weights, min_samples)
    momentum_effects = np.full_like(q_matrix, np.nan)
    for i in range(n_interfaces):
        for k in range(n_interfaces):
            if i != k and not np.isnan(q_matrix[i, k]) and not np.isnan(diffusive_q[i, k]):
                if q_weights is None or q_weights[i, k] >= min_samples:
                    momentum_effects[i, k] = q_matrix[i, k] - diffusive_q[i, k]

    momentum_significance = np.abs(np.nan_to_num(momentum_effects)) > momentum_threshold
    momentum_significance &= ~np.isnan(momentum_effects)

    pair_classification = []
    for i in range(n_interfaces - 1):
        fwd = momentum_effects[i - 1, i + 1] if i > 0 and not np.isnan(momentum_effects[i - 1, i + 1]) else 0
        bwd = momentum_effects[i + 2, i] if i + 2 < n_interfaces and not np.isnan(momentum_effects[i + 2, i]) else 0
        fwd_sig = momentum_significance[i - 1, i + 1] if i > 0 else False
        bwd_sig = momentum_significance[i + 2, i] if i + 2 < n_interfaces else False
        avg_effect = (abs(fwd) + abs(bwd)) / 2 if (i > 0 and i + 2 < n_interfaces) else (abs(fwd) if i > 0 else abs(bwd))
        if fwd_sig or bwd_sig:
            if abs(fwd + bwd) < 0.2 * (abs(fwd) + abs(bwd) + 1e-12):
                pair_classification.append("symmetric_momentum")
            elif avg_effect > momentum_threshold * 2:
                pair_classification.append("strong_momentum")
            else:
                pair_classification.append("momentum_dominated")
        else:
            pair_classification.append("free_energy_dominated")

    avg_abs_momentum = np.nanmean(np.abs(momentum_effects))
    sum_momentum = np.nansum(momentum_effects)
    if np.isnan(avg_abs_momentum):
        overall_classification = "insufficient_data"
    elif avg_abs_momentum < momentum_threshold:
        overall_classification = "free_energy_dominated"
    elif abs(sum_momentum) < 0.2 * np.nansum(np.abs(momentum_effects)):
        overall_classification = "symmetric_momentum_dominated"
    else:
        overall_classification = "directional_momentum_dominated"

    return {
        "free_energy_differences": delta_G,
        "diffusive_probabilities": diffusive_q,
        "momentum_effects": momentum_effects,
        "momentum_significance": momentum_significance,
        "classification": pair_classification,
        "overall_classification": overall_classification,
        "avg_momentum_effect": avg_abs_momentum,
        "avg_free_energy": np.nanmean(np.abs(delta_G)),
        "avg_probabilities": np.nanmean([np.nanmean(q_matrix[i, :]) for i in range(n_interfaces)]),
    }


def plot_q_matrix(q_probs, q_weights=None, q_errors=None, cmap_name='turbo',
                  title='Conditional crossing probabilities  q(i,k)'):
    """
    Stand-alone heatmap of the conditional committor matrix q(i,k).

    This deliberately gets a figure of its own rather than a panel in the
    multi-panel matrix figure: past a handful of interfaces the matrix needs
    the whole canvas before the cells -- and the numbers in them -- are
    readable at all. The canvas grows with the number of interfaces, the
    annotations shrink and then drop out, and the tick labels thin.

    Parameters
    ----------
    q_probs : (n, n) array
        Conditional crossing probabilities; NaN where undefined.
    q_weights : (n, n) array, optional
        Sample weights; entries with zero weight are drawn as "unsampled".
    q_errors : (n, n) array, optional
        Per-entry errors, annotated underneath the value on small matrices.
    cmap_name : str
        A high-contrast sequential map. 'turbo' is the default because the
        point here is to tell neighbouring q values apart across a large grid,
        which a low-contrast single-hue ramp does poorly.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    q_probs = np.asarray(q_probs, dtype=float)
    n = q_probs.shape[0]

    # ~0.45 in per cell on top of a fixed margin, clamped so small matrices
    # are not comically large and big ones still fit on a page.
    side = float(np.clip(0.45 * n + 3.5, 7.0, 22.0))
    fig, ax = plt.subplots(figsize=(side + 1.6, side))

    data = np.ma.masked_invalid(q_probs)
    if q_weights is not None:
        data = np.ma.masked_where(np.asarray(q_weights) <= 0, data)

    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad('#d9d8d2')          # (i,k) pairs that were never sampled
    im = ax.imshow(data, cmap=cmap, vmin=0.0, vmax=1.0,
                   interpolation='nearest', aspect='equal')
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label('q(i,k)')

    mask = np.ma.getmaskarray(data)
    if n <= 30:
        fontsize = float(np.clip(150.0 / n, 4.5, 14.0))
        rgba = cmap(np.ma.filled(data, np.nan))
        for i in range(n):
            for j in range(n):
                if mask[i, j]:
                    continue
                # Black or white chosen from the cell's own luminance, so the
                # annotation stays legible whichever colormap is in use.
                r, g, b = rgba[i, j, :3]
                lum = 0.299 * r + 0.587 * g + 0.114 * b
                txt = f"{q_probs[i, j]:.2f}"
                if (q_errors is not None and n <= 15
                        and not np.isnan(np.asarray(q_errors)[i, j])):
                    txt += f"\n$\\pm${np.asarray(q_errors)[i, j]:.2f}"
                ax.text(j, i, txt, ha='center', va='center',
                        color='black' if lum > 0.55 else 'white', fontsize=fontsize)

    # Thin separators so the eye can follow rows and columns on a big grid.
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=0.6)
    ax.tick_params(which='minor', length=0)

    step = 1 if n <= 25 else int(np.ceil(n / 25.0))
    ticks = np.arange(0, n, step)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    ax.set_yticklabels([str(t) for t in ticks])
    ax.set_xlabel('Target interface k')
    ax.set_ylabel('Starting interface i')
    ax.set_title(title, fontsize=14)
    fig.tight_layout()
    return fig


def plot_memory_analysis(q_tot, p, interfaces=None, q_errors=None):
    """PathEnsemble-free variant of tistools' plot_memory_analysis: same four
    figures (matrix heatmaps, transition probabilities + memory retention,
    free energy & momentum effects, and the full-size q(i,k) matrix), driven
    only by q_tot/p/interfaces.

    q_errors, if given, is a NumPy array of errors for the q-matrix (same
    shape as q_tot[0]) used for the memory-retention error bars in figure 2.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    import matplotlib.gridspec as gridspec
    import matplotlib.colors as colors
    import seaborn as sns

    q_probs, q_weights = q_tot[0], q_tot[1]
    n_interfaces = q_probs.shape[0]
    if interfaces is None:
        interfaces = list(range(n_interfaces))

    diff_ref = calculate_diffusive_reference(interfaces, q_probs, q_weights)

    BLUE, RED, NEUTRAL = "#2a78d6", "#e34948", "#f0efec"
    GRID_COLOR, MUTED_INK = "#e1e0d9", "#898781"
    cmap_memory = LinearSegmentedColormap.from_list("memory_effect", [(0, BLUE), (0.5, NEUTRAL), (1, RED)], N=256)

    def sequential_colors(n, cmap_name):
        cmap = plt.get_cmap(cmap_name)
        if n <= 1:
            return [colors.to_hex(cmap(0.5))]
        return [colors.to_hex(cmap(t)) for t in np.linspace(0.0, 0.9, n)]

    # ================ Figure 1: Matrix Heatmaps ================
    # The three panels share one canvas, so their cells shrink as the
    # interface count grows: scale the annotations with it and drop them once
    # the numbers would overlap. The q(i,k) matrix itself is in fig4, which
    # keeps its own full-size canvas precisely so it stays readable.
    # Give every panel a fixed ~0.35 in per matrix cell rather than squeezing an
    # arbitrary number of interfaces into a fixed canvas, so the annotations stay
    # the same readable size as the matrix grows. Past ~25 interfaces the figure
    # would get unwieldy, so there the canvas is clamped and the numbers dropped
    # (the colors still carry the pattern, and fig4 shows q(i,k) full size).
    panel_w = 0.35 * n_interfaces + 1.8
    fig1_w = float(np.clip(3.0 * panel_w, 15.0, 32.0))
    fig1_h = float(np.clip(0.35 * n_interfaces + 2.8, 6.5, 12.0))
    fig1 = plt.figure(figsize=(fig1_w, fig1_h))
    gs1 = gridspec.GridSpec(1, 3, width_ratios=[1.2, 1, 1])
    ann_fs = 8.0 if n_interfaces <= 25 else None

    ax1 = fig1.add_subplot(gs1[0])
    memory_effect = np.where(~np.isnan(q_probs) & ~np.isnan(diff_ref), q_probs - diff_ref, np.nan)
    max_effect = np.nanmax(np.abs(memory_effect))
    masked_data = np.ma.masked_invalid(memory_effect)
    im1 = ax1.imshow(masked_data, cmap=cmap_memory, vmin=-max_effect, vmax=max_effect, interpolation="none", aspect="auto")
    cbar1 = fig1.colorbar(im1, ax=ax1, label="q - q_diffusive")
    cbar1.ax.axhline(y=0.0, color=MUTED_INK, linestyle="--", linewidth=1)
    for i in range(n_interfaces):
        for j in range(n_interfaces):
            if not np.isnan(memory_effect[i, j]):
                text = f"{q_probs[i, j]:.2f}" if q_weights[i, j] > 0 else "N/A"
                color = "black" if abs(memory_effect[i, j]) < 0.3 else "white"
                if ann_fs:
                    ax1.text(j, i, text, ha="center", va="center", color=color, fontsize=ann_fs)
    ax1.set_xticks(range(n_interfaces)); ax1.set_yticks(range(n_interfaces))
    ax1.set_xlabel("Target interface k"); ax1.set_ylabel("Starting interface i")
    ax1.set_title("Memory effect (q - q_diffusive)", fontsize=12)

    ax2 = fig1.add_subplot(gs1[1])
    memory_ratio = np.where(
        (~np.isnan(q_probs)) & (~np.isnan(diff_ref)) & (diff_ref > 0) & (diff_ref < 1), q_probs / diff_ref, np.nan
    )
    im2 = ax2.imshow(memory_ratio, cmap=cmap_memory, norm=colors.LogNorm(vmin=0.1, vmax=10))
    fig1.colorbar(im2, ax=ax2, label="q / q_diffusive (log)")
    for i in range(n_interfaces):
        for j in range(n_interfaces):
            if not np.isnan(memory_ratio[i, j]) and q_weights[i, j] > 5:
                text_color = "white" if (memory_ratio[i, j] > 5 or memory_ratio[i, j] < 0.2) else "black"
                if ann_fs:
                    ax2.text(j, i, f"{memory_ratio[i, j]:.1f}", ha="center", va="center", color=text_color, fontsize=ann_fs)
    ax2.set_xticks(range(n_interfaces)); ax2.set_yticks(range(n_interfaces))
    ax2.set_xlabel("Target interface k"); ax2.set_ylabel("Starting interface i")
    ax2.set_title("Memory ratio (q / q_diffusive)", fontsize=12)

    ax3 = fig1.add_subplot(gs1[2])
    memory_asymmetry = np.full_like(p, np.nan)
    for i in range(n_interfaces):
        for j in range(n_interfaces):
            if i != j:
                memory_asymmetry[i, j] = p[i, j] - p[j, i]
    im3 = ax3.imshow(memory_asymmetry, cmap=cmap_memory, vmin=-0.5, vmax=0.5)
    fig1.colorbar(im3, ax=ax3, label="p(i→j) - p(j→i)")
    for i in range(n_interfaces):
        for j in range(n_interfaces):
            if not np.isnan(memory_asymmetry[i, j]):
                text_color = "white" if abs(memory_asymmetry[i, j]) > 0.3 else "black"
                if ann_fs:
                    ax3.text(j, i, f"{memory_asymmetry[i, j]:.2f}", ha="center", va="center", color=text_color, fontsize=ann_fs)
    ax3.set_xticks(range(n_interfaces)); ax3.set_yticks(range(n_interfaces))
    ax3.set_xlabel("Target interface j"); ax3.set_ylabel("Starting interface i")
    ax3.set_title("Transition asymmetry", fontsize=12)

    # The q(i,k) matrix used to be a fourth panel here; it now gets its own
    # full-size figure (fig4) so large interface counts stay readable.

    fig1.text(0.02, 0.02,
              "Color = deviation from diffusive (memoryless) behavior. Red: bias toward crossing. Blue: bias toward returning.",
              fontsize=9, color=MUTED_INK)
    plt.tight_layout(rect=[0, 0.06, 1, 0.94])
    fig1.suptitle("Memory effect analysis - transition matrices (infretis)", fontsize=14)

    # ================ Figure 4: the q(i,k) matrix, full size ================
    fig4 = plot_q_matrix(q_probs, q_weights=q_weights, q_errors=q_errors)

    # ================ Figure 2: Forward/Backward Probs + Memory Retention ================
    fig2 = plt.figure(figsize=(18, 12))
    gs2 = gridspec.GridSpec(2, 2, height_ratios=[1, 0.8])

    forward_targets = list(range(1, n_interfaces))
    forward_colors = sequential_colors(len(forward_targets), "viridis")
    backward_targets = list(range(n_interfaces - 1))
    backward_colors = sequential_colors(len(backward_targets), "plasma")

    ax4 = fig2.add_subplot(gs2[0, 0])
    for idx, k in enumerate(forward_targets):
        xs, ys, errs = [], [], []
        for i in range(k):
            if (i < k - 1 or (i == 0 and k == 1)) and not np.isnan(q_probs[i, k]) and q_weights[i, k] > 5:
                xs.append(interfaces[i]); ys.append(q_probs[i, k])
        if xs:
            ax4.plot(xs, ys, "o-", label=f"{k - 1 if k > 0 else k}→{k}", linewidth=2, markersize=8, color=forward_colors[idx])
    ax4.set_xlabel(r"Starting position $\lambda$"); ax4.set_ylabel("q(i,k)")
    ax4.set_title("Forward crossing probabilities", fontsize=12)
    ax4.set_ylim(0, 1.05); ax4.grid(axis="y", alpha=0.3, color=GRID_COLOR, zorder=0)
    sns.despine(ax=ax4); ax4.legend(title="Target", loc="best", fontsize=9)

    ax5b = fig2.add_subplot(gs2[0, 1])
    for idx, k in enumerate(backward_targets):
        xs, ys = [], []
        for i in range(k + 1, n_interfaces):
            if i > k + 1 and not np.isnan(q_probs[i, k]) and q_weights[i, k] > 5:
                xs.append(interfaces[i]); ys.append(q_probs[i, k])
        if xs:
            ax5b.plot(xs, ys, "o-", label=f"{k}←{k + 1}", linewidth=2, markersize=8, color=backward_colors[idx])
    ax5b.set_xlabel(r"Starting position $\lambda$"); ax5b.set_ylabel("q(i,k)")
    ax5b.set_title("Backward crossing probabilities", fontsize=12)
    ax5b.set_ylim(0, 1.05); ax5b.grid(axis="y", alpha=0.3, color=GRID_COLOR, zorder=0)
    sns.despine(ax=ax5b); ax5b.legend(title="Target", loc="best", fontsize=9)

    memory_index = calculate_memory_effect_index_corrected(q_probs, q_weights, q_errors=q_errors)

    ax6 = fig2.add_subplot(gs2[1, 0])
    valid_k_fwd = [k for k in range(1, n_interfaces) if not np.isnan(memory_index["forward_variation"][k])]
    if valid_k_fwd:
        positions = [interfaces[k] for k in valid_k_fwd]
        values = [memory_index["forward_variation"][k] for k in valid_k_fwd]
        errors = [memory_index["forward_variation_error"][k] if not np.isnan(memory_index["forward_variation_error"][k]) else 0 for k in valid_k_fwd]
        bar_colors = [forward_colors[k - 1] for k in valid_k_fwd]
        ax6.bar(positions, values, yerr=errors, color=bar_colors, alpha=0.85,
                width=np.mean(np.diff(interfaces)) * 0.7, capsize=5)
        ax6.set_xlim(min(positions) - 0.5, max(positions) + 0.5)
    ax6.set_xlabel("Target region"); ax6.set_ylabel("Memory index (%)")
    ax6.set_title("Forward memory retention", fontsize=12)
    ax6.grid(axis="y", alpha=0.3, color=GRID_COLOR, zorder=0); sns.despine(ax=ax6)

    ax7 = fig2.add_subplot(gs2[1, 1])
    valid_k_bwd = [k for k in range(n_interfaces - 1) if not np.isnan(memory_index["backward_variation"][k])]
    if valid_k_bwd:
        positions = [interfaces[k] for k in valid_k_bwd]
        values = [memory_index["backward_variation"][k] for k in valid_k_bwd]
        errors = [memory_index["backward_variation_error"][k] if not np.isnan(memory_index["backward_variation_error"][k]) else 0 for k in valid_k_bwd]
        bar_colors = [backward_colors[k] for k in valid_k_bwd]
        ax7.bar(positions, values, yerr=errors, color=bar_colors, alpha=0.85,
                width=np.mean(np.diff(interfaces)) * 0.7, capsize=5)
        ax7.set_xlim(min(positions) - 0.5, max(positions) + 0.5)
    ax7.set_xlabel("Target region"); ax7.set_ylabel("Memory index (%)")
    ax7.set_title("Backward memory retention", fontsize=12)
    ax7.grid(axis="y", alpha=0.3, color=GRID_COLOR, zorder=0); sns.despine(ax=ax7)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig2.suptitle("Transition probabilities & memory retention (infretis)", fontsize=14)

    # ================ Figure 3: Free Energy and Momentum Effects ================
    fig3 = plt.figure(figsize=(18, 14))
    gs3 = gridspec.GridSpec(2, 2, height_ratios=[1, 1])
    momentum_results = analyze_momentum_vs_free_energy(interfaces, q_probs, q_weights)

    ax8 = fig3.add_subplot(gs3[0, :])
    delta_G = momentum_results["free_energy_differences"]
    cumulative_G = np.zeros(len(interfaces))
    for i in range(1, len(interfaces)):
        valid_path = True
        for j in range(i):
            if np.isnan(delta_G[j, j + 1]):
                valid_path = False
                break
            cumulative_G[i] += delta_G[j, j + 1]
        if not valid_path:
            cumulative_G[i] = np.nan
    ax8.plot(interfaces, cumulative_G, "o-", linewidth=2.5, color=BLUE, markersize=8)
    for pos, g in zip(interfaces, cumulative_G):
        if not np.isnan(g):
            ax8.annotate(f"{g:.2f}", (pos, g), xytext=(0, 8), textcoords="offset points",
                         ha="center", va="bottom", fontsize=9, color=MUTED_INK)
    ax8.set_xlabel(r"Interface Position ($\lambda$)", fontsize=12)
    ax8.set_ylabel(r"Free Energy G($\lambda$) (kT)", fontsize=12)
    ax8.set_title("Free energy profile", fontsize=13)
    ax8.grid(True, alpha=0.3, linestyle="--", color=GRID_COLOR, zorder=0)
    ax8.fill_between(interfaces, 0, cumulative_G, alpha=0.15, color=BLUE)
    sns.despine(ax=ax8)

    ax9 = fig3.add_subplot(gs3[1, 0])
    diffusive_q = momentum_results["diffusive_probabilities"]
    momentum_effects = momentum_results["momentum_effects"]
    momentum_significance = momentum_results["momentum_significance"]
    valid_points = []
    for i in range(len(interfaces)):
        for j in range(len(interfaces)):
            if (abs(i - j) >= 2 and 0 < i < len(interfaces) - 1 and 0 < j < len(interfaces) - 1
                    and not np.isnan(diffusive_q[i, j]) and not np.isnan(momentum_effects[i, j])
                    and q_weights[i, j] >= 5):
                observed = diffusive_q[i, j] * (1 + momentum_effects[i, j])
                valid_points.append((diffusive_q[i, j], observed, momentum_significance[i, j], f"{i}→{j}"))
    if valid_points:
        x_vals, y_vals, significance, labels = zip(*valid_points)
        max_val = max(max(x_vals), max(y_vals)) * 1.1
        min_val = min(min(x_vals), min(y_vals)) * 0.9
        ax9.plot([min_val, max_val], [min_val, max_val], "--", color=MUTED_INK, alpha=0.6)
        for x, y, sig in zip(x_vals, y_vals, significance):
            ax9.scatter(x, y, color=(RED if sig else BLUE), s=50, alpha=0.8, zorder=3)
        for x, y, sig, label in zip(x_vals, y_vals, significance, labels):
            dx = 10 if x < 0.5 * (min_val + max_val) else -30
            dy = 10 if y < 0.5 * (min_val + max_val) else -15
            ax9.annotate(label, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=8,
                         bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=GRID_COLOR, alpha=0.85))
        ax9.set_xlim(min_val, max_val); ax9.set_ylim(min_val, max_val)
        ax9.scatter([], [], color=BLUE, label="Free Energy Dominated")
        ax9.scatter([], [], color=RED, label="Momentum Effects")
        ax9.legend(fontsize=9)
    else:
        ax9.text(0.5, 0.5, "Insufficient data (interior interfaces only)", ha="center", va="center", transform=ax9.transAxes)
    ax9.set_xlabel("Diffusive Probability (Free Energy Model)", fontsize=12)
    ax9.set_ylabel("Observed Probability", fontsize=12)
    ax9.set_title("Observed vs. diffusive probabilities", fontsize=13)
    ax9.grid(True, alpha=0.3, color=GRID_COLOR, zorder=0); sns.despine(ax=ax9)

    ax10 = fig3.add_subplot(gs3[1, 1])
    classification = momentum_results["classification"]
    class_colors = {"free_energy_dominated": BLUE, "momentum_dominated": RED, "strong_momentum": "#8c1f1f"}
    boundary_color = "#d8d7d2"
    n_intervals = len(interfaces) - 1
    modified_colors = [
        boundary_color if i in (0, n_intervals - 1) else class_colors.get(classification[i], MUTED_INK)
        for i in range(n_intervals)
    ]
    x = np.arange(n_intervals)
    ax10.bar(x, [1] * n_intervals, color=modified_colors, alpha=0.85, width=0.7)
    for i in range(n_intervals):
        if 0 < i < n_intervals - 1:
            text_color = "white" if modified_colors[i] in (RED, class_colors["strong_momentum"]) else "black"
            ax10.text(i, 0.5, classification[i].replace("_", "\n"), ha="center", va="center",
                      fontsize=9, color=text_color, fontweight="bold")
    ax10.set_xticks(x); ax10.set_xticklabels([f"{i}→{i + 1}" for i in range(n_intervals)])
    ax10.set_yticks([]); ax10.set_xlabel("Interface Pair", fontsize=12)
    ax10.set_title("Interface pair classification", fontsize=13)
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=class_colors["free_energy_dominated"], label="Free Energy Dominated"),
        Patch(facecolor=class_colors["momentum_dominated"], label="Momentum Dominated"),
        Patch(facecolor=class_colors["strong_momentum"], label="Strong Momentum Effects"),
        Patch(facecolor=boundary_color, label="Boundary (not classified)"),
    ]
    ax10.legend(handles=legend_elements, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=9)

    plt.tight_layout(rect=[0, 0.04, 1, 0.95])
    overall_class = momentum_results["overall_classification"].replace("_", " ").title()
    fig3.suptitle("Free energy & momentum effects (infretis)", fontsize=15)
    fig3.text(0.02, 0.01,
              f"{overall_class}  ·  mean momentum effect = {momentum_results['avg_momentum_effect']:.2f}"
              f"  ·  mean ΔG = {momentum_results['avg_free_energy']:.2f} kT",
              fontsize=9, color=MUTED_INK)

    return fig1, fig2, fig3, fig4


def main():
    args = parse_args()

    try:
        from tistools import (
            get_transition_probs_weights,
            construct_M_istar,
            global_pcross_msm_star,
            plot_memory_landscape,
            read_block_errors,
        )
    except ImportError as e:
        print(f"Error: Could not import tistools: {e}", file=sys.stderr)
        print("Make sure tistools is installed or in your PYTHONPATH.", file=sys.stderr)
        sys.exit(1)

    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    simdir = Path(args.simdir).resolve()
    if not simdir.exists():
        print(f"Error: Directory {simdir} does not exist", file=sys.stderr)
        sys.exit(1)

    outdir = Path(args.outdir).resolve() if args.outdir else simdir
    outdir.mkdir(parents=True, exist_ok=True)

    data_file = args.data_file or str(simdir / "infretis_data.txt")
    toml_file = args.toml_file or str(simdir / "infretis.toml")
    for f in (data_file, toml_file):
        if not Path(f).exists():
            print(f"Error: File not found: {f}", file=sys.stderr)
            sys.exit(1)

    # ---------------------------------------------------------------------
    # Reading (infretis_data.txt + infretis.toml)
    # ---------------------------------------------------------------------
    print("=== Reading infretis data ===")
    weight_results = calculate_infretis_weights(data_file, toml_file, nskip=args.nskip)
    interfaces = weight_results["interfaces"]
    n_int = len(interfaces)

    print("\n=== Computing weight matrices ===")
    weight_matrices_result = compute_weight_matrices_weights(weight_results, tr=args.tr)
    w_path = weight_matrices_result["weight_matrix_3d"]

    # ---------------------------------------------------------------------
    # Crossing-probability profile
    # ---------------------------------------------------------------------
    print("\n=== Crossing-probability profile ===")
    plocMSM = compute_plocs_efficient(weight_results, get_transition_probs_weights, construct_M_istar, global_pcross_msm_star)

    fig, ax = plt.subplots()
    ax.set_yscale("log")
    ax.errorbar(range(n_int), plocMSM, fmt="-o", c="b", ecolor="r", capsize=6, label="StapleTIS")
    ax.set_xlabel("Interface index")
    ax.set_ylabel(r"$P_A(\lambda_i|\lambda_A)$")
    ax.set_xticks(np.arange(n_int))
    ax.legend()
    fig.tight_layout()
    crossing_plot_file = outdir / "crossing_probability_profile.png"
    fig.savefig(crossing_plot_file, dpi=200)
    print(f"Crossing-probability profile saved to {crossing_plot_file}")

    # ---------------------------------------------------------------------
    # Memory analysis
    # ---------------------------------------------------------------------
    print("\n=== Memory analysis ===")
    p, q = get_transition_probs_weights(w_path)
    q_k, q_tot = memory_analysis(w_path, tr=False)

    q_errors = None
    if args.q_errors:
        try:
            q_errors = read_block_errors(args.q_errors, q_tot[0].shape)
        except Exception as e:
            print(f"Warning: could not load --q-errors '{args.q_errors}': {e}; ignoring.", file=sys.stderr)

    fig1, fig2, fig3, fig4 = plot_memory_analysis(q_tot, p, interfaces=interfaces, q_errors=q_errors)

    memory_files = {
        "memory_matrices.png": fig1,
        "memory_probabilities.png": fig2,
        "free_energy_momentum.png": fig3,
        # q(i,k) on its own canvas: it is the panel that has to stay readable
        # when the simulation has many interfaces.
        "q_matrix.png": fig4,
    }
    for fname, mfig in memory_files.items():
        fpath = outdir / fname
        mfig.savefig(fpath, dpi=200)
        print(f"Memory analysis plot saved to {fpath}")

    # Memory index and conditional probabilities over the order parameter landscape
    fig_landscape, _ = plot_memory_landscape(interfaces, q_tot, q_errors=q_errors)
    landscape_file = outdir / "memory_landscape.png"
    fig_landscape.savefig(landscape_file, dpi=200)
    print(f"Memory landscape plot saved to {landscape_file}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()

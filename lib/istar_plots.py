"""
Analysis module for interface-based Markov State Models (iSTAR) in transition interface sampling.

This module provides functions to analyze transition paths, calculate memory effects, and
extract global crossing probabilities using the iSTAR approach. The module builds on
tools from repptis_analysis to process path ensembles.

iSTAR (interface-based State TrAnsition netwoRk (lol?)) is a specific approach to analyzing
transition interface sampling (TIS) simulations by constructing a Markov state model
at the interfaces. This allows for efficient calculation of transition probabilities
and rates between different states in complex molecular systems.
"""

from json import load
import numpy as np
from .reading import *
from .istar_analysis import *
import logging
from .repptis_analysis import *
from .repptis_msm import *
import matplotlib.pyplot as plt
import matplotlib as mpl
import deeptime as dpt

# Hard-coded rejection flags found in output files
ACCFLAGS, REJFLAGS = set_flags_ACC_REJ() 

# logger
logger = logging.getLogger(__name__)

def plot_rv_star(pes, interfaces, numberof):
    """
    Plot representative trajectories for the iSTAR model on a phase space diagram.
    
    This function visualizes selected trajectories from path ensembles, showing how
    paths traverse the phase space between interfaces. It's useful for understanding
    the typical behavior of transition paths in the system.
    
    Parameters
    ----------
    pes : list
        List of :py:class:`.PathEnsemble` objects containing trajectory information.
    interfaces : list
        List of interface positions to be displayed as vertical lines.
    numberof : int
        Number of trajectories to plot for each ensemble.
        
    Notes
    -----
    The function:
    1. Selects trajectories that go from the first to the last interface
    2. Draws vertical lines at each interface position
    3. Plots position vs. momentum for each selected trajectory
    4. Creates a legend identifying each ensemble
    
    This visualization is particularly valuable for:
    - Understanding the typical path behavior in phase space
    - Identifying differences in path mechanisms between ensembles
    - Verifying that paths appropriately cross interfaces
    """
    cycle_nrs = {}
    fig, ax = plt.subplots()

    for i, pe in enumerate(pes):
        accmask = get_flag_mask(pe, "ACC")
        loadmask = get_generation_mask(pe, "ld")
        start_cond = pe.lambmins <= interfaces[0]
        end_cond = pe.lambmaxs >= interfaces[-1]
        dir_mask = pe.dirs == 1

        cycle_nrs[i] = select_with_masks(pe.cyclenumbers, [start_cond, end_cond, dir_mask, accmask, ~loadmask])

        ax.vlines(interfaces, -4, 4, color='black')
        linecolor = None
        count = 0 
        while True and i > 0:
            id = np.random.choice(cycle_nrs[i])
            if np.all(pe.orders[id][:, 1] >= 0):
                lines = ax.plot(pe.orders[id][:, 0], pe.orders[id][:, 1], color=linecolor, label=i)
                linecolor = lines[0].get_color()
                count += 1
            if count == numberof:
                break
    fig.legend()
    fig.show()

def plot_rv_repptis(pes, interfaces, numberof):
    """
    Plot representative trajectories for the REPPTIS (Replica Exchange Path TIS) model.
    
    This function is similar to plot_rv_star() but specifically designed for REPPTIS
    simulations, selecting paths based on REPPTIS-specific criteria.
    
    Parameters
    ----------
    pes : list
        List of :py:class:`.PathEnsemble` objects containing trajectory information from REPPTIS simulations.
    interfaces : list
        List of interface positions to be displayed as vertical lines.
    numberof : int
        Number of trajectories to plot for each ensemble.
        
    Notes
    -----
    The function:
    1. Selects trajectories based on REPPTIS-specific criteria for interface crossing
    2. Draws vertical lines at each interface position
    3. Plots position vs. momentum for each selected trajectory
    4. Creates a legend identifying each ensemble
    
    This visualization helps in comparing path behavior between different REPPTIS ensembles
    and understanding the sampling efficiency of the REPPTIS approach.
    """
    cycle_nrs = {}
    fig, ax = plt.subplots()

    for i, pe in enumerate(pes):
        accmask = get_flag_mask(pe, "ACC")
        loadmask = get_generation_mask(pe, "ld")
        start_cond = pe.lambmins <= pe.interfaces[0][0]
        end_cond = pe.lambmaxs >= pe.interfaces[0][2]

        cycle_nrs[i] = select_with_masks(pe.cyclenumbers, [start_cond, end_cond, accmask, ~loadmask])

        ax.vlines(interfaces, -4, 4, color='black')
        linecolor = None
        count = 0
        while True and i > 0:
            id = np.random.choice(cycle_nrs[i])
            if np.all(pe.orders[id][:, 1] >= 0):
                lines = ax.plot(pe.orders[id][:, 0], pe.orders[id][:, 1], color=linecolor, label=i)
                linecolor = lines[0].get_color()
                count += 1
            if count == numberof:
                break
    fig.legend()
    fig.show()


# def plot_rv_comp(pes, interfaces, n_repptis, n_staple, pe_idxs=None):
    """
    Compare representative trajectories for REPPTIS and iSTAR models on a single phase space diagram.
    
    This function creates a visualization that directly compares path behavior between
    REPPTIS and iSTAR (staple) approaches, highlighting differences in sampling strategies
    and efficiency between these methods.
    
    Parameters
    ----------
    pes : list
        List of :py:class:`.PathEnsemble` objects containing trajectory information.
    interfaces : list
        List of interface positions to be displayed as vertical lines.
    n_repptis : int
        Number of REPPTIS trajectories to plot.
    n_staple : int
        Number of iSTAR (staple) trajectories to plot.
    pe_idxs : tuple, optional
        Tuple of (start, end) indices for path ensembles to compare. If None, compares all.
        This allows focusing the comparison on specific subsets of ensembles.
        
    Notes
    -----
    The function:
    1. Identifies paths matching REPPTIS and iSTAR criteria from the same ensembles
    2. Plots iSTAR paths as solid lines
    3. Plots REPPTIS paths with differentiated segments (middle segment highlighted)
    4. Uses consistent colors for paths from the same ensemble
    
    The visualization specifically highlights:
    - The full path for iSTAR trajectories
    - The key middle segment for REPPTIS trajectories (with peripheral segments dashed)
    - How each approach samples different aspects of the transition path space
    """
    if pe_idxs is None:
        pe_idxs = (0, len(pes)-1)
    assert pe_idxs[0] <= pe_idxs[1] if len(pe_idxs) > 1 else True, "Invalid pe_idxs: start index must be less than or equal to end index"
    cycle_nrs_strict = {}
    cycle_nrs_all = {}
    fig, ax = plt.subplots()
    ax.set_xlabel(r"Position x (=$\lambda$)")
    ax.set_ylabel("Momentum p")

    for i, pe in enumerate(pes):
        if i not in range(pe_idxs[0],pe_idxs[1]+1 if len(pe_idxs) > 1 else pe_idxs[0]+1):
            continue
        accmask = get_flag_mask(pe, "ACC")
        loadmask = get_generation_mask(pe, "ld")
        start_cond_repptis = pe.lambmins >= interfaces[0] if i > 2 else pe.lambmins <= interfaces[0]
        start_cond_repptis = pe.lambmins <= interfaces[0]
        end_cond_repptis = pe.lambmaxs < interfaces[-1] if i < len(interfaces)-1 else pe.lambmaxs >= interfaces[-1]
        start_cond_staple = pe.lambmins <= interfaces[0]
        start_cond_staple = pe.lambmins >= -9999
        end_cond_staple =  pe.lambmaxs >= interfaces[-1]
        dir_mask = pe.dirs == 1

        cycle_nrs_strict[i] = select_with_masks(pe.cyclenumbers, [pe.lmrs == "LMR", start_cond_repptis, end_cond_repptis, dir_mask, accmask, ~loadmask])
        cycle_nrs_all[i] = select_with_masks(pe.cyclenumbers, [pe.lmrs == "LMR", start_cond_staple, end_cond_staple, dir_mask, accmask, ~loadmask])

        ax.vlines(interfaces, -4, 4, color='black')
        linecolor = None
        count = 0 
        while count < 2*len(cycle_nrs_all[i]) and i > 0:
            if count == n_staple:
                break
            id = np.random.choice(cycle_nrs_all[i])
            if np.all(pe.orders[id][:, 1] >= 0):
                lines = ax.plot(pe.orders[id][:, 0], pe.orders[id][:, 1], ".-", color=linecolor, label=i)
                linecolor = lines[0].get_color()
                count += 1

        count = 0 
        it = 0
        while count < 2*len(cycle_nrs_strict[i]) and i > 0:
            if count == n_repptis:
                break
            id = np.random.choice(cycle_nrs_strict[i])
            l_piece = pe.orders[id][:pe.istar_idx[id][0]+1, :]
            m_piece = pe.orders[id][pe.istar_idx[id][0]-1:pe.istar_idx[id][1]+2, :]
            r_piece = pe.orders[id][pe.istar_idx[id][1]:, :]
            if len(m_piece) <= 100 and len(l_piece)+len(r_piece) <= 20550:
                ax.plot(m_piece[:, 0], m_piece[:, 1], "*-", color=linecolor, label=i)
                ax.plot(l_piece[:, 0], l_piece[:, 1], "--", alpha=0.5, color=linecolor)
                ax.plot(r_piece[:, 0], r_piece[:, 1], "--", alpha=0.5, color=linecolor)
                count += 1
            it += 1
            if it > 200000:
                id = np.random.choice(pe.cyclenumbers[pe.lmrs == "LMR"])
                l_piece = pe.orders[id][:pe.istar_idx[id][0]+1, :]
                m_piece = pe.orders[id][pe.istar_idx[id][0]-1:pe.istar_idx[id][1]+2, :]
                r_piece = pe.orders[id][pe.istar_idx[id][1]:, :]
                if np.all(m_piece[:, 1] >= 0):
                    ax.plot(m_piece[:, 0], m_piece[:, 1], color=linecolor, label=i)
                    ax.plot(l_piece[:, 0], l_piece[:, 1], "--", alpha=0.5, color=linecolor)
                    ax.plot(r_piece[:, 0], r_piece[:, 1], "--", alpha=0.5, color=linecolor)
                    count += 1
    fig.legend()
    fig.show()
    

def plot_rv_comp(pes, interfaces, n_repptis, n_staple, pe_idxs=None):
    """
    Compare representative trajectories for REPPTIS and iSTAR models on a single phase space diagram.
    ... [docstring omitted for brevity] ...
    """
    # ---------------------------------------------------------
    # STYLING CONFIGURATION
    # ---------------------------------------------------------
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 9,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.linewidth": 1.,
    })
    # ---------------------------------------------------------

    if pe_idxs is None:
        pe_idxs = (0, len(pes)-1)
        
    assert pe_idxs[0] <= pe_idxs[1] if len(pe_idxs) > 1 else True, "Invalid pe_idxs: start index must be less than or equal to end index"
    
    cycle_nrs_strict = {}
    cycle_nrs_all = {}
    
    # Create figure with better dimensions
    fig, ax = plt.subplots(figsize=(5, 4), dpi=120)
    
    # Use LaTeX for axis labels
    ax.set_xlabel(r"Position $x$ ($\lambda$)")
    ax.set_ylabel(r"Momentum $p$")
    ax.set_title(r"LMR Momentum comparison (REPPTIS vs StapleTIS)", pad=15)

    # Clean up axes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, linestyle='-', alpha=0.4, zorder=0)

    # Plot interfaces (styled nicely)
    ax.vlines(interfaces, ymin=-4, ymax=4, color='black', alpha=0.6, zorder=1)

    for i, pe in enumerate(pes):
        if i not in range(pe_idxs[0], pe_idxs[1]+1 if len(pe_idxs) > 1 else pe_idxs[0]+1):
            continue
            
        accmask = get_flag_mask(pe, "ACC")
        loadmask = get_generation_mask(pe, "ld")
        
        # Keeping your original overriding logic intact
        start_cond_repptis = pe.lambmins >= interfaces[0]
        end_cond_repptis = pe.lambmaxs < interfaces[-1] if i < len(interfaces)-1 else pe.lambmaxs >= interfaces[-1]
        start_cond_staple = pe.lambmins <= interfaces[0]
        end_cond_staple =  pe.lambmaxs >= interfaces[-1]
        dir_mask = pe.dirs == 1

        cycle_nrs_strict[i] = select_with_masks(pe.cyclenumbers, [pe.lmrs == "LMR", start_cond_repptis, end_cond_repptis, dir_mask, accmask, ~loadmask])
        cycle_nrs_all[i] = select_with_masks(pe.cyclenumbers, [pe.lmrs == "LMR", start_cond_staple, end_cond_staple, dir_mask, accmask, ~loadmask])

        # Use matplotlib's standard C-cycle for colors so they match within the same ensemble
        base_color = f"C{abs(i+17 % 10)}" 
        
        count = 0 
        while count < 2*len(cycle_nrs_all[i]) and i > 0:
            if count == n_staple:
                break
            id = np.random.choice(cycle_nrs_all[i])
            if np.all(pe.orders[id][:, 1] >= 2) or np.any(pe.orders[id][:, 1] > 6):
                l_piece = pe.orders[id][:pe.istar_idx[id][0]+1, :]
                m_piece = pe.orders[id][pe.istar_idx[id][0]-1:pe.istar_idx[id][1]+2, :]
                r_piece = pe.orders[id][pe.istar_idx[id][1]:, :]
                ax.plot(m_piece[:, 0], m_piece[:, 1], ".-", color=base_color, zorder=3, label=f"Ensemble {i}")
                ax.plot(l_piece[:, 0], l_piece[:, 1], ".-", alpha=0.4, color=base_color, zorder=2)
                ax.plot(r_piece[:, 0], r_piece[:, 1], ".-", alpha=0.4, color=base_color, zorder=2)
                # ax.plot(pe.orders[id][:, 0], pe.orders[id][:, 1], ".-", 
                #         color=base_color, alpha=0.8, zorder=2, label=f"Ensemble {i}")
                count += 1

        count = 0 
        it = 0
        while count < 2*len(cycle_nrs_strict[i]) and i > 0:
            if count == n_repptis:
                break
            id = np.random.choice(cycle_nrs_strict[i])
            l_piece = pe.orders[id][:pe.istar_idx[id][0]+1, :]
            m_piece = pe.orders[id][pe.istar_idx[id][0]-1:pe.istar_idx[id][1]+2, :]
            r_piece = pe.orders[id][pe.istar_idx[id][1]:, :]
            
            if len(m_piece) <= 100 and len(l_piece)+len(r_piece) <= 20550:
                ax.plot(m_piece[:, 0], m_piece[:, 1], "*-", color=base_color, zorder=3, label=f"Ensemble {i}")
                ax.plot(l_piece[:, 0], l_piece[:, 1], "--", alpha=0.4, color=base_color, zorder=2)
                ax.plot(r_piece[:, 0], r_piece[:, 1], "--", alpha=0.4, color=base_color, zorder=2)
                count += 1
            it += 1
            
            if it > 200000:
                id = np.random.choice(pe.cyclenumbers[pe.lmrs == "LMR"])
                l_piece = pe.orders[id][:pe.istar_idx[id][0]+1, :]
                m_piece = pe.orders[id][pe.istar_idx[id][0]-1:pe.istar_idx[id][1]+2, :]
                r_piece = pe.orders[id][pe.istar_idx[id][1]:, :]
                
                if np.all(m_piece[:, 1] >= 0):
                    ax.plot(m_piece[:, 0], m_piece[:, 1], "-", color=base_color, zorder=3, label=f"Ensemble {i}")
                    ax.plot(l_piece[:, 0], l_piece[:, 1], "--", alpha=0.4, color=base_color, zorder=2)
                    ax.plot(r_piece[:, 0], r_piece[:, 1], "--", alpha=0.4, color=base_color, zorder=2)
                    count += 1

    # Deduplicate legend labels (since we add labels inside the loops)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), frameon=True, loc="upper right")

    plt.tight_layout()
    plt.show()
    
    # Reset rcParams
    plt.rcdefaults()

def plot_hist_rv(pe, interfaces):
    """
    Plot histograms of position and momentum for selected trajectories in the iSTAR model.
    Plot histograms of position and momentum for selected trajectories in the iSTAR model.
    
    This function provides a statistical overview of the distribution of positions and
    momenta for trajectories that cross specific interfaces, helping to identify common
    features and differences in path ensembles.
    
    Parameters
    ----------
    pe : :py:class:`.PathEnsemble`
        A :py:class:`.PathEnsemble` object containing trajectory information.
    interfaces : list
        List of interface positions to be displayed as vertical lines.
    pe_idxs : tuple, optional
        Tuple of (start, end) indices for path ensembles to analyze. If None, analyzes all.
        
    Notes
    -----
    The function:
    1. Selects trajectories based on crossing criteria for specified interfaces
    2. Plots histograms of position and momentum for the selected trajectories
    3. Uses consistent colors for different ensembles
    
    This visualization helps in understanding the typical ranges of positions and momenta
    for transition paths, and how these distributions may differ between ensembles or
    across interfaces.
    """
    
    # ---------------------------------------------------------
    # STYLING CONFIGURATION (LaTeX fonts and clean aesthetics)
    # ---------------------------------------------------------
    mpl.rcParams.update({
        "text.usetex": True,                # Use LaTeX to write all text
        "font.family": "serif",             # Use serif fonts
        "font.serif": ["Computer Modern Roman"],  # Standard LaTeX font
        "axes.labelsize": 12,               # Larger axis labels
        "axes.titlesize": 13,               # Larger titles
        "legend.fontsize": 9,              # Legend font size
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.linewidth": 1,              # Thicker axes lines
    })
    
    # Define a clean color palette
    color_strict = "#1f77b4"  # Muted Blue
    color_all = "#ff7f0e"     # Safety Orange

    # ---------------------------------------------------------

    # Create figure with a slightly better aspect ratio and higher DPI
    fig, ax = plt.subplots(2, 2, figsize=(10, 10), dpi=120)
    
    # Format axes labels using LaTeX math mode
    for i in range(2):
        ax[1, i].set_xlabel(r"Momentum $p$")
        ax[i, 0].set_ylabel(r"Density")
        for j in range(2):
        # Despine top and right for a cleaner, modern look
            ax[i, j].spines['top'].set_visible(False)
            ax[i, j].spines['right'].set_visible(False)
        
            # Add a subtle horizontal grid
            ax[i, j].grid(axis='y', linestyle='--', alpha=0.6, zorder=0)

    # Use nicely formatted LaTeX strings for titles
    ax[0, 0].set_title(r"Average momentum ([$%s^*$])" % (int(pe.name[-1]) - 1), pad=15)
    ax[0, 1].set_title(r"Momentum at R ($\lambda_3$) ([$%s^*$])" % (int(pe.name[-1]) - 1), pad=15)
    ax[1, 0].set_title(r"Momentum at L ($\lambda_1$) ([$%s^*$])" % (int(pe.name[-1]) - 1), pad=15)
    ax[1, 1].set_title(r"Momentum at M ($\lambda_2$) ([$%s^*$])" % (int(pe.name[-1]) - 1), pad=15)

    accmask = get_flag_mask(pe, "ACC")
    loadmask = get_generation_mask(pe, "ld")
    
    # start_cond_strict = np.full_like(pe.lambmins, True)
    start_cond_strict = pe.lambmins <= interfaces[0]
    end_cond_strict = pe.lambmaxs >= interfaces[-1]
    
    start_cond_all = np.full_like(pe.lambmins, True)
    end_cond_all =  np.full_like(pe.lambmaxs, True)
    dir_mask = pe.dirs == 1

    cycle_nrs_strict = select_with_masks(pe.cyclenumbers, [pe.lmrs == "LMR", start_cond_strict, end_cond_strict, dir_mask, accmask, ~loadmask])
    cycle_nrs_all = select_with_masks(pe.cyclenumbers, [np.logical_or(pe.lmrs == "LMR", pe.lmrs == "LMR"), start_cond_all, end_cond_all, dir_mask, accmask, ~loadmask])
    
    avg_momenta_strict = []
    momentum_at_r_strict = []
    momentum_at_l_strict = []
    momentum_at_m_strict = []
    
    avg_momenta_all = []
    momentum_at_r_all = []
    momentum_at_l_all = []
    momentum_at_m_all = []
    
    for id in cycle_nrs_strict:
        l_piece = pe.orders[id][:pe.istar_idx[id][0]+1, :]
        m_piece = pe.orders[id][pe.istar_idx[id][0]-1:pe.istar_idx[id][1]+2, :]
        r_piece = pe.orders[id][pe.istar_idx[id][1]:, :]
        
        avg_momentum_m = np.mean(m_piece[:, 1])
        momentum_at_r = r_piece[0, 1] if len(r_piece) > 0 else None
        momentum_at_l = l_piece[-1, 1] if len(l_piece) > 0 else None
        momentum_at_m = m_piece[(m_piece[:, 0] - pe.interfaces[0][2]).argmin(), 1]
        
        avg_momenta_strict.append(avg_momentum_m)
        momentum_at_r_strict.append(momentum_at_r)
        momentum_at_l_strict.append(momentum_at_l)
        momentum_at_m_strict.append(momentum_at_m)
    
    for id in cycle_nrs_all:
        l_piece = pe.orders[id][:pe.istar_idx[id][0]+1, :]
        m_piece = pe.orders[id][pe.istar_idx[id][0]-1:pe.istar_idx[id][1]+2, :]
        r_piece = pe.orders[id][pe.istar_idx[id][1]:, :]
        
        avg_momentum_m = np.mean(m_piece[:, 1])
        momentum_at_r = r_piece[0, 1] if len(r_piece) > 0 else None
        momentum_at_l = l_piece[-1, 1] if len(l_piece) > 0 else None
        momentum_at_m = m_piece[(m_piece[:, 0] - pe.interfaces[0][2]).argmin(), 1]
        avg_momenta_all.append(avg_momentum_m)
        momentum_at_r_all.append(momentum_at_r)
        momentum_at_l_all.append(momentum_at_l)
        momentum_at_m_all.append(momentum_at_m)

    # Keyword arguments for histogram styling to keep it DRY
    hist_kwargs = {'bins': 40, 'alpha': 0.5, 'density': True, 'edgecolor': None, 'linewidth': 1.0, 'zorder': 3}
    
    # Plot histograms
    ax[0, 0].hist(avg_momenta_all, color=color_all, label=r'All', weights=pe.weights[cycle_nrs_all], **hist_kwargs)
    ax[0, 0].hist(avg_momenta_strict, color=color_strict, label=r'Reactive', weights=pe.weights[cycle_nrs_strict], **hist_kwargs)
    
    ax[0, 1].hist(momentum_at_r_all, color=color_all, label=r'All', weights=pe.weights[cycle_nrs_all], **hist_kwargs)
    ax[0, 1].hist(momentum_at_r_strict, color=color_strict, label=r'Reactive', weights=pe.weights[cycle_nrs_strict], **hist_kwargs)
    
    ax[1, 0].hist(momentum_at_l_all, color=color_all, label=r'All', weights=pe.weights[cycle_nrs_all], **hist_kwargs)
    ax[1, 0].hist(momentum_at_l_strict, color=color_strict, label=r'Reactive', weights=pe.weights[cycle_nrs_strict], **hist_kwargs)
    
    ax[1, 1].hist(momentum_at_m_all, color=color_all, label=r'All', weights=pe.weights[cycle_nrs_all], **hist_kwargs)
    ax[1, 1].hist(momentum_at_m_strict, color=color_strict, label=r'Reactive', weights=pe.weights[cycle_nrs_strict], **hist_kwargs)

    # Clean up legends
    for i in range(2):
        for j in range(2):
            ax[i, j].legend(frameon=False, loc='upper right')
            ax[i, j].set_ylim(0, 0.8) 

    # Ensure layout fits well without overlapping
    plt.tight_layout()
    plt.show()
    
    # Reset rcParams back to default to avoid affecting other plots downstream
    plt.rcdefaults()
    
    return avg_momenta_strict, momentum_at_r_strict, momentum_at_l_strict, avg_momenta_all, momentum_at_r_all, momentum_at_l_all, momentum_at_m_strict, momentum_at_m_all

def plot_hist_rv_overlap(pe1, pe2, interfaces):
    """
    Plot histograms of position and momentum for selected trajectories in the iSTAR model.
    Plot histograms of position and momentum for selected trajectories in the iSTAR model.
    
    This function provides a statistical overview of the distribution of positions and
    momenta for trajectories that cross specific interfaces, helping to identify common
    features and differences in path ensembles.
    
    Parameters
    ----------
    pe : :py:class:`.PathEnsemble`
        A :py:class:`.PathEnsemble` object containing trajectory information.
    interfaces : list
        List of interface positions to be displayed as vertical lines.
    pe_idxs : tuple, optional
        Tuple of (start, end) indices for path ensembles to analyze. If None, analyzes all.
        
    Notes
    -----
    The function:
    1. Selects trajectories based on crossing criteria for specified interfaces
    2. Plots histograms of position and momentum for the selected trajectories
    3. Uses consistent colors for different ensembles
    
    This visualization helps in understanding the typical ranges of positions and momenta
    for transition paths, and how these distributions may differ between ensembles or
    across interfaces.
    """
    
    # ---------------------------------------------------------
    # STYLING CONFIGURATION (LaTeX fonts and clean aesthetics)
    # ---------------------------------------------------------
    mpl.rcParams.update({
        "text.usetex": True,                # Use LaTeX to write all text
        "font.family": "serif",             # Use serif fonts
        "font.serif": ["Computer Modern Roman"],  # Standard LaTeX font
        "axes.labelsize": 12,               # Larger axis labels
        "axes.titlesize": 13,               # Larger titles
        "legend.fontsize": 9,              # Legend font size
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.linewidth": 1,              # Thicker axes lines
    })
    
    # Define a clean color palette
    color_strict = "#1f77b4"  # Muted Blue
    color_all = "#ff7f0e"     # Safety Orange

    # ---------------------------------------------------------

    # Create figure with a slightly better aspect ratio and higher DPI
    fig, ax = plt.subplots(2, 2, figsize=(10, 10), dpi=120)
    
    # pe1.istar_idx = [(0, len(pe1.orders[id]) - 2) for id in pe1.cyclenumbers]
    # pe2.istar_idx = [(1, len(pe2.orders[id]) - 2) for id in pe2.cyclenumbers]
    
    # Format axes labels using LaTeX math mode
    for i in range(2):
        ax[1, i].set_xlabel(r"Momentum $p$")
        ax[i, 0].set_ylabel(r"Density")
        for j in range(2):
        # Despine top and right for a cleaner, modern look
            ax[i, j].spines['top'].set_visible(False)
            ax[i, j].spines['right'].set_visible(False)
        
            # Add a subtle horizontal grid
            ax[i, j].grid(axis='y', linestyle='--', alpha=0.6, zorder=0)

    # Use nicely formatted LaTeX strings for titles
    ax[0, 0].set_title(r"Average momentum (forward)", pad=15)
    ax[0, 1].set_title(r"Momentum at R/M = $\lambda_%d$ (forward)" % (int(pe1.name[-1])), pad=15)
    ax[1, 0].set_title(r"Average momentum (backward) ", pad=15)
    ax[1, 1].set_title(r"Momentum at M/L = $\lambda_%d$ (backward) " % (int(pe1.name[-1]) - 1), pad=15)

    accmask1 = get_flag_mask(pe1, "ACC")
    loadmask1 = get_generation_mask(pe1, "ld")
    accmask2 = get_flag_mask(pe2, "ACC")
    loadmask2 = get_generation_mask(pe2, "ld")
    
    start_cond1 = np.full_like(pe1.lambmins, True)
    # start_cond_strict = pe.lambmins <= interfaces[0]
    # end_cond1 = pe.lambmaxs >= interfaces[-1]
    end_cond1 = np.full_like(pe1.lambmaxs, True)
    
    start_cond2 = np.full_like(pe2.lambmins, True)
    end_cond2 = np.full_like(pe2.lambmaxs, True)

    cycle_nrs1 = select_with_masks(pe1.cyclenumbers, [np.logical_or(pe1.lmrs == "LMR", pe1.lmrs == "LMR"), start_cond1, end_cond1, accmask1, ~loadmask1])
    cycle_nrs2 = select_with_masks(pe2.cyclenumbers, [np.logical_or(pe2.lmrs == "LMR", pe2.lmrs == "LML"), start_cond2, end_cond2, accmask2, ~loadmask2])
    cycle_nrs1b = select_with_masks(pe1.cyclenumbers, [np.logical_or(pe1.lmrs == "RML", pe1.lmrs == "RMR"), start_cond1, end_cond1, accmask1, ~loadmask1])
    cycle_nrs2b = select_with_masks(pe2.cyclenumbers, [np.logical_or(pe2.lmrs == "RML", pe2.lmrs == "LML"), start_cond2, end_cond2, accmask2, ~loadmask2])
    
    avg_momenta1 = []
    momentum_at_r1 = []
    momentum_at_l1 = []
    momentum_at_m1 = []
    
    avg_momenta2 = []
    momentum_at_r2 = []
    momentum_at_l2 = []
    momentum_at_m2 = []
    
    for id in cycle_nrs1:
        l_piece = pe1.orders[id][:pe1.istar_idx[id][0], :]
        m_piece = pe1.orders[id][pe1.istar_idx[id][0]:pe1.istar_idx[id][1]+1, :]
        r_piece = pe1.orders[id][pe1.istar_idx[id][1]+1:, :]
        
        avg_momentum_m = np.mean(m_piece[np.where(np.diff(np.sign(m_piece[:,0] - pe1.interfaces[0][1])))[0][-1]:, 1])
        momentum_at_r = m_piece[-1, 1] if len(m_piece) > 0 else None
        momentum_at_l = m_piece[0, 1] if len(m_piece) > 0 else None
        momentum_at_m = m_piece[np.where(np.diff(np.sign(m_piece[:,0] - pe1.interfaces[0][1])))[0][-1], 1]
        
        avg_momenta1.append(avg_momentum_m)
        momentum_at_r1.append(momentum_at_r)
        momentum_at_l1.append(momentum_at_l)
        momentum_at_m1.append(momentum_at_m)
    
    for id in cycle_nrs2:
        l_piece = pe2.orders[id][:pe2.istar_idx[id][0], :]
        m_piece = pe2.orders[id][max(0, pe2.istar_idx[id][0]-1):pe2.istar_idx[id][1]+1, :]
        r_piece = pe2.orders[id][pe2.istar_idx[id][1]+1:, :]
        
        avg_momentum_m = np.mean(m_piece[:np.where(np.diff(np.sign(m_piece[:,0] - pe2.interfaces[0][1])))[0][0] + 1, 1])
        momentum_at_r = m_piece[-1, 1] if len(m_piece) > 0 else None
        momentum_at_l = m_piece[0, 1] if len(m_piece) > 0 else None
        momentum_at_m = m_piece[np.where(np.diff(np.sign(m_piece[:,0] - pe2.interfaces[0][1])))[0][0], 1]
        avg_momenta2.append(avg_momentum_m)
        momentum_at_r2.append(momentum_at_r)
        momentum_at_l2.append(momentum_at_l)
        momentum_at_m2.append(momentum_at_m)
        
        
    avg_momenta1b = []
    momentum_at_r1b = []
    momentum_at_l1b = []
    momentum_at_m1b = []
    
    avg_momenta2b = []
    momentum_at_r2b = []
    momentum_at_l2b = []
    momentum_at_m2b = []
    
    for id in cycle_nrs1b:
        l_piece = pe1.orders[id][:pe1.istar_idx[id][0], :]
        m_piece = pe1.orders[id][pe1.istar_idx[id][0]:pe1.istar_idx[id][1]+1, :]
        r_piece = pe1.orders[id][pe1.istar_idx[id][1]+1:, :]
        
        avg_momentum_m = np.mean(m_piece[:np.where(np.diff(np.sign(m_piece[:,0] - pe1.interfaces[0][1])))[0][0] + 1, 1])
        momentum_at_r = m_piece[0, 1] if len(m_piece) > 0 else None
        momentum_at_l = m_piece[-1, 1] if len(m_piece) > 0 else None
        momentum_at_m = m_piece[np.where(np.diff(np.sign(m_piece[:,0] - pe1.interfaces[0][1])))[0][0], 1]
        
        avg_momenta1b.append(avg_momentum_m)
        momentum_at_r1b.append(momentum_at_r)
        momentum_at_l1b.append(momentum_at_l)
        momentum_at_m1b.append(momentum_at_m)
    
    for id in cycle_nrs2b:
        l_piece = pe2.orders[id][:pe2.istar_idx[id][0], :]
        m_piece = pe2.orders[id][pe2.istar_idx[id][0]:pe2.istar_idx[id][1]+1, :]
        r_piece = pe2.orders[id][pe2.istar_idx[id][1]+1:, :]
        
        avg_momentum_m = np.mean(m_piece[np.where(np.diff(np.sign(m_piece[:,0] - pe2.interfaces[0][1])))[0][-1]:, 1])
        momentum_at_r = m_piece[0, 1] if len(m_piece) > 0 else None
        momentum_at_l = m_piece[-1, 1] if len(m_piece) > 0 else None
        momentum_at_m = m_piece[np.where(np.diff(np.sign(m_piece[:,0] - pe2.interfaces[0][1])))[0][-1], 1]
        avg_momenta2b.append(avg_momentum_m)
        momentum_at_r2b.append(momentum_at_r)
        momentum_at_l2b.append(momentum_at_l)
        momentum_at_m2b.append(momentum_at_m)

    # Keyword arguments for histogram styling to keep it DRY
    hist_kwargs = {'bins': 40, 'alpha': 0.5, 'density': True, 'edgecolor': None, 'linewidth': 1.0, 'zorder': 3}
    
    # Plot histograms
    ax[0, 0].hist(avg_momenta1, color=color_all, label=r'M-R = $\lambda_{%s}-\lambda_{%s}$ $[%s^*]$' % (int(pe1.name[-1]) - 1, int(pe1.name[-1]), int(pe1.name[-1]) - 1), weights=pe1.weights[cycle_nrs1], **hist_kwargs)
    ax[0, 0].hist(avg_momenta2, color=color_strict, label=r'L-M = $\lambda_{%s}-\lambda_{%s}$ $[%s^*]$' % (int(pe2.name[-1]) - 2, int(pe2.name[-1]) - 1, int(pe2.name[-1]) - 1), weights=pe2.weights[cycle_nrs2], **hist_kwargs)
    
    ax[0, 1].hist(momentum_at_r1, color=color_all, label=r'R [$%s^*$]' % (int(pe1.name[-1]) - 1), weights=pe1.weights[cycle_nrs1], **hist_kwargs)
    ax[0, 1].hist(momentum_at_m2, color=color_strict, label=r'M [$%s^*$]' % (int(pe2.name[-1]) - 1), weights=pe2.weights[cycle_nrs2], **hist_kwargs)
    
    ax[1, 0].hist(avg_momenta1b, color=color_all, label=r'M-R = $\lambda_{%s}-\lambda_{%s}$ $[%s^*]$' % (int(pe1.name[-1]) - 1, int(pe1.name[-1]), int(pe1.name[-1]) - 1), weights=pe1.weights[cycle_nrs1b], **hist_kwargs)
    ax[1, 0].hist(avg_momenta2b, color=color_strict, label=r'L-M = $\lambda_{%s}-\lambda_{%s}$ $[%s^*]$' % (int(pe2.name[-1]) - 2, int(pe2.name[-1]) - 1, int(pe2.name[-1]) - 1), weights=pe2.weights[cycle_nrs2b], **hist_kwargs)
    
    ax[1, 1].hist(momentum_at_m1b, color=color_all, label=r'M [$%s^*$]' % (int(pe1.name[-1]) - 1), weights=pe1.weights[cycle_nrs1b], **hist_kwargs)
    ax[1, 1].hist(momentum_at_l2b, color=color_strict, label=r'L [$%s^*$]' % (int(pe2.name[-1]) - 1), weights=pe2.weights[cycle_nrs2b], **hist_kwargs)
    

    # Clean up legends
    for i in range(2):
        for j in range(2):
            ax[i, j].legend(frameon=False, loc='upper right')
            ax[i, j].set_ylim(0, 0.8) 

    # Ensure layout fits well without overlapping
    plt.tight_layout()
    plt.show()
    
    # Reset rcParams back to default to avoid affecting other plots downstream
    plt.rcdefaults()
    
    print("Percentage of reactive trajectories in PE1: %.2f%%" % (sum(pe1.weights[cycle_nrs1][pe1.lambmaxs[cycle_nrs1] >= interfaces[-1]]) / sum(pe1.weights[cycle_nrs1]) * 100))
    print("Percentage of reactive trajectories in PE2: %.2f%%" % (sum(pe2.weights[cycle_nrs2][pe2.lambmaxs[cycle_nrs2] >= interfaces[-1]]) / sum(pe2.weights[cycle_nrs2]) * 100))
    
    return avg_momenta2, momentum_at_r2, momentum_at_l2, avg_momenta1, momentum_at_r1, momentum_at_l1, momentum_at_m2, momentum_at_m1

def plot_memory_analysis(pes, q_tot, p, interfaces=None, q_errors=None):
    """
    Generate comprehensive visualizations for memory effect analysis in TIS simulations
    with support for non-equidistant interfaces and momentum effects.
    
    Parameters:
    -----------
    pes : list of PathEnsemble
        The path ensembles analyzed
    q_tot : numpy.ndarray
        A matrix with shape [2, n_interfaces, n_interfaces] where:
        - q_tot[0][i][k]: conditional crossing probabilities
        - q_tot[1][i][k]: sample counts for each calculation
    p : numpy.ndarray
        Transition probability matrix between interfaces
    interfaces : list, optional
        The interface positions for axis labeling. If None, uses sequential indices.
        
    Returns:
    -------
    tuple
        A tuple containing three matplotlib.figure.Figure objects:
        - fig1: Matrix heatmaps (memory effect matrix, ratio, asymmetry)
        - fig2: Forward/backward probability plots with memory retention bar charts
        - fig3: Free energy landscape, momentum effects, and flux network analysis
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    import matplotlib.gridspec as gridspec
    import matplotlib.colors as colors
    import seaborn as sns
    # Add legend
    from matplotlib.lines import Line2D
    
    # Extract the probability matrix and weights matrix from q_tot
    q_probs = q_tot[0]
    q_weights = q_tot[1]
    n_interfaces = q_probs.shape[0]
    
    if interfaces is None:
        interfaces = list(range(n_interfaces))
        is_equidistant = True
    else:
        # Check if interfaces are equidistant
        if len(interfaces) > 2:
            diffs = np.diff(interfaces)
            is_equidistant = np.allclose(diffs, diffs[0], rtol=0.05)
        else:
            is_equidistant = True
    
    # Generate more descriptive state labels
    state_labels = generate_state_labels(n_interfaces)

    M = construct_M_istar(p, 2*n_interfaces, n_interfaces)
    
    # Calculate diffusive reference probabilities based on interface spacing
    diff_ref = calculate_diffusive_reference(interfaces, q_tot[0], q_tot[1])
    plocs_repptis, plocs_istar = ploc_repptis_from_staples(pes, interfaces, n_int=n_interfaces)
    
    # Function to generate high-contrast colors for plots
    def generate_high_contrast_colors(n):
        if n <= 1:
            return ["#1f77b4"]  # Default blue for single item
        
        if n <= 10:
            # Viridis with enhanced spacing for better contrast
            viridis_cmap = plt.cm.get_cmap('viridis')
            return [colors.to_hex(viridis_cmap(i/(n-1) if n > 1 else 0.5)) for i in range(n)]
        else:
            # For more interfaces, use viridis with adjusted spacing
            cmap1 = plt.cm.get_cmap('viridis')
            
            # Get colors with deliberate spacing for better contrast
            colors_list = []
            for i in range(n):
                # Distribute colors with slight variations in spacing
                # This avoids adjacent indices having too similar colors
                pos = (i / max(1, n-1)) * 0.85 + 0.1  # Scale to range 0.1-0.95
                
                # Introduce small oscillations in color position for adjacent indices
                if i % 2 == 1:
                    pos = min(0.95, pos + 0.05)
                    
                colors_list.append(colors.to_hex(cmap1(pos)))
                
            return colors_list

    # ================ Figure 1: Matrix Heatmaps ================
    fig1 = plt.figure(figsize=(18, 7))
    gs1 = gridspec.GridSpec(1, 3, width_ratios=[1.2, 1, 1])
    
    # Create custom colormap for memory effect heatmap
    cmap_memory = LinearSegmentedColormap.from_list('memory_effect', 
                                                  [(0, 'blue'), (0.5, 'white'), (1, 'red')], N=256)
    
    # Plot 1.1: Memory Effect Matrix (q_probs)
    ax1 = fig1.add_subplot(gs1[0])
    
    # Calculate memory effect as deviation from diffusive reference
    memory_effect = np.zeros_like(q_probs)
    memory_effect.fill(np.nan)
    
    for i in range(n_interfaces):
        for j in range(n_interfaces):
            if not np.isnan(q_probs[i, j]) and not np.isnan(diff_ref[i, j]):
                memory_effect[i, j] = q_probs[i, j] - diff_ref[i, j]
    
    # Create diverging colormap centered at 0
    max_effect = np.nanmax(np.abs(memory_effect))
    
    masked_data = np.ma.masked_invalid(memory_effect)  # Mask NaN values
    im1 = ax1.imshow(masked_data, cmap=cmap_memory, vmin=-max_effect, vmax=max_effect, 
                    interpolation='none', aspect='auto')
    
    # Add colorbar
    cbar1 = fig1.colorbar(im1, ax=ax1, label='Memory Effect (q - q_diff)')
    
    # Add reference line at 0
    cbar1.ax.axhline(y=0.0, color='black', linestyle='--', linewidth=1)
    cbar1.ax.text(1.5, 0.0, '0 (diffusive)', va='center', ha='left', fontsize=9)
    
    # Add annotations with more compact formatting
    for i in range(n_interfaces):
        for j in range(n_interfaces):
            if not np.isnan(memory_effect[i, j]) and not np.ma.is_masked(masked_data[i, j]):
                weight = q_weights[i, j]
                # More compact format: actual/diff
                text = f"{q_probs[i, j]:.2f}/{diff_ref[i, j]:.2f}" if weight > 0 else "N/A"
                # Only show count if it's significant
                if weight > 10:
                    text += f"\n{int(weight)}"
                color = 'black' if abs(memory_effect[i, j]) < 0.3 else 'white'
                ax1.text(j, i, text, ha='center', va='center', color=color, fontsize=7)
    
    # Set ticks and labels using state labels
    ax1.set_xticks(np.arange(n_interfaces))
    ax1.set_yticks(np.arange(n_interfaces))
    ax1.set_xticklabels([f"{i}" for i in range(n_interfaces)])
    ax1.set_yticklabels([f"{i}" for i in range(n_interfaces)])
    ax1.set_xlabel('Target Turn at k')
    ax1.set_ylabel('Starting Turn at i')
    ax1.set_title('Memory Effect Matrix: q(i,k) - q_diffuse(i,k)', fontsize=12)
    
    # Plot 1.2: Memory Effect Ratio
    ax2 = fig1.add_subplot(gs1[1])
    
    # Calculate memory effect ratio: ratio of actual prob to diffusive prob
    memory_ratio = np.zeros_like(q_probs)
    memory_ratio.fill(np.nan)
    
    for i in range(n_interfaces):
        for j in range(n_interfaces):
            if (not np.isnan(q_probs[i, j]) and not np.isnan(diff_ref[i, j]) and
                diff_ref[i, j] > 0 and diff_ref[i, j] < 1):
                memory_ratio[i, j] = q_probs[i, j] / diff_ref[i, j]
    
    # Plot heatmap with logarithmic scale
    im2 = ax2.imshow(memory_ratio, cmap='RdBu_r', norm=colors.LogNorm(vmin=0.1, vmax=10))
    
    # Add colorbar
    cbar2 = fig1.colorbar(im2, ax=ax2, label='Probability Ratio q/q_diffuse [log scale]')
    
    # Add annotations for ratio values - more compact
    for i in range(n_interfaces):
        for j in range(n_interfaces):
            if not np.isnan(memory_ratio[i, j]) and q_weights[i, j] > 5:
                text_color = 'black'
                if memory_ratio[i, j] > 5 or memory_ratio[i, j] < 0.2:
                    text_color = 'white'
                ax2.text(j, i, f"{memory_ratio[i, j]:.1f}", ha='center', va='center', 
                       color=text_color, fontsize=7)
    
    ax2.set_xlabel('Target Turn at  k')
    ax2.set_ylabel('Starting Turn at i')
    ax2.set_title('Memory Effect Ratio: Deviation from Diffusive Behavior', fontsize=12)
    ax2.set_xticks(range(n_interfaces))
    ax2.set_yticks(range(n_interfaces))
    ax2.set_xticklabels([f"{i}" for i in range(n_interfaces)])
    ax2.set_yticklabels([f"{i}" for i in range(n_interfaces)])
    
    # Plot 1.3: Memory Asymmetry
    ax3 = fig1.add_subplot(gs1[2])
    
    # Calculate memory asymmetry for pairs of interfaces (i, j) using the p matrix
    memory_asymmetry = np.zeros_like(p)
    memory_asymmetry.fill(np.nan)
    
    for i in range(n_interfaces):
        for j in range(n_interfaces):
            if i != j:
                # Asymmetry is the difference between forward and backward transition probabilities
                memory_asymmetry[i, j] = p[i, j] - p[j, i]
    
    # Plot heatmap
    im3 = ax3.imshow(memory_asymmetry, cmap='RdBu', vmin=-0.5, vmax=0.5)
    
    # Add colorbar
    cbar3 = fig1.colorbar(im3, ax=ax3, label='Probability Asymmetry (i→j vs j→i)')
    
    # Add annotations - more compact
    for i in range(n_interfaces):
        for j in range(n_interfaces):
            if not np.isnan(memory_asymmetry[i, j]):
                text_color = 'black'
                if abs(memory_asymmetry[i, j]) > 0.3:
                    text_color = 'white'
                ax3.text(j, i, f"{memory_asymmetry[i, j]:.2f}", ha='center', va='center', 
                       color=text_color, fontsize=7)
    
    ax3.set_xlabel('Target Turn at j')
    ax3.set_ylabel('Starting Turn at i')
    ax3.set_title('Memory Asymmetry: Forward vs. Backward Transitions', fontsize=12)
    ax3.set_xticks(range(n_interfaces))
    ax3.set_yticks(range(n_interfaces))
    ax3.set_xticklabels([f"{i}" for i in range(n_interfaces)])
    ax3.set_yticklabels([f"{i}" for i in range(n_interfaces)])
    
    # Add explanatory text that includes info about non-equidistant interfaces
    if is_equidistant:
        desc_text = """
        Memory Effect Matrix: Shows deviations from diffusive behavior.
        In a purely diffusive process, all values would be 0.
        Values > 0 (red) indicate bias toward crossing, < 0 (blue) indicate bias toward returning.
        """
    else:
        desc_text = """
        Memory Effect Matrix: Shows deviations from diffusive behavior.
        Due to non-equidistant interfaces, the diffusive reference varies for each transition.
        Values > 0 (red) indicate bias toward crossing, < 0 (blue) indicate bias toward returning.
        """
    fig1.text(0.02, 0.02, desc_text, fontsize=10, wrap=True)
    
    plt.tight_layout(rect=[0, 0.07, 1, 0.95])
    fig1.suptitle('TIS Memory Effect Analysis - Matrix Representations' + 
                (' (Non-equidistant Interfaces)' if not is_equidistant else ''), fontsize=14)
    
    # ================ Figure 2: Forward/Backward Probs + Memory Retention ================
    fig2 = plt.figure(figsize=(18, 12))
    gs2 = gridspec.GridSpec(2, 2, height_ratios=[1, 0.8])
    
    # Create colors for targets
    forward_targets = [k for k in range(1, n_interfaces)]
    forward_colors = generate_high_contrast_colors(len(forward_targets))
    
    backward_targets = [k for k in range(n_interfaces-1)]
    backward_colors = generate_high_contrast_colors(len(backward_targets)) 
    
    # Plot 2.1: Forward Transition Probabilities (L→R)
    ax4 = fig2.add_subplot(gs2[0, 0])

    # For each target interface k, plot q(i,k) for all starting interfaces i<k
    for idx, k in enumerate(forward_targets):
        target_data = []
        starting_positions = []
        ref_probs = []
        valid_indices = []
        repptisp = []
        
        for i in range(k):
            # Include adjacent transitions only for interface 0->1
            if (i < k-1 or (i == 0 and k == 1)) and not np.isnan(q_probs[i, k]) and q_weights[i, k] > 5:
                target_data.append(q_probs[i, k])
                starting_positions.append(interfaces[i])
                valid_indices.append(i)
                ref_probs.append(diff_ref[i, k])
                repptisp.append(plocs_repptis[k]["LMR"])
        
        if target_data:
            # Plot actual probabilities with physical positions on x-axis
            ax4.plot(starting_positions, target_data, 'o-', 
                    label=(f'{k-1 if k>0 else k}→{k}'), linewidth=2, markersize=8,
                    color=forward_colors[idx])
            
            # Plot diffusive reference as dashed lines
            ax4.plot(starting_positions, ref_probs, '--',
                    color=forward_colors[idx], alpha=0.5)
            # ax4.plot(starting_positions, repptisp, ':',
            #         color=forward_colors[idx], alpha=0.5)
    
    # Configure the forward plot
    ax4.set_xlabel('Starting interface Position ($\lambda$$\\subset$)')
    ax4.set_ylabel('Probability q(i,k)')
    ax4.set_title('Forward Transition Probabilities (L→R)', fontsize=12)
    ax4.set_ylim(0, 1.05)
    sns.despine(ax=ax4)
    
    # Create better x-axis ticks using interface indices as labels but keeping physical distances
    ax4.set_xlim(min(interfaces) - 0.1, interfaces[n_interfaces-2] + 0.1)
    # Set the physical positions of interfaces on the x-axis
    ax4.set_xticks(interfaces)
    # Use state_labels for the tick labels
    ax4.set_xticklabels(["0→"]+[f"{i}$\\subset$" for i in range(1, n_interfaces-1)] + [f"{n_interfaces-1}"])
    
    # Add explanatory text about the dashed lines
    ref_text = """
    Dashed lines: Diffusive reference probabilities
    • Based on free energy differences between interfaces
    • Calculated using detailed balance principle
    """
    # ax4.text(0.02, 0.02, ref_text, transform=ax4.transAxes, fontsize=9, 
    #          bbox=dict(facecolor='white', alpha=0.8))
    
    # Add a legend with reasonable size
    ax4.legend(title='Target Region', loc='best', fontsize=9)
    
    # Plot 2.2: Backward Transition Probabilities (R→L)
    ax5 = fig2.add_subplot(gs2[0, 1])
    
    # For each target interface k, plot q(i,k) for all starting interfaces i>k
    for idx, k in enumerate(backward_targets):
        target_data = []
        starting_positions = []
        ref_probs = []
        valid_indices = []
        
        for i in range(k+1, n_interfaces):
            # Exclude adjacent transitions (i.e., exclude i=k+1)
            if i > k+1 and not np.isnan(q_probs[i, k]) and q_weights[i, k] > 5:
                target_data.append(q_probs[i, k])
                starting_positions.append(interfaces[i])
                valid_indices.append(i)
                ref_probs.append(diff_ref[i, k])
        
        if target_data:
            # Plot actual probabilities with physical po5)
            ax5.plot(starting_positions, target_data, 'o-', 
                    label=(f'{k}←{k+1}'), linewidth=2, markersize=8,
                    color=backward_colors[idx])
            
            # Plot diffusive reference as dashed lines12
            ax5.plot(starting_positions, ref_probs, '--', 
                    color=backward_colors[idx], alpha=0.5)
    
    # Configure the backward plot
    ax5.set_xlabel('Starting interface Position ($\lambda$)$\\supset$')
    ax5.set_ylabel('Probability q(i,k)')
    ax5.set_title('Backward Transition Probabilities (R→L)', fontsize=12)
    ax5.set_ylim(0, 1.05)
    sns.despine(ax=ax5)
        
    # Add explanatory text about the dashed lines
    # ax5.text(0.02, 0.02, ref_text, transform=ax5.tra9,
    #          bbox=dict(facecolor='whiteintalpha=0.8)ices as labels but keeping physical distances
    ax5.set_xlim(interfaces[1] - 0.1, max(interfaces) + 0.1)
    # Set the physical positions of interfaces on the x-axis
    ax5.set_xticks(interfaces)
    # Use state_labels for the tick labels
    ax5.set_xticklabels(["0←"]+[f"{i}$\\supset$" for i in range(1, n_interfaces-1)] + [f"{n_interfaces-1}"])
    
    # Add explanatory text about the dashed lines
    ax5.text(0.02, 0.02, ref_text, transform=ax5.transAxes, fontsize=9,
             bbox=dict(facecolor='white', alpha=0.8))
    
    # Add a legend with reasonable size
    ax5.legend(title='Target Region', loc='best', fontsize=9)
    
    # Plot 2.3: Forward Memory Retention
    ax6 = fig2.add_subplot(gs2[1, 0])

    # Calculate memory retention using simplified approach
    memory_index = calculate_memory_effect_index(q_probs, q_weights, q_errors=q_errors)

    # Prepare data for forward plot
    valid_k_fwd = [k for k in range(1, n_interfaces) if not np.isnan(memory_index['forward_variation'][k])]
    valid_variation_fwd = [memory_index['forward_variation'][k] for k in valid_k_fwd]
    valid_error_fwd = [memory_index['forward_variation_error'][k] if not np.isnan(memory_index['forward_variation_error'][k]) else 0 for k in valid_k_fwd]
    valid_positions_fwd = [interfaces[k] for k in valid_k_fwd]
    valid_colors_fwd = [forward_colors[k-1] for k in valid_k_fwd]
    valid_counts_fwd = [memory_index['forward_sample_sizes'][k] for k in valid_k_fwd]
    # Use state_labels for valid_k_fwd
    valid_state_labels_fwd = [(f'{k}$\\subset$' if k < n_interfaces-1 else f'{k}') for k in valid_k_fwd]

    # Calculate mean difference with diffusive reference for each target interface
    forward_mean_diff = np.zeros(n_interfaces)
    forward_mean_diff.fill(np.nan)
    
    for k in range(1, n_interfaces):
        diffs = []
        for i in range(k-1):  # Skip adjacent interface (i=k-1)
                # Only consider non-adjacent transitions with enough samples
                if not np.isnan(q_probs[i, k]) and q_weights[i, k] > 5:
                    diffs.append(abs(q_probs[i, k] - diff_ref[i, k]))
        
        if diffs:
                forward_mean_diff[k] = np.mean(diffs) * 100  # Convert to percentage
    
    # Extract valid mean differences to plot
    valid_mean_diff_fwd = [forward_mean_diff[k] if not np.isnan(forward_mean_diff[k]) else 0 for k in valid_k_fwd]

    if valid_k_fwd:
        # Create a twin axis for the memory retention plot
        ax6_twin = ax6.twinx()
        ax6_twin.set_ylim(0, 100)
        
        # Create bar plot for variation
        bars = ax6.bar(valid_positions_fwd, valid_variation_fwd, yerr=valid_error_fwd, color=valid_colors_fwd, alpha=0.7, 
                            width=np.mean(np.diff(interfaces))*0.7, capsize=5)  # Use average interface spacing for width
        
        # Add line plot for mean differences
        line = ax6_twin.plot(valid_positions_fwd, valid_mean_diff_fwd, 'o--', color='red', 
                                    linewidth=2, markersize=8, label=r'Mean |$\Delta$q|')
        
        # Add annotations showing variation and sample size
        for pos, var, count, label in zip(valid_positions_fwd, valid_variation_fwd, valid_counts_fwd, valid_state_labels_fwd):
                ax6.text(pos, var + 0.5, f"SD: {var:.1f}%\nn={count}", ha='center', fontsize=9)
        
        # Configure main plot
        ax6.set_xticks(interfaces)
        ax6.set_xticklabels([(f'{k-1 if k>0 else k}→{k}' if k < n_interfaces-1 else f'{k}') for k in range(n_interfaces)])
        ax6.set_xlabel('Target Region')
        ax6.set_ylabel('Memory Effect (Std. Dev. %)', color='C0')
        ax6.tick_params(axis='y', labelcolor='C0')
        ax6.set_title('Forward Memory Retention: Variation in Crossing Probabilities', fontsize=12)
        
        # Configure twin axis
        ax6_twin.set_ylabel(r'Mean |$\Delta$q| (%)', color='red')
        ax6_twin.tick_params(axis='y', labelcolor='red')
        
        # Set reasonable y-limits
        max_y_fwd = max(10.0, max(valid_variation_fwd) * 1.2) if valid_variation_fwd else 10.0
        ax6.set_ylim(0, max_y_fwd)
        
        max_y_twin_fwd = max(10.0, max(valid_mean_diff_fwd) * 1.2) if valid_mean_diff_fwd else 10.0
        ax6_twin.set_ylim(0, max_y_twin_fwd)
        
        # Set x limits based on the valid data points rather than all interfa10s7
        if len(valid_positions_fwd) > 0:
                padding = np.mean(np.diff(interfaces)) if len(interfaces) > 1 else 0.5
                ax6.set_xlim(min(valid_positions_fwd) - padding/2, max(valid_positions_fwd) + padding/2)
        
        # Create a combined legend
        custom_lines = [
                Line2D([0], [0], color='black', lw=0, marker='s', markersize=10, markerfacecolor='C0', alpha=0.7),
                Line2D([0], [0], color='red', lw=2, marker='o', markersize=6)
        ]
        ax6.legend(custom_lines, ['Std. Dev. (%)', r'Mean |$\Delta$q| (%)'], loc='upper left')
        
    else:
        ax6.text(0.5, 0.5, "Insufficient data for forward memory retention analysis", 
                ha='center', va='center', transform=ax6.transAxes)
        ax6.set_xticks(interfaces)
        ax6.set_xticklabels([f"{k}$\\subset$" for k in range(n_interfaces)])

    # Plot 2.4: Backward Memory Retention
    ax7 = fig2.add_subplot(gs2[1, 1])

    # Prepare data for backward plot
    valid_k_bwd = [k for k in range(n_interfaces-1) if not np.isnan(memory_index['backward_variation'][k])]
    valid_variation_bwd = [memory_index['backward_variation'][k] for k in valid_k_bwd]
    valid_error_bwd = [memory_index['backward_variation_error'][k] if not np.isnan(memory_index['backward_variation_error'][k]) else 0 for k in valid_k_bwd]
    valid_positions_bwd = [interfaces[k] for k in valid_k_bwd]
    valid_colors_bwd = [backward_colors[k] for k in valid_k_bwd]
    valid_counts_bwd = [memory_index['backward_sample_sizes'][k] for k in valid_k_bwd]
    # Use state_labels for valid_k_bwd
    valid_state_labels_bwd = [(f'{k}$\\subset$' if k > 0 else f'{k}') for k in valid_k_bwd]
    
    # Calculate mean difference with diffusive reference for each target interface
    backward_mean_diff = np.zeros(n_interfaces)
    backward_mean_diff.fill(np.nan)
    
    for k in range(n_interfaces-1):
        diffs = []
        for i in range(k+2, n_interfaces):  # Skip adjacent interface (i=k+1)
                # Only consider non-adjacent transitions with enough samples
                if not np.isnan(q_probs[i, k]) and q_weights[i, k] > 5:
                    diffs.append(abs(q_probs[i, k] - diff_ref[i, k]))
        
        if diffs:
                backward_mean_diff[k] = np.mean(diffs) * 100  # Convert to percentage
    
    # Extract valid mean differences to plot
    valid_mean_diff_bwd = [backward_mean_diff[k] if not np.isnan(backward_mean_diff[k]) else 0 for k in valid_k_bwd]

    if valid_k_bwd:
        # Create a twin axis for the memory retention plot
        ax7_twin = ax7.twinx()
        ax7_twin.set_ylim(0, 100)

        # Create bar plot using interface physical positions
        bars = ax7.bar(valid_positions_bwd, valid_variation_bwd, yerr=valid_error_bwd, color=valid_colors_bwd, alpha=0.7,
                            width=np.mean(np.diff(interfaces))*0.7, capsize=5)  # Use average interface spacing for width
        
        # Add line plot for mean differences
        line = ax7_twin.plot(valid_positions_bwd, valid_mean_diff_bwd, 'o--', color='red', 
                                    linewidth=2, markersize=8, label=r'Mean |$\Delta$q|')
        
        # Add annotations showing variation and sample size
        for pos, var, count, label in zip(valid_positions_bwd, valid_variation_bwd, valid_counts_bwd, valid_state_labels_bwd):
                ax7.text(pos, var + 0.5, f"SD: {var:.1f}%\nn={count}", ha='center', fontsize=9)
        
        # Configure plot
        ax7.set_xticks(interfaces)
        ax7.set_xticklabels([f'{k}←{k+1}' for k in range(n_interfaces)])
        ax7.set_xlabel('Target Region')
        ax7.set_ylabel('Memory Effect (Std. Dev. %)', color='C0')
        ax7.tick_params(axis='y', labelcolor='C0')
        ax7.set_title('Backward Memory Retention: Variation in Crossing Probabilities', fontsize=12)
        
        # Configure twin axis
        ax7_twin.set_ylabel(r'Mean |$\Delta$q| (%)', color='red')
        ax7_twin.tick_params(axis='y', labelcolor='red')
        
        # Set reasonable y-limits
        max_y_bwd = max(10.0, max(valid_variation_bwd) * 1.2) if valid_variation_bwd else 10.0
        ax7.set_ylim(0, max_y_bwd)
        
        max_y_twin_bwd = max(10.0, max(valid_mean_diff_bwd) * 1.2) if valid_mean_diff_bwd else 10.0
        ax7_twin.set_ylim(0, max_y_twin_bwd)
        
        # Set x limits based on the valid data points rather than all interfaces
        if len(valid_positions_bwd) > 0:
                padding = np.mean(np.diff(interfaces)) if len(interfaces) > 1 else 0.5
                ax7.set_xlim(min(valid_positions_bwd) - padding/2, max(valid_positions_bwd) + padding/2)
        
        # Create a combined legend
        custom_lines = [
                Line2D([0], [0], color='black', lw=0, marker='s', markersize=10, markerfacecolor='C0', alpha=0.7),
                Line2D([0], [0], color='red', lw=2, marker='o', markersize=6)
        ]
        ax7.legend(custom_lines, ['Std. Dev. (%)', r'Mean |$\Delta$q| (%)'], loc='upper left')
        
    else:
        ax7.text(0.5, 0.5, "Insufficient data for backward memory retention analysis", 
                ha='center', va='center', transform=ax7.transAxes)
        ax7.set_xticks(interfaces)
        ax7.set_xticklabels([f'{k}←{k+1}' for k in range(n_interfaces)])
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig2.suptitle('TIS Memory Effect Analysis - Transition Probabilities and Memory Retention', fontsize=14)

    # ================ Figure 3: Free Energy Landscape and Momentum Effects ================
    fig3 = plt.figure(figsize=(18, 14))
    gs3 = gridspec.GridSpec(2, 2, height_ratios=[1, 1])
    
    # Run momentum vs free energy analysis
    momentum_results = analyze_momentum_vs_free_energy(interfaces, q_tot[0], q_tot[1])
    
    # Plot 3.1: Enhanced Free Energy Profile - spans the entire top row
    ax8 = fig3.add_subplot(gs3[0, :])
    
    # Extract data from momentum analysis
    delta_G = momentum_results['free_energy_differences']
    
    # Calculate cumulative free energy profile
    cumulative_G = np.zeros(len(interfaces))
    for i in range(1, len(interfaces)):
        # Add up all the delta_G values from the first interface
        valid_path = True
        for j in range(i):
            if np.isnan(delta_G[j, j+1]):
                valid_path = False
                break
            cumulative_G[i] += delta_G[j, j+1]
        
        if not valid_path:
            cumulative_G[i] = np.nan
    
    # Plot free energy profile with improved styling
    ax8.plot(interfaces, cumulative_G, 'o-', linewidth=2.5, color='royalblue', 
             label='Free Energy Profile')
    
    # Add marker points with annotations
    for i, (pos, g) in enumerate(zip(interfaces, cumulative_G)):
        if not np.isnan(g):
            ax8.plot(pos, g, 'o', markersize=8, color='royalblue')
            ax8.text(pos, g + 0.15, f"{g:.2f}", ha='center', va='bottom', fontsize=10, 
                    bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))
    
    # Enhance the appearance
    ax8.set_xlabel(r'Interface Position ($\lambda$)', fontsize=12)
    ax8.set_ylabel(r'Free Energy G($\lambda$) (kT)', fontsize=12)
    ax8.set_title('Free Energy Profile Along Interface Coordinate', fontsize=14)
    ax8.grid(True, alpha=0.3, linestyle='--')
    
    # Add shaded area under the curve for visual appeal
    ax8.fill_between(interfaces, 0, cumulative_G, alpha=0.2, color='royalblue')
    
    ax8.legend(loc='best', fontsize=10)
    
    # Plot 3.2: Observed vs Diffusive Probabilities Comparison (bottom-left)
    ax9 = fig3.add_subplot(gs3[1, 0])
    
    # Plot data from momentum results
    diffusive_q = momentum_results['diffusive_probabilities']
    momentum_effects = momentum_results['momentum_effects']
    momentum_significance = momentum_results['momentum_significance']
    
    # Collect valid data points, excluding transitions to/from boundary interfaces
    valid_points = []
    for i in range(len(interfaces)):
        for j in range(len(interfaces)):
            # Skip diagonal and transitions involving boundary interfaces (0 or n-1)
            if (abs(i-j) >= 2 and i > 0 and i < len(interfaces)-1 and 
                j > 0 and j < len(interfaces)-1 and 
                not np.isnan(diffusive_q[i, j]) and 
                not np.isnan(momentum_effects[i, j])):
                
                if q_weights is None or q_weights[i, j] >= 5:  # Min samples
                    observed = diffusive_q[i, j] * (1 + momentum_effects[i, j])  # q_matrix value
                    significant = momentum_significance[i, j]
                    valid_points.append((diffusive_q[i, j], observed, significant, f"{i}→{j}"))
    
    if valid_points:
        # Unpack the valid points
        x_vals, y_vals, significance, labels = zip(*valid_points)
        
        # Calculate plot limits
        max_val = max(max(x_vals), max(y_vals)) * 1.1
        min_val = min(min(x_vals), min(y_vals)) * 0.9
        
        # Plot the ideal 1:1 line
        ax9.plot([min_val, max_val], [min_val, max_val], '--', color='gray', alpha=0.7)
        
        # Plot each point, colored by significance
        for x, y, sig, label in zip(x_vals, y_vals, significance, labels):
            color = 'red' if sig else 'blue'
            ax9.scatter(x, y, color=color, s=50, alpha=0.7)
        
        # Add labels with improved readability
        for i, (x, y, sig, label) in enumerate(zip(x_vals, y_vals, significance, labels)):
            # Calculate offset direction based on point position to avoid overlaps
            dx = 10 if x < 0.5 * (min_val + max_val) else -30
            dy = 10 if y < 0.5 * (min_val + max_val) else -15
            
            # Create a small white background for the text to improve readability
            text = ax9.annotate(
                label, 
                (x, y), 
                xytext=(dx, dy),
                textcoords='offset points', 
                fontsize=9,
                fontweight='bold',
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    fc="white",
                    ec="gray",
                    alpha=0.8
                )
            )
        
        ax9.set_xlabel('Diffusive Probability (Free Energy Model)', fontsize=12)
        ax9.set_ylabel('Observed Probability', fontsize=12)
        ax9.set_title('Observed vs Diffusive Transition Probabilities\n(excluding boundary interfaces)', fontsize=14)
        ax9.set_xlim(min_val, max_val)
        ax9.set_ylim(min_val, max_val)
        ax9.grid(True, alpha=0.3)
        
        # Add legend using the same color scheme as momentum_vs_fe
        ax9.scatter([], [], color='blue', label='Free Energy Dominated')
        ax9.scatter([], [], color='red', label='Momentum Effects')
        ax9.legend(fontsize=10)
    else:
        ax9.text(0.5, 0.5, "Insufficient valid data for comparison\n(non-boundary transitions)",
                ha='center', va='center', transform=ax9.transAxes)
    # Plot 3.3: Interface Pair Classification (bottom-right)
    ax10 = fig3.add_subplot(gs3[1, 1])
    
    # Extract classification from momentum results
    classification = momentum_results['classification']
    overall = momentum_results['overall_classification']
    
    # Define colors for classifications - matching those in analyze_momentum_vs_free_energy
    class_colors = {
        "free_energy_dominated": 'blue',
        "momentum_dominated": 'red',
        "strong_momentum": 'darkred',
    }
    
    # Create a modified list of colors, excluding first and last interfaces
    n_intervals = len(interfaces) - 1
    modified_colors = []
    for i in range(n_intervals):
        if i == 0 or i == n_intervals - 1:
            # Skip classification coloring for first and last intervals
            modified_colors.append('lightgray')
        else:
            modified_colors.append(class_colors.get(classification[i], 'gray'))
    
    # Plot interface pair classifications as colored bars
    x = np.arange(n_intervals)
    bars = ax10.bar(x, [1] * n_intervals, color=modified_colors, alpha=0.7, width=0.7)
    
    # Add labels for each interface pair, but skip first and last
    for i in range(n_intervals):
        if i > 0 and i < n_intervals - 1:
            # Determine text color based on background color brightness for better readability
            bg_color = modified_colors[i]
            
            # Function to determine if background color is dark (needs white text)
            def is_dark_color(color_name):
                dark_colors = ['darkred', 'red', 'darkblue', 'navy', 'black']
                return color_name.lower() in dark_colors
            
            # Choose text color based on background brightness
            text_color = 'white' if is_dark_color(bg_color) else 'black'
            
            # Add text with a small outline for better readability
            text = classification[i].replace('_', '\n')
            # First add text with outline
            for offset_x, offset_y in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                ax10.text(i + offset_x*0.01, 0.5 + offset_y*0.01, text,
                       ha='center', va='center', fontsize=10,
                       color='black' if text_color == 'white' else 'white',
                       alpha=0.5)
            # Then add main text
            ax10.text(i, 0.5, text, 
                   ha='center', va='center', fontsize=10, 
                   color=text_color, fontweight='bold')
    
    ax10.set_xticks(x)
    ax10.set_xticklabels([f"{i}→{i+1}" for i in range(n_intervals)])
    ax10.set_yticks([])  # No y-ticks needed
    ax10.set_xlabel('Interface Pair', fontsize=12)
    ax10.set_title('Interface Pair Classification (excluding boundary interfaces)', fontsize=14)
    
    # Create custom legend for the classification - matching analyze_momentum_vs_free_energy
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=class_colors.get('free_energy_dominated', 'blue'), label='Free Energy Dominated'),
        Patch(facecolor=class_colors.get('momentum_dominated', 'red'), label='Momentum Dominated'),
        Patch(facecolor=class_colors.get('strong_momentum', 'darkred'), label='Strong Momentum Effects'),
        Patch(facecolor='lightgray', label='Boundary (not classified)')
    ]
    ax10.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.15), 
               ncol=2, fontsize=10)
    
    # Add overall title and additional information
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    overall_class = momentum_results['overall_classification'].replace("_", " ").title()
    fig3.suptitle(f'TIS Memory Analysis - Free Energy and Momentum Effects ({overall_class})', fontsize=16)
    # Add metrics as text in the bottom of figure 3
    metrics_text = (
        f"Overall Classification: {overall_class}\n"
        f"Average Momentum Effect: {momentum_results['avg_momentum_effect']:.3f}\n"
        f"Average Free Energy: {momentum_results['avg_free_energy']:.3f} kT\n"
        f"Average Probabilities: {momentum_results['avg_probabilities']:.3f}"
    )
    fig3.text(0.02, 0.01, metrics_text, fontsize=10, wrap=True)
    
    return fig1, fig2, fig3

##################################
# Network Analysis Tools
##################################

def analyze_network_connectivity(M, source_state=0, sink_state=-1, max_paths=10):
    """
    Analyze the connectivity of a transition network efficiently without enumerating all paths.
    
    Parameters
    ----------
    M : np.ndarray
        Transition matrix representing the Markov state model
    source_state : int, optional
        Index of the source state (default: 0)
    sink_state : int, optional
        Index of the sink state (default: -1)
    max_paths : int, optional
        Maximum number of paths to sample for visualization (default: 10)
        
    Returns
    -------
    dict
        Dictionary containing connectivity analysis results
    """
    from scipy import sparse
    from scipy.sparse.csgraph import connected_components, shortest_path
    import networkx as nx

    n_states = M.shape[0]
    sink_idx = n_states - 1 if sink_state == -1 else sink_state
    
    # Generate state labels for the plot
    state_labels = generate_state_labels(n_states//2)
    
    # Create a graph representation where edges exist if M[i,j] > 0
    graph = (M > 0).astype(int)
    
    # Find strongly connected components (nodes with paths in both directions)
    n_strong, strong_labels = connected_components(
        sparse.csr_matrix(graph), directed=True, connection='strong'
    )
    
    # Find weakly connected components (nodes connected ignoring direction)
    n_weak, weak_labels = connected_components(
        sparse.csr_matrix(graph), directed=True, connection='weak'
    )
    
    # Check path existence and get predecessor information
    try:
        path_lengths, predecessors = shortest_path(
            sparse.csr_matrix(graph), directed=True, 
            indices=source_state, return_predecessors=True
        )
        direct_path_exists = np.isfinite(path_lengths[sink_idx])
    except:
        direct_path_exists = False
        predecessors = None
        path_lengths = None
    
    # Create NetworkX graph for visualization
    G = nx.DiGraph()
    
    # Add edges with transition probabilities as weights
    for i in range(n_states):
        for j in range(n_states):
            if M[i, j] > 0:
                G.add_edge(i, j, weight=M[i, j])
    
    # Find critical nodes using edge betweenness centrality
    # This identifies bottleneck edges without enumerating all paths
    if direct_path_exists:
        # Calculate edge betweenness centrality only for edges on paths between source and sink
        # This is much more efficient than calculating for the entire graph
        subgraph_nodes = set()
        
        # Use a different algorithm to find a sample of paths for visualization
        # First use Yen's algorithm to find k-shortest paths
        try:
            sample_paths = []
            for i, path in enumerate(nx.shortest_simple_paths(G, source_state, sink_idx, weight='weight')):
                if i >= max_paths:
                    break
                sample_paths.append(path)
                subgraph_nodes.update(path)
        except nx.NetworkXNoPath:
            sample_paths = []
        
        # If we couldn't get paths with the above method, reconstruct at least one path using predecessors
        if not sample_paths and predecessors is not None:
            path = [sink_idx]
            current = sink_idx
            while current != source_state:
                if current < 0 or predecessors[current] < 0:
                    # No path found
                    path = []
                    break
                current = predecessors[current]
                path.append(current)
            path.reverse()
            if path:
                sample_paths.append(path)
                subgraph_nodes.update(path)
        
        # Find critical nodes using a different approach
        if len(subgraph_nodes) > 2:  # If we have nodes besides source and sink
            # Create a subgraph containing only the nodes on the sampled paths
            subgraph = G.subgraph(subgraph_nodes).copy()
            
            # See if source and sink are still connected if we remove each node
            critical_nodes = set()
            for node in subgraph_nodes:
                # Skip source and sink
                if node == source_state or node == sink_idx:
                    continue
                    
                # Remove node and check connectivity
                temp_graph = subgraph.copy()
                temp_graph.remove_node(node)
                if not nx.has_path(temp_graph, source_state, sink_idx):
                    critical_nodes.add(node)
        else:
            critical_nodes = set()
    else:
        sample_paths = []
        critical_nodes = set()
    
    # Create visualization
    plt.figure(figsize=(10, 8))
    
    # Use a more deterministic layout if possible
    try:
        pos = nx.kamada_kawai_layout(G)
    except:
        pos = nx.spring_layout(G, seed=42)  # Use seed for reproducibility
    
    # Node colors based on strongly connected component
    node_colors = [strong_labels[i] for i in range(len(pos.values()))]
    
    # Draw the basic graph with node labels using state_labels
    nx.draw(G, pos, with_labels=True, node_color=node_colors, 
           labels={int(i): state_labels[i] for i in pos.keys()},
           cmap=plt.cm.tab10, node_size=500, alpha=0.8)
    
    # Highlight source and sink
    nx.draw_networkx_nodes(G, pos, nodelist=[source_state], 
                          node_color='green', node_size=700)
    nx.draw_networkx_nodes(G, pos, nodelist=[sink_idx], 
                          node_color='red', node_size=700)
    
    # Highlight critical nodes
    if critical_nodes:
        nx.draw_networkx_nodes(G, pos, nodelist=list(critical_nodes), 
                              node_color='yellow', node_size=600)
    
    # If a direct path exists, highlight one of the paths
    if sample_paths:
        shortest = sample_paths[0]  # Just use the first path
        path_edges = list(zip(shortest[:-1], shortest[1:]))
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, 
                              edge_color='red', width=2)
    
    plt.title(f"Network Analysis: {n_strong} Strong Components, {n_weak} Weak Components")
    
    # Create legend patches with state_labels for source and sink
    legend_patches = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', 
                 markersize=10, label=f'Source State: {state_labels[source_state]}'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                 markersize=10, label=f'Sink State: {state_labels[sink_idx]}')
    ]
    if critical_nodes:
        legend_patches.append(
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='yellow', 
                     markersize=10, label='Critical Bridge States')
        )
    plt.legend(handles=legend_patches, loc='upper right')
    
    return {
        'n_strong_components': n_strong,
        'strong_component_labels': strong_labels,
        'n_weak_components': n_weak,
        'weak_component_labels': weak_labels,
        'direct_path_exists': direct_path_exists,
        'critical_nodes': critical_nodes,
        'sample_paths': sample_paths,
        'network_graph': G
    }


###################################
# Older Memory Analysis Functions
###################################

def visualize_turn_based_analysis(analysis_results, interfaces, q_matrix, q_weights=None):
    """
    Create visualizations of the turn-based memory vs free energy analysis.
    
    Parameters
    ----------
    analysis_results : dict
        Output from analyze_memory_vs_free_energy_effects function
    interfaces : list or array
        The positions of the interfaces along the reaction coordinate
    q_matrix : numpy.ndarray
        Original matrix of conditional crossing probabilities
    q_weights : numpy.ndarray, optional
        Matrix of sample counts for each q_matrix value
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure containing the visualization
    """
    import seaborn as sns
    from matplotlib.colors import LinearSegmentedColormap
    
    n_interfaces = len(interfaces)
    
    # Extract results
    delta_G = analysis_results['free_energy_differences']
    predicted_q = analysis_results['predicted_q_matrix']
    memory_effects = analysis_results['memory_effects']
    memory_significance = analysis_results['memory_significance']
    turn_metrics = analysis_results['turn_based_metrics']
    
    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(16, 18))
    
    # Create a GridSpec layout to better control spacing
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # Add overall classification information at the top as a title
    overall = analysis_results['overall_classification'].replace("_", " ").title()
    forward_vs_backward = analysis_results['forward_vs_backward'].replace("_", " ").title()
    title_text = f'Turn-Based Memory vs Free Energy Analysis\nOverall: {overall} ({forward_vs_backward})'
    fig.text(0.5, 0.98, title_text, ha='center', va='top', fontsize=16, weight='bold')
    
    # Plot 1: Free Energy Profile (top-left)
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Calculate cumulative free energy profile
    cumulative_G = np.zeros(n_interfaces)
    for i in range(1, n_interfaces):
        # Add up all the delta_G values from the first interface
        valid_path = True
        for j in range(i):
            if np.isnan(delta_G[j, j+1]):
                valid_path = False
                break
            cumulative_G[i] += delta_G[j, j+1]
        
        if not valid_path:
            cumulative_G[i] = np.nan
    
    # Plot the free energy profile
    ax1.plot(interfaces, cumulative_G, 'o-', linewidth=2, color='blue')
    
    # Add marker points with annotations
    for i, (pos, g) in enumerate(zip(interfaces, cumulative_G)):
        if not np.isnan(g):
            ax1.plot(pos, g, 'o', markersize=8, color='blue')
            ax1.text(pos, g + 0.1, f"{g:.2f}", ha='center', va='bottom', fontsize=9)
    
    ax1.set_xlabel('Interface Position')
    ax1.set_ylabel('Free Energy G (kT)')
    ax1.set_title('Free Energy Profile from Turn-Based Transitions', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Memory Effect Heatmap (top-right)
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Create a diverging colormap for memory effects
    cmap_memory = LinearSegmentedColormap.from_list('memory_effect', 
                                                  [(0, 'blue'), (0.5, 'white'), (1, 'red')], N=256)
    
    # Create a masked array for NaN values
    masked_memory = np.ma.masked_invalid(memory_effects)
    
    # Determine color range symmetrically around zero
    max_effect = np.nanmax(np.abs(memory_effects))
    
    # Plot the heatmap
    im = ax2.imshow(masked_memory, cmap=cmap_memory, vmin=-max_effect, vmax=max_effect, 
                   interpolation='none', aspect='auto')
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax2, label='Memory Effect (q - q_predicted) / q_predicted')
    
    # Add annotations
    for i in range(n_interfaces):
        for j in range(n_interfaces):
            if not np.isnan(memory_effects[i, j]):
                sig_mark = '*' if memory_significance[i, j] else ''
                text = f"{memory_effects[i, j]:.2f}{sig_mark}"
                color = 'black' if abs(memory_effects[i, j]) < 0.5 else 'white'
                ax2.text(j, i, text, ha='center', va='center', color=color, fontsize=9)
    
    ax2.set_xticks(range(n_interfaces))
    ax2.set_yticks(range(n_interfaces))
    ax2.set_xticklabels([f"{i}" for i in range(n_interfaces)])
    ax2.set_yticklabels([f"{i}" for i in range(n_interfaces)])
    ax2.set_xlabel('Target Turn at Interface k')
    ax2.set_ylabel('Starting Turn at Interface i')
    ax2.set_title('Memory Effects: Deviation from Free Energy Model', fontsize=12)
    
    # Plot 3: Comparison of Observed vs Predicted Turn Probabilities (middle-left)
    ax3 = fig.add_subplot(gs[1, 0])
    
    # Only include non-diagonal elements with valid data
    valid_points = []
    for i in range(n_interfaces):
        for j in range(n_interfaces):
            if i != j and not np.isnan(q_matrix[i, j]) and not np.isnan(predicted_q[i, j]):
                valid_points.append((predicted_q[i, j], q_matrix[i, j], memory_significance[i, j],
                                    f"{i}→{j}"))
    
    if valid_points:
        # Unpack the valid points
        x_vals, y_vals, significance, labels = zip(*valid_points)
        
        # Plot the theoretical 1:1 line
        ax3.plot([0, 1], [0, 1], '--', color='gray', alpha=0.7)
        
        # Plot each point, colored by significance
        for x, y, sig, label in zip(x_vals, y_vals, significance, labels):
            color = 'red' if sig else 'blue'
            ax3.scatter(x, y, color=color, s=50, alpha=0.7)
            ax3.annotate(label, (x, y), xytext=(5, 5), textcoords='offset points')
        
        ax3.set_xlabel('Predicted Turn Probability (Free Energy Model)')
        ax3.set_ylabel('Observed Turn Probability')
        ax3.set_title('Observed vs Predicted Turn Probabilities', fontsize=12)
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim(0, 1)
        ax3.set_ylim(0, 1)
        
        # Add legend
        ax3.scatter([], [], color='blue', label='Free Energy Dominated')
        ax3.scatter([], [], color='red', label='Memory Effects')
        ax3.legend()
    else:
        ax3.text(0.5, 0.5, "Insufficient valid data for comparison",
                ha='center', va='center', transform=ax3.transAxes)
    
    # Plot 4: Turn Metrics - Average Turn Skip (middle-right)
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Plot average turn skip
    avg_turn_skip = turn_metrics['avg_turn_skip']
    valid_indices = ~np.isnan(avg_turn_skip)
    
    if np.any(valid_indices):
        bars = ax4.bar(np.array(range(n_interfaces))[valid_indices], 
                      avg_turn_skip[valid_indices], alpha=0.7)
        
        # Add value annotations
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f"{height:.2f}", ha='center', va='bottom', fontsize=9)
        
        ax4.set_xlabel('Interface')
        ax4.set_ylabel('Average Turn Skip (interfaces)')
        ax4.set_title('Average Distance Between Consecutive Turns', fontsize=12)
        ax4.set_xticks(range(n_interfaces))
        ax4.set_xticklabels([f"{i}" for i in range(n_interfaces)])
        ax4.grid(True, axis='y', alpha=0.3)
    else:
        ax4.text(0.5, 0.5, "Insufficient data for turn skip analysis",
                ha='center', va='center', transform=ax4.transAxes)
    
    # Plot 5: Turn Asymmetry (bottom-left)
    ax5 = fig.add_subplot(gs[2, 0])
    
    # Plot turn asymmetry
    turn_asymmetry = turn_metrics['turn_asymmetry']
    valid_indices = ~np.isnan(turn_asymmetry)
    
    if np.any(valid_indices):
        bars = ax5.bar(np.array(range(n_interfaces))[valid_indices], 
                      turn_asymmetry[valid_indices], alpha=0.7)
        
        # Color bars based on direction (forward/backward bias)
        for i, bar in enumerate(bars):
            idx = np.arange(n_interfaces)[valid_indices][i]
            asym = turn_asymmetry[idx]
            bar.set_color('green' if asym > 0 else 'red')
            
            # Add value annotations
            ax5.text(bar.get_x() + bar.get_width()/2., 
                    asym + 0.05 if asym >= 0 else asym - 0.1,
                    f"{asym:.2f}", ha='center', va='center', fontsize=9)
        
        ax5.set_xlabel('Interface')
        ax5.set_ylabel('Turn Asymmetry (-1 to 1)')
        ax5.set_title('Forward vs Backward Turn Preference', fontsize=12)
        ax5.set_xticks(range(n_interfaces))
        ax5.set_xticklabels([f"{i}" for i in range(n_interfaces)])
        ax5.set_ylim(-1.1, 1.1)
        ax5.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax5.grid(True, alpha=0.3)
        
        # Add legend
        ax5.text(0.02, 0.95, "Green: Forward bias (i→k where k>i)\nRed: Backward bias (i→k where k<i)",
                transform=ax5.transAxes, fontsize=10, va='top',
                bbox=dict(facecolor='white', alpha=0.8))
    else:
        ax5.text(0.5, 0.5, "Insufficient data for turn asymmetry analysis",
                ha='center', va='center', transform=ax5.transAxes)
    
    # Plot 6: Transition Sharpness (bottom-right)
    ax6 = fig.add_subplot(gs[2, 1])
    
    # Plot transition sharpness
    transition_sharpness = turn_metrics['transition_sharpness']
    valid_indices = ~np.isnan(transition_sharpness)
    
    if np.any(valid_indices):
        bars = ax6.bar(np.array(range(n_interfaces))[valid_indices], 
                      transition_sharpness[valid_indices], alpha=0.7)
        
        # Add value annotations
        for bar in bars:
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f"{height:.2f}", ha='center', va='bottom', fontsize=9)
        
        ax6.set_xlabel('Interface')
        ax6.set_ylabel('Transition Sharpness (0-1)')
        ax6.set_title('Turn Destination Specificity', fontsize=12)
        ax6.set_xticks(range(n_interfaces))
        ax6.set_xticklabels([f"{i}" for i in range(n_interfaces)])
        ax6.set_ylim(0, 1.1)
        ax6.grid(True, axis='y', alpha=0.3)
        
        # Add explanation
        ax6.text(0.02, 0.95, "Higher values indicate more focused transitions\nLower values indicate diffuse transitions",
                transform=ax6.transAxes, fontsize=10, va='top',
                bbox=dict(facecolor='white', alpha=0.8))
    else:
        ax6.text(0.5, 0.5, "Insufficient data for transition sharpness analysis",
                ha='center', va='center', transform=ax6.transAxes)
    
    # Adjust layout to use the entire figure space effectively
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave room at top for title
    
    return fig

def plot_destination_bias(p, interfaces=None, ax_forward=None, ax_backward=None, state_labels=None):
    """
    Create separate visualizations showing the average destination interface for forward 
    and backward transitions.
    
    Parameters:
    -----------
    p : numpy.ndarray
        Transition probability matrix where p[i,j] is the probability of
        transitioning from interface i to interface j
    interfaces : list or array, optional
        The positions of the interfaces along the reaction coordinate.
        If None, uses sequential indices.
    ax_forward : matplotlib.Axes, optional
        Axes to plot forward transitions on. If None, creates a new figure and axes.
    ax_backward : matplotlib.Axes, optional
        Axes to plot backward transitions on. If None, creates a new figure and axes.
    state_labels : list, optional
        Descriptive labels for each state. If None, uses sequential indices.
        
    Returns:
    --------
    tuple
        (ax_forward, ax_backward): The axes containing the forward and backward plots
    """
    import seaborn as sns
    
    n_interfaces = p.shape[0]
    
    if interfaces is None:
        interfaces = list(range(n_interfaces))
        is_equidistant = True
    else:
        # Check if interfaces are equidistant
        if len(interfaces) > 2:
            diffs = np.diff(interfaces)
            is_equidistant = np.allclose(diffs, diffs[0], rtol=0.05)
        else:
            is_equidistant = True
    
    # Generate state labels if not provided
    if state_labels is None:
        state_labels = generate_state_labels(n_interfaces)
    
    # Create new figures if axes not provided
    if ax_forward is None:
        _, ax_forward = plt.subplots(figsize=(10, 6))
    if ax_backward is None:
        _, ax_backward = plt.subplots(figsize=(10, 6))
    
    # Calculate forward average destinations (i→j where i<j)
    forward_destinations = np.zeros(n_interfaces)
    forward_valid = np.zeros(n_interfaces, dtype=bool)
    
    for i in range(n_interfaces-1):  # Skip last interface (no forward transitions)
        # Extract forward transitions (j > i)
        forward_probs = p[i, i+1:]
        forward_targets = np.arange(i+1, n_interfaces)
        
        # Calculate weighted average if there are transitions
        if np.sum(forward_probs) > 0:
            forward_destinations[i] = np.sum(forward_targets * forward_probs) / np.sum(forward_probs)
            forward_valid[i] = True
        else:
            forward_destinations[i] = np.nan  # No valid forward transitions
    
    # Calculate backward average destinations (i→j where i>j)
    backward_destinations = np.zeros(n_interfaces)
    backward_valid = np.zeros(n_interfaces, dtype=bool)
    
    for i in range(1, n_interfaces):  # Skip first interface (no backward transitions)
        # Extract backward transitions (j < i)
        backward_probs = p[i, :i]
        backward_targets = np.arange(i)
        
        # Calculate weighted average if there are transitions
        if np.sum(backward_probs) > 0:
            backward_destinations[i] = np.sum(backward_targets * backward_probs) / np.sum(backward_probs)
            backward_valid[i] = True
        else:
            backward_destinations[i] = np.nan  # No valid backward transitions
    
    # Calculate expected destinations
    # For forward transitions: expected = i + expected_jump
    # For backward transitions: expected = i - expected_jump
    # Calculate expected destinations
    forward_expected = np.zeros_like(forward_destinations)
    backward_expected = np.zeros_like(backward_destinations)
    
    for i in range(n_interfaces):
        if forward_valid[i]:
            # For forward transitions from i, calculate expected destination
            # For a diffusive system, this would be the probability-weighted average
            # of all possible destinations j where j > i
            probs = np.zeros(n_interfaces)
            for j in range(i+1, n_interfaces):
                # In a diffusive system, probability decreases with distance
                # We use a simple geometric series: p(j) = p(i+1) * r^(j-i-1)
                # where r is a decay factor (e.g., 0.5)
                dist_factor = 0.5 ** (j - i - 1)
                probs[j] = dist_factor
            
            # Normalize probabilities
            if np.sum(probs) > 0:
                probs = probs / np.sum(probs)
                # Calculate expected destination
                forward_expected[i] = np.sum(np.arange(n_interfaces) * probs)
        
        if backward_valid[i]:
            # For backward transitions from i, calculate expected destination
            # For a diffusive system, this would be the probability-weighted average
            # of all possible destinations j where j < i
            probs = np.zeros(n_interfaces)
            for j in range(i):
                # Similar decay factor for backward transitions
                dist_factor = 0.5 ** (i - j - 1)
                probs[j] = dist_factor
            
            # Normalize probabilities
            if np.sum(probs) > 0:
                probs = probs / np.sum(probs)
                # Calculate expected destination
                backward_expected[i] = np.sum(np.arange(n_interfaces) * probs)
    
    # Calculate bias
    forward_bias = forward_destinations - forward_expected
    backward_bias = backward_destinations - backward_expected
    
    # Get x values for plotting
    x_values = np.arange(n_interfaces) if is_equidistant else np.array(interfaces)
    
    # Plot forward transitions
    plot_directional_bias(ax_forward, x_values, forward_destinations, forward_expected, 
                       forward_bias, forward_valid, 'Forward Transitions (i→j where i<j)', 'higher',
                       interfaces, is_equidistant, [(f'{i}$\\supset$' if i > 0 else f'{i}') for i in range(n_interfaces-1)])
    
    # Plot backward transitions
    plot_directional_bias(ax_backward, x_values, backward_destinations, backward_expected, 
                       backward_bias, backward_valid, 'Backward Transitions (i→j where i>j)', 'lower',
                       interfaces, is_equidistant, [(f'{i}$\\supset$' if i > 0 else f'{i}') for i in range(n_interfaces-1)])
    
    return ax_forward, ax_backward

def plot_directional_bias(ax, x_values, avg_destinations, expected_destinations, bias, valid_mask,
                       title, direction, interfaces, is_equidistant, state_labels=None):
    """Helper function to create a directional bias plot on the given axes"""    
    n_interfaces = len(x_values)
    
    if state_labels is None:
        state_labels = generate_state_labels(n_interfaces)
    
    if direction == 'higher':
        xxt = [(f'{i}$\\subset$' if i > 0 else f'{i}') for i in range(n_interfaces-1)] + [f'{n_interfaces-1}']
        yyt = [(f'{i}' if i > 0 else f'{i}') for i in range(n_interfaces)]
    else:
        xxt = [(f'{i}$\\supset$' if i > 0 else f'{i}') for i in range(n_interfaces-1)] + [f'{n_interfaces-1}']
        yyt = [(f'{i}' if i > 0 else f'{i}') for i in range(n_interfaces)]
    
    # Plot expected destination
    ax.plot(x_values[valid_mask], expected_destinations[valid_mask], '--', color='gray', 
           label='Expected destination')
    
    # Plot actual average destination
    ax.plot(x_values[valid_mask], avg_destinations[valid_mask], 'o-', color='blue', linewidth=2,
           markersize=8, label='Actual average')
    
    # Add annotations showing the bias
    for i, valid in enumerate(valid_mask):
        if valid:
            bias_text = f"{bias[i]:.2f}"
            text_color = 'red' if bias[i] > 0.2 else ('blue' if bias[i] < -0.2 else 'black')
            ax.annotate(bias_text, 
                      xy=(x_values[i], avg_destinations[i]), 
                      xytext=(0, 10 if bias[i] > 0 else -15),
                      textcoords='offset points',
                      ha='center', va='center',
                      color=text_color, fontsize=9,
                      bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8))

    # Configure the plot
    ax.set_xlabel('Starting Interface i')
    ax.set_ylabel('Average Destination Interface')
    ax.set_title(f'Average Destination Analysis: {title}', fontsize=12)
    
    # Set appropriate axis limits
    valid_y = avg_destinations[valid_mask]
    if len(valid_y) > 0:
        y_min, y_max = np.nanmin(valid_y), np.nanmax(valid_y)
        expected_min, expected_max = np.nanmin(expected_destinations[valid_mask]), np.nanmax(expected_destinations[valid_mask])
        
        y_min = min(y_min, expected_min)
        y_max = max(y_max, expected_max)
        
        y_range = y_max - y_min
        buffer = 0.5 if is_equidistant else (interfaces[1] - interfaces[0]) / 2
        ax.set_xlim(np.min(x_values) - buffer, np.max(x_values) + buffer)
        ax.set_ylim(y_min - y_range*0.1, y_max + y_range*0.1)
    
    # Add gridlines for easier reading
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Create a better legend
    handles, labels = ax.get_legend_handles_labels()
    keep_indices = [0, 1]  # Just keep the main curves
    ax.legend([handles[i] for i in keep_indices if i < len(handles)], 
             [labels[i] for i in keep_indices if i < len(labels)],
             loc='best', fontsize=10)
    
    # Set the x-axis ticks and labels using state_labels
    ax.set_xticks(x_values)
    ax.set_xticklabels(xxt)
    
    # Create y-ticks with state labels
    y_ticks = np.arange(n_interfaces)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(yyt)
    
    # Add explanatory text
    if direction == 'higher':
        info_text = """
        Average destination for forward transitions (i→j where i<j):
        • Numbers indicate deviation from expected destination
        • Positive values (red): Paths go further than expected
        • Negative values (blue): Paths go less far than expected
        """
    else:  # 'lower'
        info_text = """
        Average destination for backward transitions (i→j where i>j):
        • Numbers indicate deviation from expected destination
        • Positive values (red): Paths go less far back than expected
        • Negative values (blue): Paths go further back than expected
        """
    
    # ax.text(0.02, 0.02, info_text, transform=ax.transAxes, fontsize=9,
    #       bbox=dict(facecolor='white', alpha=0.9, boxstyle='round'), va='bottom')

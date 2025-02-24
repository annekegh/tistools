import numpy as np
import logging
from .reading import set_flags_ACC_REJ

# Hard-coded rejection flags found in output files
ACCFLAGS,REJFLAGS = set_flags_ACC_REJ() 

# logger
logger = logging.getLogger(__name__)

def construct_tau_vector(N, NS, taumm, taump, taupm, taupp):
    assert N>=3
    assert NS==4*N-5
    assert len(taumm) == N
    assert len(taump) == N
    assert len(taupm) == N
    assert len(taupp) == N
    # unravel the values into one vector
    tau = np.zeros(NS)
    # [0-]
    tau[0] = taupp[0]
    # [0+-]
    tau[1] = taumm[1]
    tau[2] = taump[1]
    tau[3] = taupm[1]
    # [1+-] etc
    for i in range(1,N-2):
        tau[4*i]   = taumm[i+1]
        tau[4*i+1] = taump[i+1]
        tau[4*i+2] = taupm[i+1]
        tau[4*i+3] = taupp[i+1]
    # [(N-2)^(-1)]
    tau[-3] = taumm[-1]
    tau[-2] = taump[-1]
    # B
    tau[-1] = 0.   # whatever
    return tau


def set_tau_first_hit_M_distrib(pe, do_last = True):
    """Set, for each pathtype, the average pathlength before the middle 
    interface is crossed. The phasepoint at the beginning, and right after 
    the crossing will still be included.
    
    Parameters
    ----------
    pe : object like :py:class:`.PathEnsemble`
        Tistools PathEnsemble object must be from a PPTIS simulation,
        for which the weights and the orders have been set.

    do_last: whether tau2 is computed (default: True)
        
    Returns
    -------
    Nothing, but sets the attribute pe.tau1 and pe.tau1avg.

    """
    pe.tau1 = []
    pe.tau1avg = {"LML": None, "LMR": None, "RML": None, "RMR": None}
    # select the accepted paths
    for i in range(len(pe.cyclenumbers)):
        if pe.flags[i] != "ACC" or pe.generation[i] == "ld":
            pe.tau1.append(0)
            continue 
        pe.tau1.append(get_tau1_path(pe.orders[i], pe.lmrs[i], pe.interfaces[0]))
    pe.tau1 = np.array(pe.tau1) 

    # get the average tau1 for each path type. Each path has a weight w.
    for ptype in ("LML", "LMR", "RML", "RMR"):
        totweight = np.sum(pe.weights[pe.lmrs == ptype])
        if totweight != 0:   # make sure total weight is not zero
            pe.tau1avg[ptype] = np.average(pe.tau1[pe.lmrs == ptype], 
                                       weights=pe.weights[pe.lmrs == ptype])

    if not do_last:
        return

    pe.tau2 = []
    pe.tau2avg = {"LML": None, "LMR": None, "RML": None, "RMR": None}
    # select the accepted paths
    for i in range(len(pe.cyclenumbers)):
        if pe.flags[i] != "ACC" or pe.generation[i] == "ld":
            pe.tau2.append(0)
            continue
        pe.tau2.append(get_tau2_path(pe.orders[i], pe.lmrs[i], pe.interfaces[0]))
    pe.tau2 = np.array(pe.tau2)

    # get the average tau2 for each path type. Each path has a weight w.
    for ptype in ("LML", "LMR", "RML", "RMR"):
        totweight = np.sum(pe.weights[pe.lmrs == ptype])
        if totweight != 0:   # make sure total weight is not zero
            pe.tau2avg[ptype] = np.average(pe.tau2[pe.lmrs == ptype],
                                       weights=pe.weights[pe.lmrs == ptype])


def get_tau1_path(orders, ptype, intfs):
    """Return the number of steps it took for this path to cross the M interface
        
    get 1st index after starting point = a    # usually a=1
    get 1st index after crossing M     = b
    return b-a = the number of phase points in the zone

    Example:
    interface     L              M           R
    phasepoint  x | x    x  x  x | x  x    x | x
    index       0   1    2  3  4   5  6    7   8
                    a              b
    then b-a = 5-1 = 4 phasepoints in the zone
    """
    if ptype in ("LMR", "LML"):
        a = np.where(orders[:,0] >= intfs[0])[0][0]  # L->M->. cross
        b = np.where(orders[:,0] >= intfs[1])[0][0]  # L->M->. cross
        return b-a
    elif ptype in ("RML", "RMR"):
        a = np.where(orders[:,0] <= intfs[2])[0][0]  # .<-M<-R cross
        b = np.where(orders[:,0] <= intfs[1])[0][0]  # .<-M<-R cross
        return b-a
    else:
        raise ValueError(f"Unknown path type {ptype}")

def get_tau2_path(orders, ptype, intfs):
    """Return the number of steps in the path after the last crossing of M interface"""
    #very similar to get_tau1_path
    # "a" serves to cut off a piece, usually a=1
    if ptype in ("LML", "RML"):
        a = np.where(orders[::-1,0] >= intfs[0])[0][0]  # L<-M<-. cross
        b = np.where(orders[::-1,0] >= intfs[1])[0][0]  # L<-M<-. cross
        return b-a
    elif ptype in ("LMR", "RMR"):
        a = np.where(orders[::-1,0] <= intfs[2])[0][0]  # .->M->R cross
        b = np.where(orders[::-1,0] <= intfs[1])[0][0]  # .->M->R cross
        return b-a
    else:
        raise ValueError(f"Unknown path type {ptype}")

def get_tau_path(orders, ptype, intfs):
    # cut off piece at start
    if ptype in ("LMR", "LML"):
        a1 = np.where(orders[:,0] >= intfs[0])[0][0]  # L->M->. cross
    elif ptype in ("RML", "RMR"):
        a1 = np.where(orders[:,0] <= intfs[2])[0][0]  # .<-M<-R cross
    else:
        raise ValueError(f"Unknown path type {ptype}")
    # cut off piece at end
    if ptype in ("LML", "RML"):
        a2 = np.where(orders[::-1,0] >= intfs[0])[0][0]  # L<-M<-. cross
    elif ptype in ("LMR", "RMR"):
        a2 = np.where(orders[::-1,0] <= intfs[2])[0][0]  # .->M->R cross
    else:
        raise ValueError(f"Unknown path type {ptype}")
    b = len(orders)       # len(pe.orders[i]) = path length of path i
    return b-a1-a2
    

def set_tau_distrib(pe):
    """Set, for each pathtype, the average total pathlength. The phasepoint at
    the beginning, and right after the crossing will still be included.
    
    Parameters
    ----------
    pe : object like :py:class:`.PathEnsemble`
        Tistools PathEnsemble object must be from a PPTIS simulation,
        for which the weights and the orders have been set.
        
    Returns
    -------
    Nothing, but sets the attribute pe.tau and pe.tauavg.
    """
    pe.tau = []
    pe.tauavg = {"LML": None, "LMR": None, "RML": None, "RMR": None,
                 "L*L": None, "R*R": None}
    # determine pathtypes
    if pe.in_zero_minus:
        if pe.has_zero_minus_one:
            ptypes = ["LML", "LMR", "RML", "RMR", "L*L", "R*R"]
        else:
            ptypes = ["RMR",]
    else:
            ptypes = ["LML", "LMR", "RML", "RMR",]
    #pe.tauavg = {}
    for ptype in ptypes:
        pe.tauavg[ptype] = None

    # select the accepted paths and collect path lengths
    for i in range(len(pe.cyclenumbers)):
        if pe.flags[i] != "ACC" or pe.generation[i] == "ld":
            pe.tau.append(0)
            continue 
        pe.tau.append(get_tau_path(pe.orders[i], pe.lmrs[i], pe.interfaces[0]))
        # pe.orders[i] contains the order parameters of the i-th path
        # including the start/end point
    pe.tau = np.array(pe.tau)

    # get the average tau for each path type. Each path has a weight w.
    for ptype in ptypes:
        totweight = np.sum(pe.weights[pe.lmrs == ptype])
        if totweight != 0:   # make sure total weight is not zero
            pe.tauavg[ptype] = np.average(pe.tau[pe.lmrs == ptype], 
                                      weights=pe.weights[pe.lmrs == ptype])


# COLLECTING
# collect_tau
# collect_tau1
# collect_tau2
# collect_taum

def collect_tau(pathensembles):
    """Compute average path lengths"""
    
    # pathensembles -- list of pathensemble instances
    print("Collect tau")
    taumm = np.zeros(len(pathensembles))
    taump = np.zeros(len(pathensembles))
    taupm = np.zeros(len(pathensembles))
    taupp = np.zeros(len(pathensembles))
    
    for i in range(len(pathensembles)):
        print("ensemble", i, pathensembles[i].name)
        taumm[i] = pathensembles[i].tauavg['LML']
        taump[i] = pathensembles[i].tauavg['LMR']
        taupm[i] = pathensembles[i].tauavg['RML']
        taupp[i] = pathensembles[i].tauavg['RMR']
    # TODO pieces missing [0-]   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    return taumm, taump, taupm, taupp


def collect_tau1(pathensembles):
    """Compute and collect average time to hit M"""
    # average path lengths, but only the part before the 1st crossing
    # points before start and after M are not counted
    print("Collect tau1")
    tau1_mm = np.zeros(len(pathensembles))
    tau1_mp = np.zeros(len(pathensembles))
    tau1_pm = np.zeros(len(pathensembles))
    tau1_pp = np.zeros(len(pathensembles)) 

    for i in range(len(pathensembles)):
        tau1_mm[i] = pathensembles[i].tau1avg['LML']
        tau1_mp[i] = pathensembles[i].tau1avg['LMR']
        tau1_pm[i] = pathensembles[i].tau1avg['RML']
        tau1_pp[i] = pathensembles[i].tau1avg['RMR']
    return tau1_mm, tau1_mp, tau1_pm, tau1_pp

def collect_tau2(pathensembles):
    """Compute and collect average time after hitting M"""
    # average path lengths, but only the part after the last crossing
    # points before M and after end are not counted
    print("Collect tau2")
    tau2_mm = np.zeros(len(pathensembles))
    tau2_mp = np.zeros(len(pathensembles))
    tau2_pm = np.zeros(len(pathensembles))
    tau2_pp = np.zeros(len(pathensembles))
    
    for i in range(len(pathensembles)):
        tau2_mm[i] = pathensembles[i].tau2avg['LML']
        tau2_mp[i] = pathensembles[i].tau2avg['LMR']
        tau2_pm[i] = pathensembles[i].tau2avg['RML']
        tau2_pp[i] = pathensembles[i].tau2avg['RMR']
    return tau2_mm, tau2_mp, tau2_pm, tau2_pp

def collect_taum(pathensembles):
    """Compute and collect average time between first and last hit of M"""
    # average path lengths, but only the part after the first crossing
    # and before the last crossing of M
    # so, point in the middle (m)
    # other points are not counted
    print("Collect taum")
    taum_mm = np.zeros(len(pathensembles))
    taum_mp = np.zeros(len(pathensembles))
    taum_pm = np.zeros(len(pathensembles))
    taum_pp = np.zeros(len(pathensembles))

    for i in range(len(pathensembles)):
        pe = pathensembles[i]
        # check if tau exists for this path type, then rest should exist too
        if pe.tauavg['LML'] is not None:
            taum_mm[i] = pathensembles[i].tauavg['LML'] \
                    - pathensembles[i].tau1avg['LML'] \
                    - pathensembles[i].tau2avg['LML']
        if pe.tauavg['LMR'] is not None:            
            taum_mp[i] = pathensembles[i].tauavg['LMR'] \
                   - pathensembles[i].tau1avg['LMR'] \
                   - pathensembles[i].tau2avg['LMR']
        if pe.tauavg['RML'] is not None:
            taum_pm[i] = pathensembles[i].tauavg['RML'] \
                   - pathensembles[i].tau1avg['RML'] \
                   - pathensembles[i].tau2avg['RML']
        if pe.tauavg['RMR'] is not None:
            taum_pp[i] = pathensembles[i].tauavg['RMR'] \
                   - pathensembles[i].tau1avg['RMR'] \
                   - pathensembles[i].tau2avg['RMR']

    return taum_mm, taum_mp, taum_pm, taum_pp


def set_taus(pe):
    """Set, for each path type, the average pathlength before and after the
    middle interface is crossed, and the average total path length.

    Parameters
    ----------
    pe : object like :py:class:`.PathEnsemble`
        Tistools PathEnsemble object must be from a PPTIS simulation,
        for which the weights and the orders have been set.

    Returns
    -------
    Nothing, but sets the attributes pe.tau1, pe.tau2, pe.tau, and their averages.
    """
    pe.tau1 = []
    pe.tau2 = []
    pe.tau = []
    
    pe.tau1avg = {"LML": None, "LMR": None, "RML": None, "RMR": None}
    pe.tau2avg = {"LML": None, "LMR": None, "RML": None, "RMR": None}
    pe.tauavg = {"LML": None, "LMR": None, "RML": None, "RMR": None,
                 "L*L": None, "R*R": None}

    # Determine path types
    if pe.in_zero_minus:
        if pe.has_zero_minus_one:
            ptypes = ["LML", "LMR", "RML", "RMR", "L*L", "R*R"]
        else:
            ptypes = ["RMR",]
    else:
        ptypes = ["LML", "LMR", "RML", "RMR"]

    # Loop over all paths, compute tau1, tau2, and total tau
    for i in range(len(pe.cyclenumbers)):
        if pe.flags[i] != "ACC" or pe.generation[i] == "ld":
            # If not accepted or if generation is "ld", set zero for all values
            pe.tau1.append(0)
            pe.tau2.append(0)
            pe.tau.append(0)
            continue
        
        # Compute tau1 and tau2
        tau1_value = get_tau1_path(pe.orders[i], pe.lmrs[i], pe.interfaces[0])
        tau2_value = get_tau2_path(pe.orders[i], pe.lmrs[i], pe.interfaces[0])

        # Compute total tau (the path length)
        tau_value = get_tau_path(pe.orders[i], pe.lmrs[i], pe.interfaces[0])

        # Append results
        pe.tau1.append(tau1_value)
        pe.tau2.append(tau2_value)
        pe.tau.append(tau_value)

    # Convert lists to numpy arrays for efficient processing
    pe.tau1 = np.array(pe.tau1)
    pe.tau2 = np.array(pe.tau2)
    pe.tau = np.array(pe.tau)

    # Calculate the average tau1, tau2, and total tau for each path type
    for ptype in ptypes:
        # Total weight for each path type
        totweight = np.sum(pe.weights[pe.lmrs == ptype])
        
        # Compute average tau1 for the current path type
        if totweight != 0:  # Ensure total weight is not zero
            pe.tau1avg[ptype] = np.average(pe.tau1[pe.lmrs == ptype],
                                           weights=pe.weights[pe.lmrs == ptype])
            pe.tau2avg[ptype] = np.average(pe.tau2[pe.lmrs == ptype],
                                           weights=pe.weights[pe.lmrs == ptype])
            pe.tauavg[ptype] = np.average(pe.tau[pe.lmrs == ptype],
                                          weights=pe.weights[pe.lmrs == ptype])
            

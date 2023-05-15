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
def create_figure_trajs(figbasename,data_op,length,accepted,first,acc=True,interfaces=None,generation=None):

    # TODO I did not use "acc=True"

    ntraj,indices = get_ntraj(accepted,acc=acc)
    # only take first trajectories:
    maxtraj = min(first,ntraj)
    ncol = int(np.ceil(np.sqrt(maxtraj)))

    fig = plt.figure(figsize=(ncol*3,ncol*3))  # in inches
    
    for i in range(ncol):
        for j in range(ncol):
            count = i*ncol+j
            if count>=maxtraj: break

            index = indices[count]
            da = select_traj(data_op,length,index)

            plt.subplot(ncol,ncol,count+1)
            if interfaces is not None:
                # plot interface
                for val in interfaces:
                    plt.plot([0,len(da)-1],[val,val],color='grey')

            # plot order parameter:
            # time starts from zero, op is in data col=0
            plt.plot(da[:,0],linewidth=1,marker='o',markersize=3)
            # plot other particles alos, these are in col=1...
            for k in range(1,da.shape[1]):
                plt.plot(da[:,k])

            # add title
            if generation is None:
                plt.title("cycle %i"%(index))
            else:
                plt.title("cycle %i %s"%(index,generation[index]))
    
    plt.tight_layout()
    plt.savefig(figbasename+".png")
    plt.savefig(figbasename+".pdf")
    plt.close()


def make_plot_trajs(figbasename,folder,interfaces,first):
    """
    folder -- can be somedir/000, somedir/001, somedir/002, etc
    interfaces -- [lambda0], [lambda[0], [lambda0,lambda1], etc...
    """
    op,ep = get_data_ensemble_consistent(folder)

    accepted = [(acc == 'ACC') for acc in op.flags]

    create_figure_trajs(figbasename,op.ops,op.lengths,accepted,first,
        acc=True,interfaces=interfaces,generation=op.generation)


def analyze_lengths(simul,ens,interfaces,above=0,skip=0):

    op,ep = get_data_ensemble_consistent(simul,ens,interfaces,)

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
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values-average)**2, weights=weights)
    return (average, np.sqrt(variance))
    #mean = np.average(lengths,weights=weights)

def histogram_lengths(figname,retislengths,retisweights,dtc,above=0,skip=0):

    def plot_histogram(figname,hist,bin_mids,mean,std):
        plt.figure()
        plt.plot(bin_mids,hist,)
        plt.xlabel("time (in ps)")
        plt.ylabel("count")
        plt.title("mean %.2f +- %.2f (ps), binwidth=%.1f (ps)" %(mean,std,bin_mids[1]-bin_mids[0]))
        plt.plot([mean,mean,],[min(hist),max(hist)],color='red',linewidth=2)
        plt.plot([mean-std,mean-std,],[min(hist),max(hist)],color='grey',linewidth=2)
        plt.plot([mean+std,mean+std,],[min(hist),max(hist)],color='grey',linewidth=2)
        plt.savefig(figname)
        plt.close()

    assert len(retislengths) == len(retisweights)
    assert (skip>=0) and skip<len(retislengths)

    # skip some lines
    lengths = retislengths[skip:]
    weights = retisweights[skip:]

    lengths = np.array([l-2 for l in lengths])  # delete first and last point

    #bins = 50
    bins = np.linspace(0,600,61)
    # for flat:
    if FLAT1D:       # TODO
        bins =50

    if above == 0:
        hist,bin_edges = np.histogram(lengths*dtc,bins=bins,weights=weights)
        bin_mids = 0.5*(bin_edges[:-1]+bin_edges[1:])
        #bin_mids = bin_edges[:-1]+np.diff(bin_edges)/2.
        mean,std = weighted_avg_and_std(lengths*dtc, weights)
        #plot_histogram(figbasename+".all",hist,bin_mids,mean,std,)

    else:
        tooshort = above
        weights2 = weights*(lengths>tooshort)
        hist,bin_edges = np.histogram(lengths*dtc,bins=bins,weights=weights2)
        bin_mids = 0.5*(bin_edges[:-1]+bin_edges[1:])
        #bin_mids = bin_edges[:-1]+np.diff(bin_edges)/2.
        mean,std = weighted_avg_and_std(lengths*dtc, weights2)

    plot_histogram(figname,hist,bin_mids,mean,std)

#    tooshort = 50
#    weights2 = weights*(lengths>tooshort)
#    hist,bin_edges = np.histogram(lengths*dtc,bins=bins,weights=weights2)
#    bin_mids = 0.5*(bin_edges[:-1]+bin_edges[1:])
#    #bin_mids = bin_edges[:-1]+np.diff(bin_edges)/2.
#    mean,std = weighted_avg_and_std(lengths*dtc, weights2)
#    plot_histogram(figbasename+".above%i"%tooshort,hist,bin_mids,mean,std)




#-------------------------------------------------
# Investigate efficiency of sampling
#-------------------------------------------------

#-------------------------------------------------
# Investigate decay of first path (ld, from load)
#-------------------------------------------------

def decay_path(figbasename,list_ensembles,):

    nens = len(list_ensembles)
    all_data_path = []
    for ens in list_ensembles:
        fn_path = "%s/pathensemble.txt"%(ens)
        pe = read_pathensemble(fn_path)
        all_data_path.append(pe)

    # determine size
    ncycles = [pe.ncycle for pe in all_data_path]
    max_cycles = max(ncycles)
    print("ncycles:   ", ncycles)
    print("max_cycles:", max_cycles)
    # or index of cycles
    ncycles = [pe.cyclenumbers[-1] for pe in all_data_path]
    max_cycles = max(ncycles)
    print("ncycles:   ", ncycles)
    print("max_cycles:", max_cycles)

    # cycle numbers
    cycle_numbers = [pe.cyclenumbers.tolist() for pe in all_data_path]

    # fill in the decay matrix
    decay = np.zeros((max_cycles,nens))
    factor = 0.8
    decay[0,:] = 1.

    row_index = np.zeros(nens)
    for i in range(1,max_cycles):
        for j,pe in enumerate(all_data_path):
            # just copy previous
            decay[i,j] = decay[i-1,j]
            # is this index present in ensemble j?
            try:
                index = cycle_numbers[j].index(i)
                #if i in cycle_numbers[j]
                # different cases
                if pe.flags[index] == 'ACC':
                    if pe.generation[index] == 'ld':
                        decay[i,j] = 1. 
                    elif pe.generation[index] == 'sh':
                        decay[i,j] = decay[i-1,j]*factor
                    elif pe.generation[index] == '00':
                        decay[i,j] = decay[i-1,j]
                    elif pe.generation[index] == 'tr':
                        decay[i,j] = decay[i-1,j]
                    elif pe.generation[index] == 's+':
                        decay[i,j] = decay[i-1,j+1]
                    elif pe.generation[index] == 's-':
                        decay[i,j] = decay[i-1,j-1]
                    else:
                       print("FAIL")
            except ValueError:
                pass

    print("decay:",decay.shape)
    #print(decay)
    plot_decay(figbasename,decay,list_ensembles)
    list_newpathnumbers = [pe.newpathnumbers for pe in all_data_path]
    plot_newpathnumbers(figbasename+".newpaths.png",list_newpathnumbers)

    print("Counting phase points")
    totcount = 0
    for i,pe in enumerate(all_data_path):
        count = compute_phasepoints(pe.lengths,pe.flags,pe.generation,"{:03d}".format(i))
        totcount += count
        print("count {:03d} {}".format(i,count))
    dt = 0.0002  # ps   # TODO
    print("ASSUME dt=%f"%dt)
    print("totcount = {} = {:.1f} ps = {:.4f} ns".format(totcount,totcount*dt,totcount*dt/1000.))
        

def plot_decay(figbasename,decay,list_ensembles):
    """Plot the decay of the load path"""
    plt.figure()
    plt.plot(decay[:150,:])
    plt.xlabel("cycle")
    plt.ylabel("initial cond survival")
    plt.legend(list_ensembles)
    plt.savefig(figbasename+".zoom.png")
    plt.clf()

    plt.plot(decay)
    plt.xlabel("cycle")
    plt.ylabel("initial cond survival")
    plt.legend(list_ensembles)
    plt.savefig(figbasename+".png")


def plot_newpathnumbers(figname,list_newpathnumbers):
    """Investigate how many new paths are generated"""
    plt.figure()
    for i,newpathnumbers in enumerate(list_newpathnumbers):
        plt.plot(newpathnumbers,label = str(i))
    plt.xlabel("cycle")
    plt.ylabel("# new paths")
    plt.legend()
    plt.savefig(figname)

#-------------------------------------------------
# Compute how many phase points were computed
#-------------------------------------------------

def compute_phasepoints(lengths,flags,generation,ensemble):
    """compute the number of phasepoints that were calculated in this ensemble"""
    # ensemble -- 000 or 001 or 002, etc
    ncycle = len(lengths)
    assert ncycle == len(flags)
    assert ncycle == len(generation)

    count = 0
    for i in range(ncycle):
        if generation[i] in ["sh","ki"]:
            # one point is recycled, i.e. the shooting point
            count += lengths[i] - 1

        elif generation[i] in ["tr","00"]:
            pass    # do not add anything

        elif generation[i] in ["s+","s-"]:
            if ensemble in ["000","001"]:
                # two points are recycled, 1 before and 1 after the interface
                count += lengths[i]-2
            else:
                pass    # do not add anything
    return count


#    list_generation = [pe.generation for pe in list_pe]



#-------------------------------------------------
# computation for xi
# The Big Reweight
#-------------------------------------------------
def calc_xi(lmrs,weights):
    """Compute the factor xi for ensemble [0-']

    xi = (# paths) / (# paths that end at R)
    """
    n_lml = np.sum((lmrs=="LML")*weights)
    n_rml = np.sum((lmrs=="RML")*weights)
    n_lmr = np.sum((lmrs=="LMR")*weights)
    n_rmr = np.sum((lmrs=="RMR")*weights)
    n_lstarl = np.sum((lmrs=="L*L")*weights)
    n_rstarl = np.sum((lmrs=="R*L")*weights)
    n_lstarr = np.sum((lmrs=="L*R")*weights)
    n_rstarr = np.sum((lmrs=="R*R")*weights)
    n_ends_l = n_lml + n_rml + n_lstarl + n_rstarl
    n_ends_r = n_lmr + n_rmr + n_lstarr + n_rstarr

    n_all = np.sum(weights)

    # the load path (ld) can be LM* or so, and still be accepted with a weight>0
    # # (a bit weird). Therefore, I SKIP the load path if this is the case, ￼   
    # in the calculation of xi

    if weights[0] > 0:
        if lmrs[0] not in ["LML", "RML", "RMR", "LMR", "L*L", "R*L", "L*R", "R*R"]:
            n_all = np.sum(weights[1:])  # skip this first path
    
    assert n_all == n_ends_r + n_ends_l
    
    if n_ends_r > 0:
        print("big reweight!!")
        xi = n_all/n_ends_r

        # earlier attempt:
        # theory: factor = ncycle / (ncycle - n-(RML/LMR))
        #n_left2 = n_rml + n_lmr # n_left = 0.5*(n_rml+n_lmr)
        #factor = 2*ncycle_true / (2*ncycle_true - n_left2)
        #print("factor",factor)
    else:
        print("no big reweight!!, because n_end_r <= 0")
        xi = 1
    return xi

def print_lmr_000(lmrs,weights):
    """print the codes of 000, such as LML and RML"""
    print("count paths in 000")
    for code in ["LML","LMR","L*R","L*L","RMR","RML","R*R","R*L"]:
        n = np.sum(lmrs==code)
        nw = np.sum((lmrs==code)*weights)
        print(code,n,nw)


def print_concentration_lambda0(ncycle,trajs,cut,dlambda,dt,w_all,xi):
    """Compute density at interface lambda0"""

    print("."*10)
    print("for permeability")
    print("n         --  count in histogram bin")
    print("n/ncycle  --  count in histogram bin, per path")
    print("rho       --  density, unit [1/unit-of-dlambda]")
    print("tau/Dz    --  time spent per length, unit [unit-of-dt/unit-of-dlambda]")
    print("P/Pcross  --  permeability/crossing-probability, unit [1/unit-of-dlambda/unit-of-dt]")

    print("dt       ",dt)
    print("dlambda  ",dlambda)

    print("Get some values around interface")
    # take last bin of histogram to the left (histL)
    bins = np.arange(cut-30*dlambda,cut+dlambda/10.,dlambda)      # HARD CODED
    histL,edgesL = np.histogram(trajs,bins=bins,weights=w_all,)
    nL = histL[-1]   # because of choice bins
    print("bins at cut: lambda =",cut)
    print("left of cut: ",histL[-2],histL[-1],"|")

    # take first bin of histogram to the right (histR)
    bins = np.arange(cut,cut+30*dlambda+dlambda/10.,dlambda)      # HARD CODED
    histR,edgesR = np.histogram(trajs,bins=bins,weights=w_all)
    nR = histR[0]   # because of choice bins
    print("right of cut:","|",histR[0],histR[1])

    #ncycle_true = np.sum(weights)  # do not compute again because I have the big reweigth?
    #ncycle can be a little bit more

    print("Now compute n, rho, P/Pcross; left (L) and right (R) of the interface")
    rhoL = nL/dlambda/ncycle    # in units of 1/dlambda
    tauL = nL*dt/dlambda/ncycle  # in units of dt/dlambda
    PL = 1./tauL      # = P/Pcross = 1/dt/rhoL = ncycle / nL * dlambda/dt
    rhoR = nR/dlambda/ncycle
    tauR = nR*dt/dlambda/ncycle  # in units of dt/dlambda
    PR = 1./tauR      # = ncycle / nR * dlambda/dt

    tauL = nL*dt/dlambda/ncycle  # in units of dt/dlambda

    print("** Left histogram **")
    print("L n            ",nL)
    print("L n/ncycle     ",nL/ncycle)
    print("L rho          ",rhoL)
    print("L tau/Dz       ",tauL)
    print("L P/Pcross     ",PL)
    print("** Left histogram (xi)**")
    print("L n        (xi)",nL*xi)
    print("L n/ncycle (xi)",nL/ncycle*xi)
    print("L rho      (xi)",rhoL*xi)
    print("L tau/Dz   (xi)",tauL*xi)
    print("L P/Pcross (xi)",PL/xi)
    print("** Right histogram **")
    print("R n            ",nR)
    print("R n/ncycle     ",nR/ncycle)
    print("R rho          ",rhoR)
    print("L tau/Dz       ",tauR)
    print("R P/Pcross     ",PR)
    print("."*10)
    #print("Sum/ncycle",len(trajs)/ncycle)  # no, because not weighted TODO
    #print("Sum/ncycle no xi",len(trajs)/ncycle/xi)

    return nL
    

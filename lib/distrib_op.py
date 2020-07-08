import numpy as np
from .reading import *
from .analyze_op import *
import matplotlib.pyplot as plt

ACCFLAGS,REJFLAGS = set_flags_ACC_REJ() # Hard-coded rejection flags found in output files


def get_bins(folders,interfaces,dlambda,lmin=None,lmax=None):
    """bins: fix the lambda interval"""
    if lmin is None:
        if len(folders)<len(interfaces):
            zero_left = interfaces[-1]
            lm = zero_left - 10*dlambda
        else:
            lm = interfaces[0]-30*dlambda
    else:
        lm = lmin
    if lmax is None:
        lM = interfaces[len(folders)-1] + 10*dlambda # not interfaces[-1] because there might be a zero_left
    else:
        lM = lmax
    bins = np.arange(lm,lM+dlambda/10.,dlambda)
    return bins

    #What I used to have for make_histogram_op:
    #bins = np.linspace(-3.6,-2.2,101)

def create_distrib(folders,interfaces,dt,dlambda,lmin,lmax,dlambda_conc,
    op_index,op_weight,do_abs):
    """create figure of distributions of order parameter
    First figure: only path ensembles 000 and 001
    Second figure: all path ensembles

    folders -- ["somedir/000", "somedir/001", "somedir/002", ...]
    interfaces -- [lambda0,lambda1,...,lambdaN]
                    if lambda_-1 exists: [lambda0,lambda1,...,lambdaN, lambda_-1]
    op_index -- list of selected order parameters, e.g. [0] or [0,2,3]
    op_weight -- weight for each of the selected order parameters
    do_abs -- wether to take the absolute value of the order parameters before histogramming
    """
    bins = get_bins(folders,interfaces,dlambda,lmin,lmax)

    plt.figure(1)
    if len(folders)>2:
        plt.figure(2)

    # folder -- somedir/000, somedir/001, etc
    # fol  --  000, 001, etc
    # ifol  --  0, 1, etc
    for ifol,folder in enumerate(folders):
        fol = folder[-3:]
        print("-"*20)
        print("folder",fol)
        ofile = "%s/order.txt" %(folder)
        #ostart = 0
        ostart = -1    # TODO
        op = read_order(ofile,ostart)
        flags = op.flags
        trajs = op.longtraj
        lens = op.lengths
        ncycle = op.ncycle   # this is len(flags) = len(lengths)
        weights, ncycle_true = get_weights(flags,ACCFLAGS,REJFLAGS)


        # TODO EXTRA
        #print(len(op.ops),op.ops.shape)
        #print(op.longtraj.shape)
        assert len(op.ops) == len(op.longtraj)
        # TODO print(ah)

        print("ncycle:",ncycle)

        # make one long list of weights
        w_all = np.array([ weights[i] for i in range(ncycle) for k in range(lens[i])])
        #print(w_all[:30])
        print("one long traj:",len(w_all),len(trajs))

        print("Statistics")
        i = 0
        for flag in ACCFLAGS+REJFLAGS:
            j = np.sum(flags==flag)
            i += j
            print(flag,j)
        print("total flags",i)


        # EXTRA for zero_left
        #------------------------
        xi = 1.
        if fol == "000":
            ofile = "%s/pathensemble.txt" %(folder)
            pe = read_pathensemble(ofile)
            lmrs = pe.lmrs
            flags1 = pe.flags

            # verify
            for i,(flag0,flag1) in enumerate(zip(flags,flags1)):
                if flag0!=flag1: raise ValueError("traj",i,flag0,flag1)

            print_lmr_000(lmrs,weights)

            # reweigh with xi if possible
            xi = calc_xi(lmrs,weights)
            print("xi",xi)

            if xi !=1:
                weights_xi = np.array(weights,float)*xi
                # reconstruct
                w_all_xi = np.array([ weights_xi[i] for i in range(ncycle) for k in range(lens[i])])
            else:
                # just a copy
                weights_xi = weights
                w_all_xi = w_all

        # get concentration at interface and print        
        if fol == "000" or fol == "001":
            cut = interfaces[0]   # location of lambda0
            # dlambca_conc is the width of the bins
            nL = print_concentration_lambda0(ncycle,trajs,cut,dlambda_conc,dt,w_all,xi)

        # Creating histogram
        # if trajs were not to be flat yet...
        #trajs_flat = [item for subitem in trajs for item in subitem]
        #print(trajs_flat[:30])

        hist,edges = np.histogram(trajs,bins=bins,weights=w_all)
        centers = edges[:-1]+np.diff(edges)/2.

        # TODO ADAPTTTT HISTOGRAM
        hist = np.zeros(len(bins)-1)
        n_op = len(op_index)
        for i in range(n_op):
            if op_weight != 0:
                histi,edges = np.histogram(op.ops[:,op_index[i]],bins=bins,weights=w_all*op_weight[i])
                hist += histi


        plt.figure(1)
        plt.plot(centers,hist,marker='x',label=fol)  #label=r"%s"%(intf_names[i]))
        for c in interfaces:
            plt.plot([c,c],[0,max(hist)],"--",color='grey',label=None)
        if fol == "000":
            title = "ncycle=%i nL=%i"%(ncycle,nL)
            if xi != 1:
                plt.plot(centers,hist*xi,marker='x',label=fol+"-xi",color="k")
                title += " xi=%.2f"%xi
            plt.title(title)

        if ifol<2:
            plt.figure(2)
            plt.plot(centers,hist,marker='x',label=fol)  #label=r"%s"%(intf_names[i]))
            for c in interfaces:
                plt.plot([c,c],[0,max(hist)],"--",color='grey',label=None)
            if fol == "000":
                title = "ncycle=%i nL=%i"%(ncycle,nL)
                if xi != 1:
                    plt.plot(centers,hist*xi,marker='x',label=fol+"-xi",color="k")
                    title += " xi=%.2f"%xi
                plt.title(title)
 
    plt.figure(1)
    plt.legend()
    plt.xlabel("lambda (dlambda=%.3f)" %dlambda)
    plt.xlim(bins[0],bins[-1])   # always same bins anyways
    plt.tight_layout()
    plt.savefig("hist.png")

    plt.figure(2)
    plt.legend()
    plt.xlabel("lambda (dlambda=%.3f)" %dlambda)
    plt.xlim(bins[0],bins[-1])   # always same bins anyways
    plt.tight_layout()
    plt.savefig("hist.01.png")

#######################################################################################################


def make_histogram_op(figbasename,folder,interfaces,bins,skip=0,):
    """
    folder -- can be somedir/000, somedir/001, somedir/002, etc
    interfaces -- [lambda0], [lambda[0], [lambda0,lambda1], etc...
    """

    op,ep = get_data_ensemble_consistent(folder)

    weights, ncycle_true = get_weights(op.flags,ACCFLAGS,REJFLAGS)

    if skip > 0:
        figbasename += ".skip%i"%skip

    histogram_op(figbasename,interfaces,op.ops,op.lengths,weights,bins,skip=skip)


def histogram_op(figbasename,interfaces,data_op,length,weights,bins,skip=0):
    ntraj = len(length)
    assert len(weights) == len(length)

    index = 0   # column with the order parameter

    hist = np.zeros(len(bins)-1)
    for i in range(skip,ntraj):
        if weights[i] > 0:
            histi,_ = np.histogram(select_traj(data_op,length,i)[:,index],bins=bins)
            hist   += weights[i]*histi

    # now delete the first and last point
    hist2 = np.zeros(len(bins)-1)
    for i in range(skip,ntraj):
      if weights[i] > 0:
        histi,_ = np.histogram(select_traj(data_op,length,i)[1:-1,index],bins=bins)
        hist2   += weights[i]*histi

    # now delete short traj
    tooshort = 3
    hist3 = np.zeros(len(bins)-1)
    for i in range(skip,ntraj):
      if weights[i] > 0:
        if length[i] > tooshort:
            histi,_ = np.histogram(select_traj(data_op,length,i)[:,index],bins=bins)
            hist3   += weights[i]*histi


    bin_mids = 0.5*(bins[:-1]+bins[1:])
    #bin_mids = bin_edges[:-1]+np.diff(bin_edges)/2.
    plt.figure()
    plt.plot(bin_mids,hist,"x",label="w. st/end pts")
    plt.plot(bin_mids,hist2,"x",label="wo. st/end pts")
    plt.plot(bin_mids,hist3,"x",label="wo. short trajs")
    plt.xlabel("lambda")
    plt.ylabel("count")
    plt.title("bin width = %.3f" %(bins[1]-bins[0]))
    # plot line at lambda0
    lambda0 = interfaces[0]
    plt.plot([lambda0,lambda0],[0,max(hist)],color='grey',linewidth=2)
    plt.legend()
    plt.savefig(figbasename+".png")

    plt.figure()
    plt.plot(bin_mids,-np.log(hist),"x",label="w. st/end pts")
    plt.plot(bin_mids,-np.log(hist2),"x",label="wo. st/end pts")
    plt.plot(bin_mids,-np.log(hist3),"x",label="wo. short trajs")
    plt.xlabel("lambda")
    plt.ylabel("-ln(count)")
    plt.plot([lambda0,lambda0],[0,-np.log(max(hist))],color='grey',linewidth=2)
    plt.savefig(figbasename+".log.png")




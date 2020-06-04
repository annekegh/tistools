import numpy as np
from .reading import *
from .analyze_op import *

ACCFLAGS,REJFLAGS = set_flags_ACC_REJ() # Hard-coded rejection flags found in output files

def create_distrib(indir,folders,interfaces,dt,dlambda,lmin, lmax,dlambda_conc):
    """create figure of distributions of order parameter
    First figure: only path ensembles 000 and 001
    Second figure: all path ensembles
    """

    # bins: fix the lambda interval
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

    import matplotlib.pyplot as plt
    plt.figure(1)
    if len(folders)>2:
        plt.figure(2)

    for ifol,fol in enumerate(folders):
        print("-"*20)
        print("folder",fol)
        ofile = "%s/%s/order.txt" %(indir,fol)
        #ostart = 0
        ostart = -1    # TODO
        op = read_order(ofile,ostart)
        flags = op.flags
        trajs = op.longtraj
        lens = op.lengths
        ncycle = op.ncycle   # this is len(flags) = len(lengths)
        weights, ncycle_true = get_weights(flags,ACCFLAGS,REJFLAGS)

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
            ofile = "%s/%s/pathensemble.txt" %(indir,fol)
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

        print("Histogram")
        # if trajs were not to be flat yet...
        #trajs_flat = [item for subitem in trajs for item in subitem]
        #print(trajs_flat[:30])

        hist,edges = np.histogram(trajs,bins=bins,weights=w_all)
        centers = edges[:-1]+np.diff(edges)/2.

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



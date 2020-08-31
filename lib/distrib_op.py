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

def create_distrib(folders,interfaces_input,outputfile,do_pdf,
    dt,dlambda,dlambda_conc,lmin,lmax,ymin,ymax,offset,box,
    op_index, op_weight, do_abs,
    do_time, do_density):
    """create figure of distributions of order parameter
    First figure: only path ensembles 000 and 001
    Second figure: all path ensembles

    folders -- ["somedir/000", "somedir/001", "somedir/002", ...]
    interfaces -- [lambda0,lambda1,...,lambdaN]
                    if lambda_-1 exists: [lambda0,lambda1,...,lambdaN, lambda_-1]
    do_pdf  --  whether to save figures as pdf as well
    op_index -- list of selected order parameters, e.g. [0] or [0,2,3]
    op_weight -- weight for each of the selected order parameters
    do_abs -- wether to take the absolute value of the order parameters before histogramming
    """

    # for figure extensions
    extensions = ["png"]
    if do_pdf:
        extensions += ["pdf"]

    if offset != 0.:
        interfaces = [interf - offset for interf in interfaces_input]
    else:
        interfaces = interfaces_input
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

        if offset != 0.:
            trajs = op.longtraj - offset


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

            # compute and print factor xi
            xi = calc_xi(lmrs,weights)
            print("xi",xi)

            # reweigh with xi if possible --- this is not needed anymore
            #if xi !=1:
            #    weights_xi = np.array(weights,float)*xi
            #    # reconstruct
            #    w_all_xi = np.array([ weights_xi[i] for i in range(ncycle) for k in range(lens[i])])
            #else:
            #    # just a copy
            #    weights_xi = weights
            #    w_all_xi = w_all

        # get concentration at interface and print        
        if fol == "000" or fol == "001":
            cut = interfaces[0]   # location of lambda0
            # dlambca_conc is the width of the bins
            nL = print_concentration_lambda0(ncycle,trajs,cut,dlambda_conc,dt,w_all,xi)   


        # Computing histogram
        #---------------------
        # if trajs were not to be flat yet...
        #trajs_flat = [item for subitem in trajs for item in subitem]
        #print(trajs_flat[:30])

        # Previously:
        #hist,edges = np.histogram(trajs,bins=bins,weights=w_all)
        #centers = edges[:-1]+np.diff(edges)/2.

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

        hist = np.zeros(len(bins)-1)
        n_op = len(op_index)
        for i in range(n_op):
            if op_weight[i] != 0:
                if op_index[i] == 0: offset1 = offset
                else: offset1 = 0
                if box is None:
                  if do_abs:
                    histi,edges = np.histogram(abs(op.ops[:,op_index[i]]-offset1),bins=bins,weights=w_all*op_weight[i])
                  else:
                    histi,edges = np.histogram(op.ops[:,op_index[i]]-offset1,bins=bins,weights=w_all*op_weight[i])
                else:
                  # L = high - low
                  # d = pos-np.floor(pos/L)*L + low
                  low = box[0]
                  high = box[1]
                  L = high-low
                  if do_abs:
                    histi,edges = np.histogram(abs(
                            op.ops[:,op_index[i]]-offset1
                            - np.floor(op.ops[:,op_index[i]]/L-offset1/L)*L + low
                            ),
                            bins=bins,weights=w_all*op_weight[i])
                  else:
                    histi,edges = np.histogram(
                            op.ops[:,op_index[i]]-offset1
                            - np.floor(op.ops[:,op_index[i]]/L-offset1/L)*L + low
                            ,
                            bins=bins,weights=w_all*op_weight[i])
                hist += histi
        centers = edges[:-1]+np.diff(edges)/2.
        # dlambda = dlambda,edges[1]-edges[0]   I checked this already


        # Normalization?
        #---------------
        # Several options...
        # I chose "time in interval dlambda per path" and "prob_dens"
        if do_time:
            #hist3 = hist/ncycle   # time steps in interval dlambda per path

            #hist3 = hist/ncycle*dt   # time in interval dlambda per path in unit [time]

            hist3 = hist/ncycle*dt/dlambda   # time per dlambda per path in unit [time/length]

            hist = hist3

        elif do_density:
            hist3 = hist/ncycle*dt/dlambda / time_ens
            # time per lambda per path / time per path [1/length]
            # so this is the probability density in A' = rho_{A',ref}

            
            #unit = 1./dlambda    # per dlambda (per bin)
            #hist3 = hist*unit     # histogram in unit [1/length]


            #hist3 = hist2/np.sum(hist2)  # normalized histogram, per bin, then end points should be removed
            # np.sum(hist2) = 1

            #hist3 = hist2/np.sum(hist2)*unit  # normalized histogram in unit [1/length]
            # np.sum(hist2)*dlambda = 1

            hist = hist3


        # Creating histogram
        #---------------------

        for k in [1,2]:
          if k == 1 or ( len(folders)>2 and k==2 ):
            plt.figure(k)

            # plot vertical lines at interfaces
            # plot each time, because then I will reach the max of all histograms
            for c in interfaces:
                plt.plot([c,c],[0,max(hist)],"--",color='grey',label=None)

            # plot 000, do xi plot if present
            if fol == "000":
                plt.plot(centers,hist,marker='x',label=fol,color="green")  #label=r"%s"%(intf_names[i]))
                title = "ncycle=%i nL=%i"%(ncycle,nL)
                if xi != 1:
                    # the xi plot is not relevant for 'probability density'
                    if not do_density:
                        plt.plot(centers,hist*xi,marker='o',label=fol+"-xi",color="green",fillstyle='none')
                    # but always adapt the title, so I have xi printed
                    title += " xi=%.2f"%xi
                plt.title(title)

            # plot 001 in the same color as 000
            elif fol == "001":
                plt.plot(centers,hist,marker='x',label=fol,color="orange")  #label=r"%s"%(intf_names[i]))

            # plot other ensembles
            else:
                if k==1:
                    plt.plot(centers,hist,marker='x',label=fol)  #label=r"%s"%(intf_names[i]))

        if ifol == 1:  # I did 000 and 001
            plt.figure(2)
            plt.legend()
            plt.xlabel("lambda (dlambda=%.3f)" %dlambda)
            plt.xlim(bins[0],bins[-1])   # always same bins anyways
            plt.ylim(ymin,ymax)
            if do_time:
                plt.ylabel("time spent per length")
            elif do_density:
                plt.ylabel("prob. dens.")
            plt.tight_layout()

            # save figure for the desired file extensions (png, pdf)
            for ext in extensions:
                if do_time:
                    plt.savefig(outputfile+".time.01.%s"%ext)
                elif do_density:
                    plt.savefig(outputfile+".dens.01.%s"%ext)
                else:
                    plt.savefig(outputfile+".01.%s"%ext)

    plt.figure(1)
    plt.legend()
    plt.xlabel("lambda (dlambda=%.3f)" %dlambda)
    plt.xlim(bins[0],bins[-1])   # always same bins anyways
    plt.ylim(ymin,ymax)
    if do_time:
        plt.ylabel("time spent per length")
    elif do_density:
        plt.ylabel("prob. density")
    plt.tight_layout()

    # save figure for the desired file extensions (png, pdf)
    for ext in extensions:
        if do_time:
            plt.savefig(outputfile+".time.%s"%ext)
        elif do_density:
            plt.savefig(outputfile+".dens.%s"%ext)
        else:
            plt.savefig(outputfile+".%s"%ext)


def compute_time_in_ensemble(w_all,lens,dt):
    """compute time spent in ensemble

    remove first and last phase point of each path, which are not part of the ensemble
    I think that it also works when the path length is 1

    w_all -- array with weights of each order parameter path, for each stored phase point
    lens -- array/list? with length of each path
    dt -- time step"""
    w_all2 = np.copy(w_all)
    ncycle = len(lens)
    count = 0
    for i in range(ncycle):
         assert lens[i]>0
         w_all2[count] = 0
         w_all2[count+lens[i]-1]=0
         count += lens[i]
    time = np.sum(w_all2)*dt/ncycle
    print("time spent per path:",time)
    return time, w_all2



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




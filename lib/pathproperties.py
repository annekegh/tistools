import numpy as np
from .reading import *
from .analyze_op import *
import matplotlib.pyplot as plt

ACCFLAGS,REJFLAGS = set_flags_ACC_REJ() # Hard-coded rejection flags found in output files


# first crossing point distribution

def get_first_crossing_point(traj,lambdacross,lmr):

    #print(traj)
    #print(lmr,lambdacross)

    #if "M" in lmr:
    if lmr.startswith("L"):
        #crossed = (traj > interface)
        # detect when you get to the right of the interface
        #cross_index = np.argmax(traj>=lambdacross)
        # this is somewhat safer:
        a = np.where(traj>=lambdacross)[0]

    elif lmr.startswith("R"):
        # detect when you get to the left of the interface
        a = np.where(traj<=lambdacross)[0]

    if len(a) == 0:
            return -1,None
    else:
        cross_index = a[0]
        crossing = traj[cross_index]
        return cross_index,crossing


def get_first_crossing_distr(flags,weights,lens,w_all,ncycle,ncycle_true,trajs,lmrs,lambdacross): # check arguments
    assert len(lens) == ncycle   # the number of paths
    assert len(lmrs) == ncycle

    cross_indices = []

    count = 0
    for i in range(ncycle):
        assert lens[i] > 0
        traj = trajs[count:count+lens[i]]
        count += lens[i]
        if weights[i] > 0:
            #print(i,weights[i])
            cross_index, crossing = get_first_crossing_point(traj,lambdacross,lmrs[i])
            cross_indices.append(cross_index)
        else:
            cross_indices.append(-1)

    assert count == len(trajs)
    cross_indices = np.array(cross_indices)
    return cross_indices
    



# TODO doc

def create_distrib_first_crossing(folders,interfaces_input,outputfile,
          do_pdf,
          dt, offset):
    """create figure of distributions of order parameter
    First figure: only path ensembles 000 and 001
    Second figure: all path ensembles

    folders -- ["somedir/000", "somedir/001", "somedir/002", ...]
    interfaces -- [lambda0,lambda1,...,lambdaN]
                    if lambda_-1 exists: [lambda0,lambda1,...,lambdaN, lambda_-1]
    """

    # for figure extensions
    extensions = ["png"]
    if do_pdf:
        extensions += ["pdf"]

    if offset != 0.:
        interfaces = [interf - offset for interf in interfaces_input]
    else:
        interfaces = interfaces_input
    ##### bins = get_bins(folders,interfaces,dlambda,lmin,lmax) TODO

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
        op = parse_order_file(ofile,ostart)
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


        #-----------------------------------------------------------
        #-----------------------------------------------------------
        # did not change until here!!!! TODO
        #-----------------------------------------------------------
        #-----------------------------------------------------------


        # EXTRA, not only for zero_left, also other ensembles
        #----------------------------------------------------
        ofile = "%s/pathensemble.txt" %(folder)
        pe = read_pathensemble(ofile)
        lmrs = pe.lmrs
        flags1 = pe.flags

        # verify
   #     for i,(flag0,flag1) in enumerate(zip(flags,flags1)):
   # TODO put back         if flag0!=flag1: raise ValueError("traj",i,flag0,flag1)

        print_lmr(lmrs,weights)

        # compute and print factor xi
        xi = calc_xi(lmrs,weights)
        print("xi",xi)



#        # get concentration at interface and print 
#        #------------------------------------------       
#        if fol == "000" or fol == "001":
#            cut = interfaces[0]   # location of lambda0
#            # dlambca_conc is the width of the bins
#            nL = print_concentration_lambda0(ncycle,trajs,cut,dlambda_conc,dt,w_all,xi)   



        # Computing time spent in ensemble
        #---------------------

        # TODO
        # Optional:
        # remove first and last phase point of each path, which are not part of the ensemble
        # I think that it also works when the path length is 1
        # Also: compute time spent in the ensemble
#        do_remove_endpoints = False
#        if do_remove_endpoints:
#            # reconstruct w_all
#            time,w_all = compute_time_in_ensemble(w_all,lens,dt)
#        else:
#            # do not store w_all
#            time_ens, _ = compute_time_in_ensemble(w_all,lens,dt)

        # Computing distribution of first crossing point
        #-----------------------------------------------

        if ifol == 0:
            if len(interfaces) == len(folders):
                lambdacross = interfaces[0]
            elif len(interfaces) == len(folders)+1:
                lambdacross = (interfaces[0]+interfaces[-1])/2.
            else: issue
        elif ifol == 1:
            lambdacross = interfaces[0]
        else:
            lambdacross = interfaces[ifol-1]
        cross_indices = get_first_crossing_distr(flags,weights,lens,w_all,ncycle,ncycle_true,trajs,lmrs,lambdacross)


        # histogram
        bins = np.arange(-1,max(cross_indices)+2) - 0.5

        hist,edges = np.histogram(cross_indices,bins=bins,weights=weights)
        centers = edges[:-1]+np.diff(edges)/2.

        hist2,edges = np.histogram(lens,bins=bins,weights=weights)

        plt.figure(2)
        #plt.plot(centers,hist)
        plt.bar(centers,hist,label="cross")
        plt.plot(centers,hist2,label="path len+2",color='green')
        plt.xlim(xmin=-1)
        #plt.xlim(-5,20)
        plt.xlabel("phase pt crossing lambda_i=%.3f"%lambdacross)
        plt.ylabel("histogram")
        plt.title("%s, ncycle=%i, dt=%.3f"%(fol,ncycle,dt))
        plt.legend(loc='best')
        plt.savefig("crosshist.%s.png"%fol)
        plt.clf()

        if ifol != 0:
            # other histogram, normalized
            bins = np.arange(-1,51) - 0.5
            #bins = np.linspace(-1,51,53)-0.5  # This is a zoom TODO
            hist,edges = np.histogram(cross_indices,bins=bins,weights=weights)
            centers = edges[:-1]+np.diff(edges)/2.

            # normalize and shift
            plt.figure(1)
            plt.plot(centers,hist/float(np.sum(hist))+0.2*ifol,label=fol)

    # finalize figure
    plt.figure(1)
    plt.xlim(xmin=-1)
    plt.xlabel("phase pt crossing lambda_i")
    plt.ylabel("histogram")
    plt.legend()
    plt.title("normalized, ncycle=%i, dt=%.3f"%(ncycle,dt))
    plt.savefig("crosshist.all.png")



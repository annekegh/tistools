"""script to see order parameter in the ensembles
and to get the density at the interfaces

op  --  order parameter
a..  --  accepted

AG, March 2019
AG, May 26, 2020, adapted"""

import numpy as np

def read_pathdens_fast(filename):
    import pickle
    #fn = "pathdens_fast.pickle"
    print("Reading file...",fn)
    f = open(filename,"rb")
    pic = pickle.load(f,)
    f.close()
    return pic

def get_info_from_pic(pic,doprint=False):
    if doprint:
        print(pic[0].keys())
        print(pic[1].keys())
        print(pic[2].keys())

    interfaces = pic[2]['interfaces']
    intf_names = pic[2]['intf_names']
    if doprint:
        print("interfaces and interfaces names")
        print(pic[2]['interfaces'])
        print(pic[2]['intf_names'])

    folders = pic[2]['path']
    nfolders = len(folders)
    if doprint:
        print("folders")
        print(pic[2]['path'])
        print("nb of interfaces",nfolders)

    if nfolders < len(intf_names):
       if doprint:
           print("adapt intf_names")
       intf_names = intf_names[1:]

    ncycle = pic[2]['long_cycle'][1]
    if doprint:
        print("num_op = number of order parameters")
        print(pic[2]['num_op'])
        print(pic[2]['long_cycle'])
        print("ncycle",ncycle)
    
    
    if doprint:
        print("-"*10)
        print("op and weights")
    op = []
    weights = []
    for fol in folders:
        op0 = np.array(pic[0]['aop1',fol])
        w0 = np.array(pic[0]['astatw',fol])
        if doprint:
            print("weights, min/max of weights:")
            print(w0)
            print(min(w0),max(w0))
        op.append(op0)
        weights.append(w0)
    
    return interfaces,interf_names,folders,nfolders,ncycle,op,weights

"""script to investigate order parameter of POPC system
as a function of time

AG, Nov 26, 2019
AG, adapted May, 2020
"""

import numpy as np
import matplotlib.pyplot as plt
from functionsanalyzeop import *
from functionsop import *


ACCFLAGS,REJFLAGS = set_flags_ACC_REJ() # Hard-coded rejection flags found in output files



##############################################3

# GLOBAL
INOUT = False
#INOUT = True


#ens = "000"
#interfaces = [1.]

#ens = "001"
#interfaces = [1.,]

#ens = "002"
#interfaces = [1.,1.2,]

#ens = "003"
#interfaces = [1.,1.4,]

#ens = "004"
#interfaces = [1.,1.6,]

#ens = "005"
#interfaces = [1.,1.8]

simul = "s.16"
simul = "s.3"
simul = "s.1"
simul = "t"

FLAT1D = True
simul = "1dflat"

list_ensembles =["000","001","002","003","004","005"]
list_interfaces = [[1],[1.],[1,1.2],[1,1.4],[1,1.6],[1,1.8],]

list_ensembles =["000","001","002","003"]
list_interfaces = [[-2.5],[-2.5],[-2.5,-2.3],[-2.5,-2.1],]

# for test
list_ensembles =["000","001","002"]
list_interfaces = [[-0.1],[-0.1],[-0.1,0],]

decay_path(simul,list_ensembles)

for ens,interfaces in zip(list_ensembles,list_interfaces):
    make_histogram_op(simul,ens,interfaces,)  # skip=0

for ens,interfaces in zip(list_ensembles,list_interfaces):
#    for skip in [1000,2000,3000,4000,5000,6000,7000]:
#
        skip = None
        skip = 0
        make_histogram_op(simul,ens,interfaces,skip=skip,)
#
#        # TODO make this only the first trajs
#        #make_plot_trajs(simul,ens,interfaces)   # NO do not do this with many trajs

#for ens,interfaces in zip(list_ensembles,list_interfaces):
#    #for above in [0,20,50,]:
#    for above in [20]:
#        for skip in [0,1000]:
#            analyze_lengths(simul,ens,interfaces,above=above,skip=skip)



"""Script to read a simple histogram file and derive equilibrium constants

    filename = "hist.oxygen.txt"

AG, Nov 4, 2019
AG, May 26, 2020, adapted"""

import numpy as np

def read_histogram_file(filename):
    """read a simple histogram file
     with columns z / histogram / density / Free energy"""
    #filename = "hist.oxygen.txt"
    data = np.loadtxt(filename)

    print("Reading...",filename)
    print("data:",data.shape)
    # headers:
    # z hist dens F
    z = data[:,0]
    hist = data[:,1]
    dens = data[:,2]
    F = data[:,3]
    return z,hist,dens,F

def get_equil_constant_from_hist(z,hist,start=None):
    assert len(z) == len(hist)
    nbins = len(z)
    dz = (z[-1]-z[0])/(nbins-1)
    print("nbins",nbins)
    print("dz",dz)

    # find states:
    # where is Fmax located?
    F = -np.log(hist)
    i = np.argmax(F)
    print("max of F: bin %i, z %f, F %f" %(i,z[i],F[i]))

    if start is not None:
        i = start  # choose i manually

    # example: 
    # i1 = 19
    # i2 = 80
    # nbins = 100
    if i> nbins/2:
        i2 = i
        i1 = nbins-i-1
    else:
        i1 = i
        i2 = nbins-i+1

    Nw = np.sum(hist[:i1+1]) + np.sum(hist[i2:])
    Nm = np.sum(hist[i1+1:i2])
    Pw = Nw/(Nw+Nm)
    Pm = Nm/(Nw+Nm)
    K = Nm/Nw
    
    # now intensive quantities
    Lw = i1+1 + nbins-i2
    Lm = i2-i1-1
    assert Lw+Lm == nbins
    
    cw = Nw / Lw
    cm = Nm / Lm
    pw = cw/(cw+cm)
    pm = cm/(cw+cm)
    K2 = cm/cw
    
    print("Nw",Nw)
    print("Nm",Nm)
    print("Lw",Lw)
    print("Lm",Lm)
    print("Pw",Pw)
    print("Pm",Pm)
    print("K ",K, 1/K,)
    
    print("cw",cw)
    print("cm",cm)
    print("pw",pw)
    print("pm",pm)
    print("K2",K2, 1/K2)

if __name__ == "__main__":
    filename = "examples/hist.oxygen.txt"

    z,hist,dens,F = read_histogram_file(filename)
    print("---- use max of F")
    get_equil_constant_from_hist(z,hist,start=None)
    print("---- use bin 35")
    get_equil_constant_from_hist(z,hist,start=35)


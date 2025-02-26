# MOVE TO DEV
"""investigate

- double well
- flat potential
- flat potential with harmonic walls

You can execute this script directly:
python numerical.py
"""

import numpy as np
import matplotlib.pyplot as plt

def investigate_doublewell(a,b,c,kB,mass,temp,interfaces):
    """
    investigate double well potential
    a  -- is energy/length**4
    b  -- is energy/length**2
    c  -- is length
    """

    print("-"*10)
    print("double well")

    assert len(interfaces) >= 2
    h = interfaces[-1]-interfaces[0]

    dx = (interfaces[-1]-interfaces[0])/1e5   # this is a length
    x = np.arange(interfaces[0],interfaces[-1]+dx/10,dx)
    
    # The double well potential
    Vpot = a*x**4 - b*(x-c)**2
    Vpot -= Vpot[0]   #put reference
    #print(Vpot)

    plt.figure()
    plt.plot(x,Vpot)
    plt.savefig("Vpot.png")

    D = kB*temp/gamma/mass
    P1 = np.sum(np.exp(Vpot/temp/kB))*dx / D
    P = 1./P1
    K = np.sum(np.exp(-Vpot/temp/kB))/len(Vpot)
    #P = D*K/h

    print("interfaces")
    print(interfaces)
    print("K    ",K)
    print("h    ",h)
    print("temp ",temp)
    print("gamma",gamma)
    print("D    ",D)
    print("P analyt",P)
    print("-"*10)


def calc_analytical_conc(system):
    """calculate the concentration in the water phase
    using analytical formulat

    system -- presently: flat_periodic, flat_harmwall"""

    print("calculate analytical conc for %s"%system)

    if system == "flat_harmwall":
        H = 0.4    # part that is flat
        print("H:",H)
        # k is spring constant of harmonic wall
        for k in [0.1,1.,10,1000,10000,100000]:
            c = 1./(H+np.sqrt(2*np.pi/k))
            print("k",k,"c",c)

    elif system == "flat_periodic":
        H = 0.4    # size of the box
        print("H:",H)
        c = 1./H
        print("c",c)

    else:
        raise ValueError("system not known:"%system)

if __name__ == "__main__":
    calc_analytical_conc("flat_harmwall")
    calc_analytical_conc("flat_periodic")

if __name__ == "__main__":
    a = 1.  # in energy/length**4
    b = 2.  # in energy/length**2
    c = 0.  # in length

    kB = 1.
    mass = 1.
    gamma = 0.3
    temp = 0.07

    interfaces = [-0.9,0.,1.]    # in length
    investigate_doublewell(a,b,c,kB,mass,temp,interfaces)
    interfaces = [-1.0,0.,1.]
    investigate_doublewell(a,b,c,kB,mass,temp,interfaces)

    print("Switching to different units...")
    kB = 8.3144598  # J/mol/K
    kB = 8.3144598/1000.  # kJ/mol/K
    mass = 39.948
    temp = 300.

    interfaces = [-0.9,0.,1.]    # in length
    investigate_doublewell(a,b,c,kB,mass,temp,interfaces)
    interfaces = [-1.0,0.,1.]
    investigate_doublewell(a,b,c,kB,mass,temp,interfaces)



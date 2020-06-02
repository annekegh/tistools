import numpy as np

def calc_analytical_conc(system):

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



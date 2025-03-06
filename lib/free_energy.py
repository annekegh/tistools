"""Script to read a simple histogram file and derive equilibrium constants

    filename = "hist.oxygen.txt"

AG, Nov 4, 2019
AG, May 26, 2020, adapted
EW, Feb 26, 2025, add docs + improvements"""

import numpy as np

def read_histogram_file(filename):
    """
    Read a simple histogram file with columns z, histogram, density, and free energy.

    Parameters
    ----------
    filename : str
        Path to the histogram file.

    Returns
    -------
    tuple
        z : np.ndarray
            Array of z values.
        hist : np.ndarray
            Array of histogram values.
        dens : np.ndarray
            Array of density values.
        F : np.ndarray
            Array of free energy values.
    """
    # Load data from the file
    data = np.loadtxt(filename)
    print(f"Reading file: {filename}")
    print(f"Data shape: {data.shape}")

    # Extract columns from the data
    z = data[:, 0]
    hist = data[:, 1]
    dens = data[:, 2]
    F = data[:, 3]

    return z, hist, dens, F

def get_equil_constant_from_hist(z, hist, start=None):
    """
    Calculate equilibrium constants from histogram data.

    Parameters
    ----------
    z : np.ndarray
        Array of z values.
    hist : np.ndarray
        Array of histogram values.
    start : int, optional
        Starting bin index for the calculation. If None, the maximum of F is used.

    Returns
    -------
    None
    """
    assert len(z) == len(hist)
    nbins = len(z)
    dz = (z[-1] - z[0]) / (nbins - 1)
    print(f"Number of bins: {nbins}")
    print(f"Bin width (dz): {dz}")

    # Calculate free energy from histogram
    F = -np.log(hist)
    i = np.argmax(F)  # Find the bin with the maximum free energy
    print(f"Max of F: bin {i}, z {z[i]:.3f}, F {F[i]:.3f}")

    if start is not None:
        i = start  # Use the provided starting bin index

    # Determine the indices for the two states
    if i > nbins / 2:
        i2 = i
        i1 = nbins - i - 1
    else:
        i1 = i
        i2 = nbins - i + 1

    # Calculate the number of counts in each state
    Nw = np.sum(hist[:i1 + 1]) + np.sum(hist[i2:])
    Nm = np.sum(hist[i1 + 1:i2])
    Pw = Nw / (Nw + Nm)
    Pm = Nm / (Nw + Nm)
    K = Nm / Nw

    # Calculate the lengths of each state
    Lw = i1 + 1 + nbins - i2
    Lm = i2 - i1 - 1
    assert Lw + Lm == nbins

    # Calculate the concentrations and probabilities
    cw = Nw / Lw
    cm = Nm / Lm
    pw = cw / (cw + cm)
    pm = cm / (cw + cm)
    K2 = cm / cw

    # Print the results
    print(f"Nw: {Nw}")
    print(f"Nm: {Nm}")
    print(f"Lw: {Lw}")
    print(f"Lm: {Lm}")
    print(f"Pw: {Pw:.3f}")
    print(f"Pm: {Pm:.3f}")
    print(f"K: {K:.3f} (1/K: {1/K:.3f})")
    print(f"cw: {cw:.3f}")
    print(f"cm: {cm:.3f}")
    print(f"pw: {pw:.3f}")
    print(f"pm: {pm:.3f}")
    print(f"K2: {K2:.3f} (1/K2: {1/K2:.3f})")

if __name__ == "__main__":
    filename = "examples/hist.oxygen.txt"
    z, hist, dens, F = read_histogram_file(filename)
    print("---- Using max of F ----")
    get_equil_constant_from_hist(z, hist, start=None)
    print("---- Using bin 35 ----")
    get_equil_constant_from_hist(z, hist, start=35)
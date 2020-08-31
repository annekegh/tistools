"""Functions that write files"""


def write_histogram(fn,bin_midst,hist,F):
    with open(filename,"w+") as f:
        dens = hist/np.sum(hist)
        f.write('# {} {} {} {}\n'.format('z', 'hist', 'dens', 'F',))
        for i in range(len(hist)):
            f.write('{} {} {} {}\n'.format(bin_midst[i], hist[i], dens[i], F[i],))


# Functions for dealing with pyretis output

import numpy as np

def set_flags_ACC_REJ():
    """
    Define and return acceptance and rejection flags used in output files.

    The function categorizes flags into:
    - `REJFLAGS`: Hard-coded rejection reasons found in output files.
    - `ACCFLAGS`: Acceptance flags.

    Returns
    -------
    tuple
        ACCFLAGS (list): List of acceptance flags.
        REJFLAGS (list): List of rejection flags.

    Rejection Flags (REJFLAGS)
    --------------------------
    - 'MCR': Momenta change rejection.
    - 'BWI': Backward trajectory ended at wrong interface.
    - 'BTL': Backward trajectory too long (detailed balance condition).
    - 'BTX': Backward trajectory too long (max-path exceeded).
    - 'BTS': Backward trajectory too short.
    - 'KOB': Kicked outside of boundaries.
    - 'FTL': Forward trajectory too long (detailed balance condition).
    - 'FTX': Forward trajectory too long (max-path exceeded).
    - 'FTS': Forward trajectory too short.
    - 'NCR': No crossing with middle interface.
    - 'EWI': Initial path ends at wrong interface.
    - 'SWI': Initial path starts at wrong interface.
    - 'TSS': No valid indices to select for swapping.
    - 'TSA': Rejection due to the target swap acceptance criterion.
    - 'HAS': High acceptance swap rejection for SS/CS detailed balance.
    - 'CSA': Common sense super detailed balance rejection.
    - 'NSG': Path has no suitable segments.
    - 'SWD': PPTIS swap with incompatible propagation direction.
    - 'SWH': PPTIS swap problems after first extension.
    - 'ILL': Illegal state.
    - '0-L': Specific rejection flag added manually.

    Acceptance Flags (ACCFLAGS)
    ---------------------------
    - 'ACC': Generic acceptance flag.

    Raises
    ------
    ValueError
        If a flag is found in both ACCFLAGS and REJFLAGS.
    """
    # Define rejection flags based on various rejection conditions
    REJFLAGS = [
        'FTL', 'NCR', 'BWI', 'BTL', 'BTX', 'FTX', 'BTS', 'KOB', 'FTS',
        'EWI', 'SWI', 'MCR', 'TSS', 'TSA', 'HAS', 'CSA', 'NSG', 'SWD',
        'SWH', 'ILL', '0-L'
    ]

    # Define acceptance flags
    ACCFLAGS = ['ACC']

    # Ensure no flag is mistakenly in both ACCFLAGS and REJFLAGS
    conflicting_flags = set(ACCFLAGS) & set(REJFLAGS)
    if conflicting_flags:
        raise ValueError(f"Conflicting flags found in both acceptance and rejection lists: {conflicting_flags}")

    # Ensure '0-L' is included in at least one of the lists
    if '0-L' not in REJFLAGS and '0-L' not in ACCFLAGS:
        raise ValueError("'0-L' must be included in either ACCFLAGS or REJFLAGS.")

    return ACCFLAGS, REJFLAGS


class PathEnsemble(object):
    """
    A class to represent a path ensemble, which is a collection of paths generated
    during a path sampling simulation. The class stores information about each path,
    such as cycle numbers, path numbers, lengths, flags, and more.

    Attributes
    ----------
    cyclenumbers : np.ndarray
        Array of cycle numbers for each path.
    pathnumbers : np.ndarray
        Array of path numbers for each path.
    newpathnumbers : np.ndarray
        Array of new path numbers for each path.
    lmrs : np.ndarray
        Array of LMR (Left, Middle, Right) identifiers for each path.
    lengths : np.ndarray
        Array of lengths (in time steps) for each path.
    flags : np.ndarray
        Array of flags (e.g., 'ACC' for accepted, 'REJ' for rejected) for each path.
    generation : np.ndarray
        Array of generation types (e.g., 'ld' for load) for each path.
    lambmins : np.ndarray
        Array of minimum lambda values for each path.
    lambmaxs : np.ndarray
        Array of maximum lambda values for each path.
    dirs : np.ndarray
        Array of direction values for each path.
    istar_idx : np.ndarray
        Array of tuples representing indices for istar.
    ncycle : int
        Total number of cycles in the ensemble.
    totaltime : int
        Total time of all paths combined.
    weights : list
        List of weights for each path.
    shootlinks : np.ndarray
        Array of shoot links for each path.
    name : str
        Name of the path ensemble.
    interfaces : list
        List of interfaces, each represented as [L, M, R] and their string representation.
    has_zero_minus_one : bool
        Indicates if the ensemble has a lambda_-1 interface.
    in_zero_minus : bool
        Indicates if the ensemble is in the [0-] state.
    in_zero_plus : bool
        Indicates if the ensemble is in the [0+] or [0+-] state.
    orders : list
        List of orders for each path.
    """

    def __init__(self, data=None):
        """
        Initializes a :py:class:`.PathEnsemble` object.

        Parameters
        ----------
        data : list, optional
            A list of lines.split() from the pathensemble.txt file. If provided, the
            object is populated with data from the file.
        """
        if data is not None:
            # Populate attributes from data
            self.cyclenumbers = np.array([int(dat[0]) for dat in data])
            self.pathnumbers = np.array([int(dat[1]) for dat in data])
            self.newpathnumbers = np.array([int(dat[2]) for dat in data])
            self.lmrs = np.array(["".join(dat[3:6]) for dat in data])
            self.lengths = np.array([int(dat[6]) for dat in data])
            self.flags = np.array([dat[7] for dat in data])
            self.generation = np.array([dat[8] for dat in data])
            self.lambmins   = np.array([float(dat[9]) for dat in data])
            self.lambmaxs   = np.array([float(dat[10]) for dat in data])

            self.ncycle = len(self.lengths)
            self.totaltime = np.sum(self.lengths)

            self.weights = []
            self.shootlinks = np.full_like(self.cyclenumbers, None, dtype=object)
            self.name = ""
            self.interfaces = []  # [ [L, M, R], string([L,M,R]) ] 2 lists in a list

            self.has_zero_minus_one = False
            self.in_zero_minus = False
            self.in_zero_plus = False

            self.orders = None

    def set_name(self, name):
        """
        Sets the name of the path ensemble.

        Parameters
        ----------
        name : str
            The name to assign to the path ensemble.
        """
        self.name = name

    def set_weights(self, weights):
        """
        Sets the weights for each path in the ensemble.

        Parameters
        ----------
        weights : list
            A list of weights corresponding to each path.
        """
        self.weights = weights

    def set_interfaces(self, interfaces):
        """
        Sets the interfaces for the path ensemble.

        Parameters
        ----------
        interfaces : list
            A list of interfaces, each represented as [L, M, R] and their string representation.
        """
        self.interfaces = interfaces

    def set_zero_minus_one(self, has_zero_minus_one):
        """
        Sets whether the ensemble has a lambda_-1 interface.

        Parameters
        ----------
        has_zero_minus_one : bool
            True if the ensemble has a lambda_-1 interface, False otherwise.
        """
        self.has_zero_minus_one = has_zero_minus_one

    def set_in_zero_minus(self, in_zero_minus):
        """
        Sets whether the ensemble is in the [0-] state.

        Parameters
        ----------
        in_zero_minus : bool
            True if the ensemble is in the [0-] state, False otherwise.
        """
        self.in_zero_minus = in_zero_minus

    def set_in_zero_plus(self, in_zero_plus):
        """
        Sets whether the ensemble is in the [0+] or [0+-] state.

        Parameters
        ----------
        in_zero_plus : bool
            True if the ensemble is in the [0+] or [0+-] state, False otherwise.
        """
        self.in_zero_plus = in_zero_plus

    def update_shootlink(self, cycnum, link):
        """
        Updates the shoot link for a specific cycle number.

        Parameters
        ----------
        cycnum : int
            The cycle number to update.
        link : object
            The shoot link to assign to the cycle.
        """
        cycnumlist = (self.cyclenumbers).tolist()
        cyc_idx = cycnumlist.index(cycnum)
        self.shootlinks[cyc_idx] = link

    def get_shootlink(self, cycnum):
        """
        Retrieves the shoot link for a specific cycle number.

        Parameters
        ----------
        cycnum : int
            The cycle number to retrieve the shoot link for.

        Returns
        -------
        object
            The shoot link associated with the specified cycle number.
        """
        cycnumlist = (self.cyclenumbers).tolist()
        cyc_idx = cycnumlist.index(cycnum)
        return self.shootlinks[cyc_idx]

    def save_pe(self, fn):
        """
        Saves the path ensemble object to a file using pickle.

        Parameters
        ----------
        fn : str
            The filename to save the path ensemble to (without extension).
        """
        import pickle
        with open("pe_" + fn + ".pkl", 'wb') as g:
            pickle.dump(self, g, pickle.HIGHEST_PROTOCOL)

    def unify_pe(self):
        """
        Unify the path ensemble by replacing zero-weight paths with the previous
        non-zero weight path. This only works if high_acceptance = False.

        Returns
        -------
        :py:class:`.PathEnsemble`
            A new :py:class:`.PathEnsemble` object with unified paths.
        """
        new_pe = PathEnsemble()
        new_pe.cyclenumbers = self.cyclenumbers
        new_pe.pathnumbers = self.pathnumbers
        new_pe.newpathnumbers = self.newpathnumbers
        new_pe.weights = np.ones_like(self.weights)
        new_pe.lmrs = np.repeat(self.lmrs, self.weights)
        new_pe.lengths = np.repeat(self.lengths, self.weights)
        new_pe.flags = np.repeat(self.flags, self.weights)
        new_pe.generation = np.repeat(self.generation, self.weights)
        new_pe.lambmins = np.repeat(self.lambmins, self.weights)
        new_pe.lambmaxs = np.repeat(self.lambmaxs, self.weights)
        new_pe.interfaces = self.interfaces
        new_pe.name = self.name
        new_pe.has_zero_minus_one = self.has_zero_minus_one
        new_pe.in_zero_minus = self.in_zero_minus
        new_pe.in_zero_plus = self.in_zero_plus

        print("Are all weights 1? ", np.all(new_pe.weights == 1))
        print("Are all paths accepted? ", np.all(new_pe.flags == "ACC"))

        return new_pe

    def sample_pe(self, cycle_ids):
        """
        Samples the path ensemble by keeping only the specified cycle numbers.

        Parameters
        ----------
        cycle_ids : list or np.ndarray
            Indices of the cycles to keep in the sampled ensemble.

        Returns
        -------
        :py:class:`.PathEnsemble`
            A new :py:class:`.PathEnsemble` object containing only the specified cycles.
        """
        new_pe = PathEnsemble()
        new_pe.cyclenumbers = self.cyclenumbers[cycle_ids]
        new_pe.pathnumbers = self.pathnumbers[cycle_ids]
        new_pe.newpathnumbers = self.newpathnumbers[cycle_ids]
        new_pe.lmrs = self.lmrs[cycle_ids]
        new_pe.lengths = self.lengths[cycle_ids]
        new_pe.flags = self.flags[cycle_ids]
        new_pe.generation = self.generation[cycle_ids]
        new_pe.weights = self.weights[cycle_ids]
        new_pe.lambmins = self.lambmins[cycle_ids]
        new_pe.lambmaxs = self.lambmaxs[cycle_ids]
        new_pe.name = self.name
        new_pe.interfaces = self.interfaces
        new_pe.has_zero_minus_one = self.has_zero_minus_one
        new_pe.in_zero_minus = self.in_zero_minus
        new_pe.in_zero_plus = self.in_zero_plus
        return new_pe

    def bootstrap_pe(self, N, Bcycle, Acycle=0):
        """
        Bootstraps the path ensemble by sampling N elements within the cycle range [Acycle, Bcycle].

        Parameters
        ----------
        N : int
            Number of paths to sample.
        Bcycle : int
            Upper bound of the cycle range.
        Acycle : int, optional
            Lower bound of the cycle range. Default is 0.

        Returns
        -------
        :py:class:`.PathEnsemble`
            A new :py:class:`.PathEnsemble` object containing the bootstrapped paths.
        """
        import random

        # Get indices of cycles within the range [Acycle, Bcycle]
        idx, idx_w = [], []
        for i in range(len(self.cyclenumbers)):
            if self.cyclenumbers[i] >= Acycle and \
               self.cyclenumbers[i] <= Bcycle and \
               self.flags[i] == 'ACC' and \
               self.generation[i] != 'ld':
                idx.append(i)
                idx_w.append(self.weights[i])

        # Sample N indices from the list of indices, respecting the weights
        idx_sample = random.choices(idx, weights=idx_w, k=N)

        # Create a new path ensemble object
        pe_new = PathEnsemble()

        # Update the path ensemble object with the sampled indices
        pe_new.cyclenumbers = self.cyclenumbers[idx_sample]
        pe_new.pathnumbers = self.pathnumbers[idx_sample]
        pe_new.newpathnumbers = self.newpathnumbers[idx_sample]
        pe_new.lmrs = self.lmrs[idx_sample]
        pe_new.lengths = self.lengths[idx_sample]
        pe_new.flags = self.flags[idx_sample]
        pe_new.generation = self.generation[idx_sample]
        pe_new.weights = self.weights[idx_sample]
        pe_new.lambmins = self.lambmins[idx_sample]
        pe_new.lambmaxs = self.lambmaxs[idx_sample]
        pe_new.name = self.name
        pe_new.interfaces = self.interfaces
        pe_new.in_zero_minus = self.in_zero_minus
        pe_new.in_zero_plus = self.in_zero_plus
        pe_new.has_zero_minus_one = self.has_zero_minus_one

        return pe_new

    def set_orders(self, load=False, acc_only=True, save=False):
        """
        Sets the orders for the path ensemble by reading from a file or checking consistency.

        Parameters
        ----------
        load : bool, optional
            If True, loads orders from a file without checking consistency. Default is False.
        acc_only : bool, optional
            If True, only checks consistency for accepted paths. Default is True.
        save : bool, optional
            If True, saves the orders to a file. Default is False.

        Raises
        ------
        ValueError
            If there is a mismatch in accepted paths when `acc_only` is True.
        """
        print("Setting orders for path ensemble", self.name)
        orders, cyclenumbers, lengths, flags, generation =\
            load_order_parameters(self.name + "/order.txt", load=load, acc_only=acc_only)

        if load:
            print("Loaded orders from file, NOT CHECKING FOR CONSISTENCY.")
            self.orders = orders
            return

        check_cycnum = (self.cyclenumbers == cyclenumbers).all()
        check_length = (self.lengths == lengths).all()
        check_flag = (self.flags == flags).all()
        check_generation = (self.generation == generation).all()

        if not acc_only:
            assert check_cycnum, "cyclenumbers do not match"
            assert check_length, "lengths do not match"
            assert check_flag, "flags do not match"
            assert check_generation, "generations do not match"
            print("Everything matches, setting orders.")
            self.orders = orders
            if save:
                print(f"Saving orders to {self.name}/order.npy")
                np.save(self.name + "/order.npy", np.array(orders, dtype=object),
                        allow_pickle=True)
            return

        if check_cycnum and check_length and check_flag and check_generation:
            print("Everything matches, setting orders.")
            self.orders = orders
            if save:
                print(f"Saving orders to {self.name}/order.npy")
                np.save(self.name + "/order.npy", np.array(orders, dtype=object),
                        allow_pickle=True)
            return

        # Check for mismatches in rejected paths only
        for el, pe_el, typ in zip([cyclenumbers, lengths, flags, generation],
                                  [self.cyclenumbers, self.lengths, self.flags,
                                   self.generation],
                                  ["cyclenumbers", "lengths", "flags",
                                   "generation"]):
            idx_mismatch = np.where(el != pe_el)[0]
            if len(idx_mismatch) == 0:
                continue
            if (flags[idx_mismatch] != "REJ").all():
                print(f"Mismatch in REJ paths for {typ}, ignoring.")
                continue
            else:
                msg = f"Mismatch in {typ} for accepted paths.\n"
                msg += f"Mismatching ids: {idx_mismatch}"
                raise ValueError(msg)

        print("Everything matched for the ACC paths, setting orders.")
        self.orders = orders
        if save:
            print(f"Saving orders to {self.name}/order.npy")
            np.save(self.name + "/order.npy", np.array(orders, dtype=object),
                    allow_pickle=True)

class OrderParameter(object):
    """
    A class to represent order parameters associated with a path ensemble.

    This class stores and manages order parameters for each path in a trajectory, ensuring
    consistency between the input arrays and providing functionality to extract specific
    trajectory segments.

    Attributes
    ----------
    cyclenumbers : np.ndarray
        Array of cycle numbers for each path.
    lengths : np.ndarray
        Array of lengths (in time steps) for each path.
    flags : np.ndarray
        Array of flags (e.g., 'ACC' for accepted, 'REJ' for rejected) for each path.
    generation : np.ndarray
        Array of generation types (e.g., 'ld' for load) for each path.
    longtraj : np.ndarray
        Array representing the long trajectory data.
    ops : np.ndarray
        Array of order parameters for each time step in the trajectory.
    ncycle : int
        Total number of cycles in the ensemble.
    totaltime : int
        Total time of all paths combined.
    """

    def __init__(self, cyclenumbers, lengths, flags, generation, longtraj, data):
        """
        Initializes an OrderParameter object.

        Parameters
        ----------
        cyclenumbers : list or np.ndarray
            Array of cycle numbers for each path.
        lengths : list or np.ndarray
            Array of lengths (in time steps) for each path.
        flags : list or np.ndarray
            Array of flags (e.g., 'ACC' for accepted, 'REJ' for rejected) for each path.
        generation : list or np.ndarray
            Array of generation types (e.g., 'ld' for load) for each path.
        longtraj : list or np.ndarray
            Array representing the long trajectory data.
        data : np.ndarray
            Array of order parameters for each time step in the trajectory.

        Notes
        -----
        The constructor ensures that the input arrays are consistent in size and that the
        total length of the trajectory data matches the sum of individual path lengths.
        """
        # Ensure input arrays are consistent
        assert_consistent(cyclenumbers, lengths, flags, generation, longtraj, data)

        # Initialize attributes
        self.cyclenumbers = np.array(cyclenumbers)
        self.lengths = np.array(lengths)
        self.flags = np.array(flags)
        self.generation = np.array(generation)
        self.longtraj = np.array(longtraj)
        self.ops = data  # Order parameters for each time step
        self.ncycle = len(self.lengths)  # Total number of cycles
        self.totaltime = np.sum(self.lengths)  # Total time of all paths combined


def assert_consistent(cyclenumbers, lengths, flags, generation, longtraj, data):
    """
    Asserts that the input arrays are consistent with each other in terms of their lengths
    and the total length of the trajectory data.

    Parameters
    ----------
    cyclenumbers : list or np.ndarray
        Array of cycle numbers for each path.
    lengths : list or np.ndarray
        Array of lengths (in time steps) for each path.
    flags : list or np.ndarray
        Array of flags (e.g., 'ACC' for accepted, 'REJ' for rejected) for each path.
    generation : list or np.ndarray
        Array of generation types (e.g., 'ld' for load) for each path.
    longtraj : list or np.ndarray
        Array representing the long trajectory data.
    data : list or np.ndarray
        Array representing the raw trajectory data.

    Raises
    ------
    ValueError
        If the lengths of `cyclenumbers`, `lengths`, `flags`, or `generation` do not match.
    ValueError
        If the total length of `lengths` does not match the length of `data` or `longtraj`.

    Notes
    -----
    This function ensures that the input arrays are consistent in size and that the total
    length of the trajectory data matches the sum of individual path lengths.
    """
    n = len(cyclenumbers)

    # Check if all input arrays have the same length
    if not (n == len(lengths) == len(flags) == len(generation)):
        raise ValueError(
            "Input arrays (cyclenumbers, lengths, flags, generation) must have the same length."
        )

    # Check if the total length of paths matches the length of data and longtraj
    total_length = np.sum(lengths)
    if not (total_length == len(data) == len(longtraj)):
        raise ValueError(
            "The sum of paths' `lengths` must match the length of `data` and `longtraj`."
        )

def read_pathensemble(fn, ostart=0):
    """
    Reads a pathensemble.txt file and returns a :py:class:`.PathEnsemble` object.

    Parameters
    ----------
    fn : str
        The filename of the pathensemble.txt file.
    ostart : int, optional
        The starting cycle number to read from (default is 0).

    Returns
    -------
    :py:class:`.PathEnsemble`
        An object containing the data from the pathensemble.txt file.

    Notes
    -----
    The format of the pathensemble.txt file is expected to be:
         0          1          0 L M L     383 ACC ld  9.925911427e-01  1.490895033e+00       0     191  0.000000000e+00       0       0  1.000000000e+00
         1          2          1 L M L     250 ACC sh  9.794998169e-01  1.581833124e+00       0     137  1.284500122e+00     243     174  1.000000000e+00
    """
    data = []
    with open(fn, "r") as f:
        for line in f:
            words = line.split()
            cycle = int(words[0])
            if cycle >= ostart:
                data.append(words)
                # Extracting specific data points from the line
                # start_middle_end = "".join(words[3:6])  # Combine chars to a string: "L M R"
                # length = int(words[6])  # Get length
                # status_flag = words[7]  # Get status/flag/acceptance: "ACC"
                # generation = words[8]  # Get generation

    pe = PathEnsemble(data)

    try:
        pe_name = fn.split("/")[-1]  # Extract filename from path
    except IndexError:
        pe_name = fn
    finally:
        pe.set_name(pe_name)

    return pe

def strip_endpoints(order_list):
    """
    Removes the first and last elements from each sublist in a list of order parameter trajectories.

    This function is useful for trimming the endpoints of the order parameter list of a certain trajectory.

    Parameters
    ----------
    order_list : list of list
        A list of order parameter trajectories, where each trajectory is represented as a list.

    Returns
    -------
    list of list
        A new list of order parameter trajectories with the first and last elements removed from
        each sublist.

    Example
    -------
    >>> order_list = [[1, 2, 3, 4], [5, 6, 7, 8]]
    >>> strip_endpoints(order_list)
    [[2, 3], [6, 7]]
    """
    stripped_order_list = []
    for orders in order_list:
        stripped_order_list.append(orders[1:-1])  # Remove first and last elements
    return stripped_order_list


def get_flat_list_and_weights(orders, weights):
    """
    Flattens a list of order parameter trajectories and generates corresponding weights.

    This function combines multiple order parameter trajectories into a single flat list and
    creates a corresponding list of weights, where each weight is repeated according to the
    length of its associated trajectory.

    Parameters
    ----------
    orders : list of list
        A list of order parameter trajectories, where each trajectory is represented as a list.
    weights : list of float
        A list of weights corresponding to each trajectory in `orders`.

    Returns
    -------
    all_orders : np.ndarray
        A flattened array containing all order parameters from all trajectories.
    all_ws : list of float
        A list of weights, where each weight is repeated according to the length of its
        associated trajectory.

    Example
    -------
    >>> orders = [[1, 2], [3, 4, 5]]
    >>> weights = [0.5, 0.8]
    >>> get_flat_list_and_weights(orders, weights)
    (array([1, 2, 3, 4, 5]), [0.5, 0.5, 0.8, 0.8, 0.8])
    """
    # Flatten the list of order parameter trajectories
    all_orders = np.array([item for sublist in orders for item in sublist])

    # Generate weights corresponding to each order parameter
    all_ws = []
    for w, o in zip(weights, orders):
        all_ws += [w] * len(o)  # Repeat weight for each element in the trajectory

    return all_orders, all_ws


# READ ORDERS AND MAKE ORDERPARAMETER OBJECT
def parse_order_file(fn, ostart=0):
    """
    Reads an order.txt file and returns an OrderParameter object.

    This function processes the order.txt file, extracting cycle numbers, flags, generation types,
    and order parameters for each trajectory. It ensures consistency in the data and handles 
    trajectories with zero length appropriately. The function also supports reading from a specified 
    starting cycle number.

    Parameters
    ----------
    fn : str
        The filename of the order.txt file.
    ostart : int, optional
        The starting cycle number to read from (default is 0).

    Returns
    -------
    OrderParameter
        An object containing the data from the order.txt file, including cycle numbers, flags,
        generation types, and order parameters.

    Raises
    ------
    ValueError
        If the first line of the file does not start with "# Cycle:".

    Notes
    -----
    The format of the order.txt file is expected to be:
    # Cycle: 0, status: ACC, move: ('ld', 0, 0, 0)
    #     Time       Orderp
         0     0.992591     0.751421     0.122551     1.440613     0.235256     0.625671
         1     1.002283     0.737442     0.095843     1.420257     0.252691     0.565861
    ...

    The function reads the file line by line, extracting relevant data and handling trajectories
    with zero length by issuing warnings and adjusting the data accordingly. It also verifies the
    consistency of the extracted data before creating the OrderParameter object.
    """
    cyclenumbers = []
    lengths = []
    flags = []
    generation = []
    longtraj = []
    data = []

    ntraj = 0
    ntraj_started = 0
    last_length = 0

    # Check first line
    with open(fn, "r") as f:
        line = f.readline()
        if not line.startswith("# Cycle:"):
            raise ValueError(f"First line of {fn} does not start with `# Cycle:`.")

    with open(fn, "r") as f:
        for i, line in enumerate(f):
            # Skip lines until reaching the starting cycle number
            if i < ostart:
                continue

            # Header time
            if line.startswith("#     Time"):
                continue

            # Header Cycle
            elif line.startswith("# Cycle:"):
                if ntraj_started > 0:  # Not the very first traj
                    if last_length > 0:
                        # Successfully update the previous one
                        ntraj += 1
                        lengths[-1] = last_length
                    elif last_length == 0:
                        print("WARNING" * 30)
                        print("Encountered traj with length 0 at cyclenumber", line)
                        # Undo a few things
                        cyclenumbers.pop()
                        flags.pop()
                        generation.pop()
                        ntraj_started -= 1  # Previous was a false alarm

                # Reset for new one
                ntraj_started += 1
                last_length = 0

                # Extract the time, cyclenumber, flag, generation
                words = line.split()
                cyclenumbers.append(int(words[2][:-1]))  # Remove the last character, which is a comma
                flags.append(words[4][:-1])  # Remove the last character, which is a comma: ACC,
                generation.append(words[6][2:4])  # Remove characters: ('sh',
                lengths.append(0)  # Length to be updated

            # Collect order parameter of traj
            else:
                words = line.split()
                assert int(words[0]) == last_length  # Ensure the time step matches the expected length
                last_length += 1  # Update traj length

                # Collect the order parameters
                longtraj.append(float(words[1]))
                if len(words[1:]) > 1:
                    data.append([float(word) for word in words[1:]])  # Skip the time
                else:
                    data.append(float(words[1]))  # This is a copy of longtraj

    # Finish the last trajectory when done with reading
    if last_length > 0:
        ntraj += 1
        lengths[-1] = last_length
    elif ntraj_started == ntraj + 1:
        # Undo a few things
        cyclenumbers.pop()
        flags.pop()
        generation.pop()
        lengths.pop()

    data = np.array(data)
    if len(data.shape) == 1:
        data = data.reshape((len(data), 1))

    # Verify lengths
    assert_consistent(cyclenumbers, lengths, flags, generation, longtraj, data)
    op = OrderParameter(cyclenumbers, lengths, flags, generation, longtraj, data)
    return op

# READ ORDER AND RETURN LIST OF ORDERPARAMETERS TO FEED INTO PATH ENSEMBLE
def load_order_parameters(fn, load=False, acc_only=True):
    """
    Reads an order.txt file and returns the order parameter trajectories as a list of numpy arrays.

    This function processes the order.txt file, extracting cycle numbers, path lengths, flags, 
    and generation types for each trajectory. It ensures consistency in the data and handles 
    trajectories with zero length appropriately. The function also supports loading from a .npy 
    file and filtering for accepted trajectories only.

    Parameters
    ----------
    fn : str
        The filename of the order.txt file.
    load : bool, optional
        If True, load the order parameter trajectories from a .npy file. The default is False.
    acc_only : bool, optional
        If True, only load the accepted trajectories. Rejected trajectories are given empty arrays. 
        The default is True.

    Returns
    -------
    tuple
        A tuple containing:
        - subdata_list (list of np.ndarray): A list of numpy arrays containing the order parameter trajectories.
        - cyclenumbers (np.ndarray): Array of cycle numbers.
        - lengths (np.ndarray): Array of path lengths.
        - flags (np.ndarray): Array of flags indicating the status of each trajectory.
        - generations (np.ndarray): Array of generation types.

    Raises
    ------
    ValueError
        If the first line of the file does not start with "# Cycle:".

    Notes
    -----
    The format of the order.txt file is expected to be:
    # Cycle: 0, status: ACC, move: ('ld', 0, 0, 0)
    #     Time       Orderp
             0     0.992591     0.751421     0.122551
             1     1.002283     0.737442     0.095843
    ...
    # Cycle: 1, status: ACC, move: ('sh', 0, 0, 0)
    #     Time       Orderp
             0     0.992591     0.751421     0.122551
             1     1.002283     0.737442     0.095843
    """
    if load:
        return np.load(fn.replace(".txt", ".npy"), allow_pickle=True), None, None, None, None

    # Initialize lists to store data
    cyclenumbers, lengths, flags, generations = [], [], ["ACC"], []
    subdata_list, subdata = [], []
    ntraj, ntraj_started, last_length = 0, 0, 0

    # Check first line
    with open(fn, "r+") as f:
        if not f.readline().startswith("# Cycle:"):
            raise ValueError(f"First line of {fn} does not start with `# Cycle:`.")

    with open(fn, 'r') as f:
        for i, line in enumerate(f):
            # Header time
            if line.startswith("#     Time"):
                continue

            # Header Cycle
            elif line.startswith("# Cycle:"):
                # Append the previous trajectory, and then reset
                if not acc_only or (acc_only and flags[-1] == "ACC"):
                    subdata_list.append(np.array(subdata))
                else:
                    subdata_list.append(np.array([]))
                subdata = []
                if ntraj_started > 0:  # Not the very first traj
                    if last_length > 0:
                        # Successfully update the previous one
                        ntraj += 1
                        lengths[-1] = last_length
                    elif last_length == 0:
                        msg = "WARNING " * 3
                        msg += f"Traj with len 0 at cyclenumber {line}\n"
                        msg += "We do not load this one.\n"
                        msg += "If this happens, something is VERY wrong."
                        print(msg)

                        # Undo a few things
                        cyclenumbers.pop()
                        flags.pop()
                        generations.pop()
                        ntraj_started -= 1  # Previous was a false alarm
                else:
                    # Very first traj, we remove our fake "ACC" flag 
                    flags = []

                # Reset for new one
                ntraj_started += 1
                last_length = 0

                # Extract the time, cyclenumber, flag, generation
                words = line.split()
                cyclenumbers.append(int(words[2][:-1]))  # Remove the last character, which is a comma
                flags.append(words[4][:-1])  # Remove the last character, which is a comma: ACC,
                generations.append(words[6][2:4])  # Remove characters: ('sh',
                lengths.append(0)  # Length to be updated

            # Collect order parameter of traj
            else:
                words = line.split()
                assert int(words[0]) == last_length  # Ensure the time step matches the expected length
                last_length += 1  # Update traj length

                # Collect all order parameters: words[1:]
                subdata.append([float(word) for word in words[1:]])

    # Finish the last trajectory when done with reading
    if last_length > 0:
        ntraj += 1
        lengths[-1] = last_length
    elif ntraj_started == ntraj + 1:
        # Undo a few things
        cyclenumbers.pop()
        flags.pop()
        generations.pop()
        lengths.pop()

    subdata_list.append(np.array(subdata))
    subdata_list = subdata_list[1:]

    generations = np.array(generations)
    cyclenumbers = np.array(cyclenumbers)
    lengths = np.array(lengths)
    flags = np.array(flags)

    return subdata_list, cyclenumbers, lengths, flags, generations

def get_weights(flags, ACCFLAGS, REJFLAGS, verbose=True):
    """
    Calculates the weights of each trajectory based on acceptance and rejection flags.

    This function processes a list of flags indicating the status of each trajectory and 
    calculates the weights for accepted trajectories. Rejected trajectories increase the 
    weight of the previous accepted trajectory. The function also provides a summary of 
    the number of accepted, rejected, and omitted trajectories.

    Parameters
    ----------
    flags : list of str
        A list of flags indicating the status of each trajectory.
    ACCFLAGS : list of str
        A list of flags that indicate accepted trajectories.
    REJFLAGS : list of str
        A list of flags that indicate rejected trajectories.
    verbose : bool, optional
        If True, prints a summary of the weights and the number of accepted, rejected, 
        and omitted trajectories. The default is True.

    Returns
    -------
    tuple
        A tuple containing:
        - weights (np.ndarray): An array with the weight of each trajectory, 0 if not accepted.
        - ncycle_true (int): The sum of weights, representing the total number of cycles.

    Raises
    ------
    ValueError
        If the first flag is not 'ACC'.

    Example
    -------
    >>> flags = ['ACC', 'REJ', 'ACC', 'REJ', 'REJ', 'ACC']
    >>> ACCFLAGS = ['ACC']
    >>> REJFLAGS = ['REJ']
    >>> weights, ncycle_true = get_weights(flags, ACCFLAGS, REJFLAGS, verbose=True)
    weights:
    accepted      3
    rejected      3
    omitted       0
    total trajs   6
    total weights 6
    >>> print(weights)
    [1 0 2 0 0 3]
    >>> print(ncycle_true)
    6
    """
    ntraj = len(flags)
    weights = np.zeros(ntraj, int)

    accepted = 0
    rejected = 0
    omitted = 0

    acc_w = 0
    acc_index = 0
    tot_w = 0

    if flags[0] != 'ACC':
        raise ValueError("The first flag must be 'ACC'.")

    for i, flag in enumerate(flags):
        if flag in ACCFLAGS:
            # Store previous trajectory with accumulated weight
            weights[acc_index] = acc_w
            tot_w += acc_w
            # Info for new trajectory
            acc_index = i
            acc_w = 1
            accepted += 1
        elif flag in REJFLAGS:
            acc_w += 1  # Increase weight of previous accepted trajectory
            rejected += 1
        else:
            omitted += 1

    # Store the last accepted path with its weight
    weights[acc_index] = acc_w
    tot_w += acc_w

    if verbose:
        print("weights:")
        print("accepted     ", accepted)
        print("rejected     ", rejected)
        print("omitted      ", omitted)
        print("total trajs  ", ntraj)
        print("total weights", np.sum(weights))

    if omitted > 0 and verbose:
        print(f"Warning: There are {omitted} omitted trajectories. Please check if this is intentional.")

    ncycle_true = np.sum(weights)
    miss = len(flags) - 1 - ncycle_true
    for i in range(miss):
        if flags[-(i + 1)] not in REJFLAGS:
            raise ValueError("Unexpected flag at the end of the list.")

    return weights, ncycle_true

def get_data_ensemble_consistent(folder):
    """
    Reads and verifies the consistency of order and path ensemble data from a specified folder.

    This function reads the order.txt and pathensemble.txt files from the given folder, 
    compares their cycle numbers, total times, lengths, flags, and generation types to 
    ensure consistency. If inconsistencies are found, appropriate actions are taken to 
    correct them or raise errors.

    Parameters
    ----------
    folder : str
        The folder containing the order.txt and pathensemble.txt files. 
        Example: 'somedir/000', 'somedir/001', 'somedir/002', etc.

    Returns
    -------
    tuple
        A tuple containing:
        - op (OrderParameter): The OrderParameter object from the order.txt file.
        - pe (:py:class:`.PathEnsemble`): The :py:class:`.PathEnsemble` object from the pathensemble.txt file.

    Raises
    ------
    ValueError
        If there are inconsistencies in the lengths, flags, or generation types between 
        the order and path ensemble data.

    Example
    -------
    >>> op, pe = get_data_ensemble_consistent('somedir/000')
    >>> print(op.ncycle, pe.ncycle)
    """
    fn_op = f"{folder}/order.txt"
    fn_path = f"{folder}/pathensemble.txt"

    # READ
    op = parse_order_file(fn_op)
    pe = read_pathensemble(fn_path)

    print("Reading...")
    print("cycle_op  ", op.ncycle)
    print("cycle_path", pe.ncycle)
    print("total time op  ", op.totaltime)
    print("total time path", pe.totaltime)

    if op.totaltime < pe.totaltime:
        print("fix lengths...")
        data_path = data_path[:-1]
        lengths = lengths[:-1]
        flags = flags[:-1]
        generation = generation[:-1]
        print("data_op  ", op.shape)
        print("data_path", len(data_path))
        print("total time op  ", len(op))
        print("total time path", np.sum(lengths))
    elif op.totaltime > pe.totaltime:
        raise ValueError("Total time in order.txt is greater than in pathensemble.txt.")
    
    if op.totaltime != pe.totaltime:
        raise ValueError("Total times in order.txt and pathensemble.txt do not match.")

    # Matching
    if op.ncycle != pe.ncycle:
        raise ValueError("Number of cycles in order.txt and pathensemble.txt do not match.")

    if not all(l1 == l2 for l1, l2 in zip(op.lengths, pe.lengths)):
        raise ValueError("Lengths in order.txt and pathensemble.txt do not match.")

    if not all(f1 == f2 for f1, f2 in zip(op.flags, pe.flags)):
        raise ValueError("Flags in order.txt and pathensemble.txt do not match.")

    if not all(g1 == g2 for g1, g2 in zip(op.generation, pe.generation)):
        raise ValueError("Generation types in order.txt and pathensemble.txt do not match.")

    return op, pe

# reading the RESTART file
# the pyretis.restart file is a pickle object

def read_restart_file(filename):
    """
    Reads restart information for a simulation from a specified file.

    This function reads a restart file, typically named "pyretis.restart", 
    using the pickle module to load the simulation state information.

    Parameters
    ----------
    filename : str
        The name of the file to read from. Example: "pyretis.restart".

    Returns
    -------
    dict
        A dictionary containing the restart information for the simulation.

    Raises
    ------
    ValueError
        If the file cannot be read or the content is not a valid pickle object.
    """
    import pickle

    try:
        with open(filename, 'rb') as infile:
            info = pickle.load(infile)
    except (FileNotFoundError, pickle.UnpicklingError) as e:
        raise ValueError(f"Error reading the restart file: {e}")

    return info


#########################################
# Other functions


def select_traj(data_op, lengths, i):
    """
    Selects a specific trajectory segment from the order parameter data.

    This function extracts a segment of the order parameter data corresponding to the 
    i-th trajectory based on the provided lengths.

    Parameters
    ----------
    data_op : np.ndarray
        The order parameter data, typically the ops attribute of an OrderParameter object.
    lengths : list or np.ndarray
        A list or array of lengths (in time steps) for each trajectory.
    i : int
        The index of the trajectory to select.

    Returns
    -------
    np.ndarray
        A numpy array containing the selected trajectory segment.

    Example
    -------
    >>> data_op = np.random.rand(100, 5)
    >>> lengths = [20, 30, 50]
    >>> i = 1
    >>> selected_traj = select_traj(data_op, lengths, i)
    >>> print(selected_traj.shape)
    (30, 5)
    """
    if i == 0:
        previous_length = 0
    else:
        previous_length = np.sum(lengths[:i])
    this_length = lengths[i]

    # Ensure data_op has more than one dimension
    if len(data_op.shape) <= 1:
        raise ValueError("data_op must have more than one dimension.")

    data_sel = np.zeros((this_length, data_op.shape[1]))
    data_sel[:, :] = data_op[previous_length:previous_length + this_length, :]
    return data_sel

def get_ntraj(accepted, acc=True):
    """
    Determines the number of trajectories and their indices based on acceptance criteria.

    This function calculates the number of trajectories and their indices based on whether 
    they are accepted or not.

    Parameters
    ----------
    accepted : list or np.ndarray
        A list or array indicating whether each trajectory is accepted (True) or rejected (False).
    acc : bool, optional
        If True, only accepted trajectories are considered. If False, all trajectories are considered.
        The default is True.

    Returns
    -------
    tuple
        A tuple containing:
        - ntraj (int): The number of trajectories.
        - indices (list or np.ndarray): The indices of the trajectories.

    Example
    -------
    >>> accepted = [True, False, True, True]
    >>> ntraj, indices = get_ntraj(accepted, acc=True)
    >>> print(ntraj)
    3
    >>> print(indices)
    [0, 2, 3]
    """
    if not acc:
        # OPTION: do all
        ntraj = len(accepted)
        indices = np.arange(ntraj)
    else:
        # OTHER OPTION
        ntraj = np.sum(accepted)
        indices = [i for i in range(len(accepted)) if accepted[i]]
    return ntraj, indices


def read_inputfile(filename):
    """read interfaces and timestep from inputfile

    Add the zero_left interface to the end of the list, if this
    is present."""

    # interfaces
    with open(filename,"r") as f:
        for line in f:
            if "interfaces" in line and "=" in line and not line.strip().startswith("#"):
                parts = line.split("[")
                parts2 = parts[1].split("]")
                words = parts2[0].split(",")
                interfaces = [float(word) for word in words]
                break
    #st = line.find("[",)  # beg=0, end=len(string))
    #end = line.find("]")
    #line = line[st+1,end]
    #words = line.split(",")

    zero_left = None
    with open(filename,"r") as f:
        for line in f:
            if "zero_left" in line and "=" in line and not line.strip().startswith("#"):
                parts = line.split("=")
                zero_left = float(parts[1])
                break

    # interfaces
    with open(filename,"r") as f:
        for line in f:
            if "timestep" in line and "=" in line and not line.strip().startswith("#"):
                parts = line.split("=")
                parts = parts[1].split("#")   # cut off comments if that is still there
                timestep = float(parts[0])
                break

    return interfaces,zero_left,timestep

def get_LMR_interfaces(interfaces, zero_left):
    """Get the left, middle, right interfaces for each PyRETIS folder-ensemble"""
    LMR_interfaces = []
    LMR_strings = []
    if zero_left:
        LMR_interfaces.append([zero_left, (zero_left + interfaces[0])/2., interfaces[0]])
        LMR_strings.append(["l_[-1]", "( l_[-1] + l_[0] ) / 2", "l_[0]"])
    else:
        LMR_interfaces.append([interfaces[0], interfaces[0], interfaces[0]])
        LMR_strings.append(["l_[0]", "l_[0]", "l_[0]"])
    LMR_interfaces.append([interfaces[0], interfaces[0], interfaces[1]])
    LMR_strings.append(["l_[0]", "l_[0]", "l_[1]"])
    for i in range(1, len(interfaces)-1):
        LMR_interfaces.append([interfaces[i-1], interfaces[i], interfaces[i+1]])
        LMR_strings.append(["l_[{}]".format(i-1), "l_[{}]".format(i), "l_[{}]".format(i+1)])

    return LMR_interfaces, LMR_strings
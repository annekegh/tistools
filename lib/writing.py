"""Functions that write files"""

import numpy as np
import matplotlib.pyplot as plt

def write_histogram(fn,bin_midst,hist,F):
    with open(fn,"w+") as f:
        dens = hist/np.sum(hist)
        f.write('# {} {} {} {}\n'.format('z', 'hist', 'dens', 'F',))
        for i in range(len(hist)):
            f.write('{} {} {} {}\n'.format(bin_midst[i], hist[i], dens[i], F[i],))

def write_plot_block_error(filename, running_estimate, rel_errors, interval):
    """
    Generates and saves a block error analysis plot and writes results to a text file.

    This function creates a plot of relative error vs. block length, highlighting 
    the average relative error in the second half of the dataset. It also saves the 
    computed values in a `.txt` file.

    Parameters
    ----------
    filename : str
        The output filename.
    
    running_estimate : list or np.ndarray
        The sequence of running estimates.

    rel_errors : list or np.ndarray
        The computed relative errors for different block lengths.

    """
    x = np.arange(1, len(rel_errors) + 1)  # Block lengths (x-axis)
    best_estimate = running_estimate[-1]  # The final, most accurate estimate
    second_half_err_avg = np.mean(rel_errors[len(rel_errors) // 2:])  # Avg error in second half

    # Set up the plot
    plt.ioff()
    plt.figure(figsize=(10, 6))
    plt.plot(x * interval, rel_errors, label="Relative Error")
    plt.axhline(y=second_half_err_avg, color='r', linestyle='--',
                label=f"Second Half Avg = {second_half_err_avg:.4f}")
    
    # Labels and title
    plt.xlabel("Block Length")
    plt.ylabel("Relative Error")
    plt.title(f"Best Estimate = {best_estimate:.4f}, "
              f"Relative Error = {100 * second_half_err_avg:.2f} %")
    plt.legend()

    # Save the figure
    plt.savefig(filename, dpi=500, bbox_inches='tight')
    plt.close()

    # Write results to a text file
    with open(filename + ".txt", "w") as f:
        f.write(f"# Length of the dataset: {len(running_estimate)} with interval of {interval}\n")
        f.write(f"# Best estimate: {best_estimate:.12f}\n")
        f.write(f"# Averaged relative error: {second_half_err_avg:.12f}\n")
        f.write("# ===============================\n")
        f.write("# Block-Length\tRelative-Error\n")
        
        for i in range(len(x)):
            f.write(f"{x[i]*interval}\t{rel_errors[i]:.12f}\n")


def write_running_estimates(filename, cycles, *args):
    """
    Writes running estimates to a formatted text file.

    This function takes multiple arrays of running estimates along with their 
    corresponding labels and writes them into a structured text file.

    Parameters
    ----------
    filename : str
        The output filename.

    cycles : list or np.ndarray
        The cycle numbers corresponding to the running estimates.

    *args : tuple
        Alternating arrays and labels. Each array contains running estimates, 
        and each label is a string describing the corresponding array.
    """
    # Convert input arguments to NumPy arrays if they are not already
    running_estimates = [np.array(arr) if not isinstance(arr, np.ndarray) else arr for arr in args[0::2]]
    labels = [str(label) for label in args[1::2]]  # Ensure labels are strings

    # Determine column widths for formatting
    col_widths = [8]  # Width of the "cycle" column
    extended_labels = []

    for idx, label in enumerate(labels):
        array_shape = running_estimates[idx].shape
        num_columns = array_shape[1] if len(array_shape) > 1 else 1  # Number of columns in the data

        for i in range(num_columns):
            extended_label = f"{label}_{i}+" if num_columns > 1 else label  # Labeling for multi-column data
            extended_labels.append(extended_label)
            col_widths.append(max(len(extended_label), 18))  # Ensure sufficient column width

    # Write the data to file
    with open(filename, "w") as f:
        # Write the header
        header = "".join(f"{label:<{col_widths[i]}}" for i, label in enumerate(["cycle"] + extended_labels))
        f.write(header + "\n")

        # Write the data rows
        for i in range(len(cycles)):
            row_values = [f"{cycles[i]:<{col_widths[0]}}"]  # Cycle number column

            for j in range(len(labels)):
                array = running_estimates[j]
                for col in range(array.shape[1] if len(array.shape) > 1 else 1):
                    value = array[i, col] if len(array.shape) > 1 else array[i]
                    value_str = f"{value:.10f}"  # Ensure consistent floating-point format
                    row_values.append(f"{value_str:<{col_widths[j + 1]}} ")

            f.write("".join(row_values) + "\n")

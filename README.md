# tistools

Tools to analyse the output of TIS or RETIS simulations

mcdiff is a Python program that extracts the information from the output
of TIS or RETIS simulations performed with the program PyRETIS
(see www.pyretis.org). It can be used to assess the statistics
of the path ensembles and those of the order parameter(s).

Warning: The package is in full development and is not (!) well tested
nor complete. It mainly serves as a working tool while developing new methods.

tistools uses the following third-party libraries

- numpy
- matplotlib

## Getting Started

Installing the code takes approximately 5 minutes.

### System Dependencies

- Linux or MacOS 
- python >= 3.

<!--
### Install the Requirements 

- -Installation via anaconda

```
conda install "numpy>=1.16.0" "scipy>=1.2.0" "matplotlib>=2.0.2"
```

```
conda install numpy matplotlib
```

or pip

```
pip install numpy matplotlib
```

-->


### Download and Install tistools

First, download the code and go to the directory with the code.

```
git clone https://github.com/annekegh/tistools.git
cd tistools
```

Next, use one of the installation commands

```
pip install .
```
or when doing code development in tistools:
```
pip install -e .    # code is editable
```

In case you want to be sure that pip installs
in the correct environment (pip can be unpredictable)
```
which python   # check that this is the python you want to use
python -m pip install -e .   # code is editable, leave out -e if that's not needed.
```

### Executables and Examples

The setup process installs the following terminal command


```tistools-distr-op```

With this command, you can plot the distributions of the order parameter in each path ensemble.
You can either execute the command in the directory where the data is located,
or you can give the directory in an optional argument.

```tistools-distr-op -i myproject/system8/```

Use the help function to see the other optional arguments.

```tistools-distr-op -h```


### License

See LICENSE file.

### Authors
- An Ghysels, Ghent University


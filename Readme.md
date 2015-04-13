# Optimal immune repertoires

This repository contains the code associated with the manuscript

Mayer, Balasubramanian, Mora, Walczak : "How a well adapted immune system is organized"

It allows reproduction of all numerical results reported in the manuscript.

[![DOI](https://zenodo.org/badge/doi/10.5281/zenodo.16796.svg)](http://dx.doi.org/10.5281/zenodo.16796)

## Dependencies

The code is written in Python and depends on a number of numerical and scientific libraries.
Below we give the version numbers of the packages for which the code is known to run.

* Python 2.7.8
* Numpy 1.8.2
* Scipy 0.16.0
* Matplotlib 1.3.1
* Cython 0.20.2 and relevant development versions of libraries (only needed for figure 5) 

Optionally pyFFTW can be used to speed up some of the calculations.
In the absence of this package the code automatically falls back to the corresponding scipy functions.

## Usage

Download the source code by cloning the repository `git clone https://github.com/andim/optimmune`. 
Follow the following set of instructions in the given order.

* run `make cython` in library directory (only needed for figure 5)
* run `python run*.py` in figure directories to produce data.
    - In a number of cases the optimization files produce results for a range of parameters. Which parameters are used is controlled by a command line argument. The command line argument is a single integer between one and the number of different parameter combinations. On a computing cluster on which a grid engine is installed the looping over different arguments can be performed via the provided submit files (`qsub arrayjob.sh`).
    - Warning: some of the optimizations run for a long time (> 1h).
* for some figures the results need to be postprocessed by invoking `python calc*.py`
* run `python fig*.py` in figure directories to produce figures

Here the * is a placeholder for the specific filenames. Note: As most of the simulations are stochastic you generally do not get precisely equivalent plots.

## License

The source code is freely available under an MIT license, unless otherwise noted within a file.

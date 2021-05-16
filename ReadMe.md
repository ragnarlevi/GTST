This repository contains code for kernel two-sample tests for graphs.

The folder [GraphMMDExperiments](https://github.com/ragnarlevi/MMDGraph/tree/master/GraphMMDExperiments) contains scripts that simulates the p-values, powers for different samples. .py files are script that generate data .ipynb are jupyter notebooks used to visualize output from the .py files. Currently the samples are only simulates from a binomial random graph and basic stochastic block models. 

The file [MMDforGraphs](https://github.com/ragnarlevi/MMDGraph/blob/master/MMDforGraphs.py) contains function that do p-value permutation/bootstrap, calculate MMD test statistics and create kernel matrix (based on the grakel package).

The file [SBM](https://github.com/ragnarlevi/MMDGraph/blob/master/SBM.py) is a function that generates the basic stochastic block model.

The folder [data](https://github.com/ragnarlevi/MMDGraph/tree/master/data) contains simulation outputs

TODO: simulate degree corrected SBM, implement Wasserstein Weisfeiler-Lehman Graph Kernels,
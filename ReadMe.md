This repository contains code for kernel two-sample tests for graphs.

To clone the repository use `git clone https://github.com/ragnarlevi/MMDGraph.git`.

This repository depends on

* `numpy` - `pip install numpy`
* `pandas` - `pip install pandas`
* `networkx` - `pip install networkx`
* `grakel` - `pip install grakel`
* `matplotlib` - `pip install matplotlib`
* `numba` - `pip install numba`
* `ot` - `pip install pot` (only used in the WWL kernel)

The experiments available are under the folder [Experiments](https://github.com/ragnarlevi/MMDGraph/tree/master/Experiments). Each folder contains a single `run.py` and which take the same input parser. Why input parser? Because I wanted to run the scripts on a cluster where I can not use a notebook/gui to change the variables. So to run an experiment one has to use a terminal. The general input is as follows:

`python Experiments/{Experiment Name}/run.py -B {int} -N {int} -p {string} -d {int} -[Kernel specifics, see below] -[Experiment specifics, see below]`

* -N number of iterations used to estimate the power
* -B number of bootstraps used in the permutation test, estimates the p_value of each iteration
* -p the path to where the pandas dataframe (containing ROC curve for each test) should be saved
* -d Number of processes used to simulate the iterations. Each core will process N/d simulations. Preferably, we should have the relationship $d*i = N$, where $i$ is an integer

The kernel specifics depends on the kernel that is used. If the kernel of choice does not use a specific parameter, simply skip

* -kernel The name of the kernel to be used:
  * wl for the Weisfeiler-Lehman subtree kernel (Grakel package)
  * sp for the shortest path kernel (Grakel package)
  * pyramid for the pyramid kernel (Grakel package)
  * prop for the Propagation kernel (Grakel package)
  * wloa for the Weisfeiler-Lehman optimal assignment (Grakel package)
  * vh for the vertex histogram kernel (Grakel package)
  * rw for the random walk kernel (Grakel package)
  * odd for the ODD-STh kernel (Grakel package)
  * dk for the Deep Graph kernel (Under [myKernel](https://github.com/ragnarlevi/MMDGraph/tree/master/myKernels))
  * WWL for the Wasserstein Weisfeiler-Lehman kernel (Under [myKernel](https://github.com/ragnarlevi/MMDGraph/tree/master/myKernels))
* -norm int, Should the kernel be normalized, 0 gives False.
* -nitr int, Number of WL iterations. Used in wl, wloa, dl and WWL
* -wlab int, Should the kernel consider labels? Used in sp and rw. 0 gives False
* -type str, Type of kernel. Used in rw where the argument can be exponential or geometric and the dk where the argument can be sp (shortest path similarity) or wl (wl similarity)
* -l float, Discount used in rw and wwl
* -tmax, int or omitted, Number of steps used in prop and rw. It has to be given for the prop kernel but can be omitted in the rw kernel. If omitted the number of walks is infinite
* -L int, Histogram level used in the pyramid kernel.
* -dim int, The dimension of the hypercube used in the pyramid kernel (number of eigenvector embeddings)
* -w float, The binwidth of the local sensitive hashing. Used in the prop and hash kernel
* -M str The preserved distance metric (on local sensitive hashing) used in prop. Vales are TV (Total Variation) and H (Hellinger)
* -dagh int, Maximum (single) dag height. If omitted there is no restriction. Used in ODD
* -sk int. Should the Wasserstein calculation be approximated with the sinkhorn method? Used in wwl. if 0 then False

The Experiment specifics are the following:

* BGDegreeLabel. Graphs generated using a binomial graph model. The nodes are labeled according to their degree.
  * -n1 number of samples in sample 1
  * -n2 number of samples in sample 2
  * -nnode1 number of nodes of each graph in sample 1
  * -nnode2 number of nodes of each graph in sample 2
  * -k1 average degree of the nodes in the graphs in sample 1
  * -k2 average degree of the nodes in the graphs in sample 2
* SBMOnlyRandomLabel. Graphs generated according to a SBM model with 3 blocks. Both samples have graphs with the same topology but the nodes are labeled slightly differently.
  * -n1 number of samples in sample 1
  * -n2 number of samples in sample 2
  * -noise how much noise is in the label generation in each block. Example: One sample labels node according to their block membership, that is, nodes in block 1 get the label 1. The other sample has some noise. The nodes in block 1 can get the label 2 with probability noise/2 or label 3 with probability noise/2
  
An example of a command running an Binomial Graph experiment using a propagation kernel:
`python Experiments/BGDegreeLabel/run.py -B 1000 -N 10 -p 'data/test.pkl' -n1 20 -n2 20 -nnode1 20 -nnode2 20 -k1 4 -k2 5 -d 4 -norm 0 -kernel prop -w 0.01 -M TV -tmax 4` 


Most experiments run pretty quickly. But some kernels take longer, those are the shortest path kernel, the ODD kernel and the RW kernel. The RW kernel is especially slow.

In this repository you can also find the folder [Analysis](https://github.com/ragnarlevi/MMDGraph/tree/master/Analysis). This folder contains notebooks which takes data from the folder [data](https://github.com/ragnarlevi/MMDGraph/tree/master/data) (a folder that contains experiments that have already been simulated, a lot of files) and plots some graphs. The notebooks use functions from [mmdutils.py](https://github.com/ragnarlevi/MMDGraph/blob/master/mmdutils.py)

Another folder is [_Workbench](https://github.com/ragnarlevi/MMDGraph/tree/master/_Workbench) which contains notebooks where I do some testing before doing experiments or before writing "production" code. The notebooks use functions/classes from [MMDforGraphs.py](https://github.com/ragnarlevi/MMDGraph/blob/master/MMDforGraphs.py)




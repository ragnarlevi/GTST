---
title: 'MMDGraph: A Python package for graph two-sample testing'
tags:
  - Python
  - kernel
  - two sample testing
  - graph
authors:
  - name: Ragnar Leví Guðmundarson
    equal-contrib: true
    affiliation: 1 
  - name: Gareth W. Peters
    equal-contrib: true
    affiliation: 2
affiliations:
 - name: Centre for Networks & Enterprise, Business School, Heriot Watt University
   index: 1
 - name: 	Department of Statistics & Applied Probability, University of California, Santa Barbara
   index: 2
date: 16 November 2022
bibliography: paper.bib

---

# Summary
In many practical applications, one needs to draw inference from a sample of networks. Kernel methods have proven to be useful in pattern recognition tasks such as classification and can be further extended as an inference procedure to two-sample hypothesis testing on structured data. The method embeds the graphs into a reproducing kernel Hilbert space (RKHS) via a feature map which is then extended further to the embedding of a probability distribution. The two-sample null hypothesis is that the generating mechanism behind the two samples is the same and the test statistic, which is called the maximum mean discrepancy (MMD), is the largest distance between the means of the two sample embeddings. Graph kernels are already well established and widely-used for solving classification tasks on graphs and can further be used to compare samples of graphs and to perform graph screening. They provide a very flexible way of comparing graphs as they exist for a wide range of different graph structures, for example, weighted, directed, labeled and attributed graphs. Their performance depends on their expressiveness, that is, their ability to distinguish non isomorphic graphs. The difficulty of distinguishing two-samples of graphs varies strongly based on the type of graphs. The package provides functions to perform two-sample hypothesis testing using various estimators and various kernels. The package further allows the estimatation of graphs from a real valued data matrix using the graph lasso method.

The workflow is following: 1) Use two data arrays to estimate two sequences/samples of graphs using the graphical lasso [@friedman2008sparse], This step can be skipped if the practisioner already has the samples of graphs in a networkx format [@hagberg2020networkx]. 2) Select a graph kernel. 3) Select an estimator of the MMD or try multiple estimators to obtain a p-value.

# Brief Introduction to the problem of Graph Two-Sample Testing

Let $G(V,E)$ denote a graph with vertex set $V$ and edge set $E$. In the two-sample testing of graph-valued, we assume we are given two sets of samples/observations that comprise collections of graph-valued data $\{G_1,...,G_{n}\}$ and $\{G'_1,...,G'_{n'}\}$ where $G_i, G'_j \in \Omega, \quad \forall i,j$. The graphs in the two samples are all generated independently from two probability spaces $(\Omega, \mathcal{F}, \mathbb{P})$  and $(\Omega, \mathcal{F}, \mathbb{Q})$, and the goal is to infer whether $ \mathbb{P} = \mathbb{Q}$. 




# Statement of need

Graph two-sample hypothesis testing is a problem that frequently arises in variuous disciplines, for example in bioinformatics [@bassett2008hierarchical], community detection [@girvan2002community], and risk management [@carreno2017identifying]. Graph two-sample hypothesis have mostly been performed by using graph statistics such has the degree centrality and shortest paths. Although theses method can often give good performances they fail to take into account various attributes that are often present in real graphs such as node labels, edge labels, node attributes and edge weights. When the kernel two-sample hypothesis testing was introduces [@Gretton2012] a flood gate opened to allow for testing of such attributes and therefore providing a flexible way of performing two-sample hypothesis testing. Luckily, there also exists a  vast literature on graph kernels [@Kriege2020] [@Nikolentzos2019]. Until now, there is no package which allows one to estimate graphs from real valued data matrices and perform hypothesis testing in a flexible manner. The package provides three functionalities 1) Functions to estimate the two-sample hypothesis test statistic, 2) Functions to calculate different graph kernels and 3) A function to estimate graphs from data matrices. The package allows other network science researches who may not be programmes to use the MMD testing framework. 


The code was orignally used in a paper CITE OUR PAPER to compare pairwise asset return process relationships to study and understand risk and return in portfolio management practice. This allows one to statistically test for significance of any detected differences in portfolio diversification between any portfolio investment strategy when applying differing investment screening criteria or optimal investment strategies. We further remark that this package can furhter be used in other fields other than portfolio comparision. For example, the package allows, in a straight forwards manner, comparing different communities detect by community detection algorithms, finding change-point events in graphs, testing for traffic diffrences in traffic networks, and comparing ego-netwroks of entities.

# The MMDGraph package


There exists a python package called GraKel [@JMLR:v21:18-370] which is dedicated to calculating various graph kernels. The package is very user-friendly so the MMDGraph user can use all graph kernels available in the Grakel package. We then allow users to use other kernels not available in GraKel such as: Fast random walk kernels based on ideas from [@Kang2012] along with an additional fast random walk kernel for edge-labeled graphs, 2) The Wasserstein Weisfeiler-Lehman Graph kernel [@Togninalli2019] whose original code was adjusted for the package needs, 3) The Deep Graph kernel [@DK], and 4) The Graph neural tangent kernel [@Du2019] whose original code was adjusted for the package needs. The package assumes that the graphs passed are a networkx object [@hagberg2020networkx]. One can additionally use its own pre-computed kernel to perform tests.

The MMDGraph package allows the user to use 4 different estimates of the MMD, namely the unbiased version $\text{MMD}_u$, the biased version  $\text{MMD}_b$, the unbised computationally cheaper estimate $\text{MMD}_l$, and a robust estimate $\text{MONK}$. The MONK estimator was developed by $\text{MONK}$ and they do provide the code online and in a package environment. However, we have adjusted the code slighlty to allows for robust comparion of samples of different sizes. The MMDGraph then estimates the $p$-value of test by using a bootstrap or a Permutation sampling scheme.


The package also allows for estimating graphs using sklearn's graphical lasso [@sklearn_api]. Additionall preprocessing can be done by using the nonparanormal transform [@liu2009nonparanormal]. The best graph is found by using the EBIC criterio [@Orzechowski2019]



# References

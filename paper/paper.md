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
    equal-contrib: true # (This is how you can denote equal contributions between multiple authors)
    affiliation: 2
  - name: Author with no affiliation
    corresponding: true # (This is how to denote the corresponding author)
    affiliation: 3
affiliations:
 - name: Centre for Networks & Enterprise, Business School, Heriot Watt University
   index: 1
 - name: 	Department of Statistics & Applied Probability, University of California, Santa Barbara
   index: 2
date: 16 November 2022
bibliography: paper.bib

---

# Summary
In many practical applications, one needs to draw inference from a sample of networks. Kernel methods have proven to be useful in pattern recognition tasks such as classification and two-sample testing on structured data. The method embeds the graphs into a reproducing kernel Hilbert space (RKHS) via a feature map which is then extended further to the embedding of a probability distribution. The two-sample null hypothesis is that the generating mechanism behind the two samples is the same and the test statistic, which is called the maximum mean discrepancy (MMD), is the largest distance between the means of the two sample embeddings. Graph kernels are already well established and widely-used for solving classification tasks on graphs and can further be used to compare samples of graphs and to perform graph screening. They provide a very flexible way of comparing graphs as they exist for a wide range of different graph structures, for example, weighted, directed, labeled and attributed graphs. Their performance depends on their expressiveness, that is, their ability to distinguish non isomorphic graphs. The difficulty of distinguishing two-samples of graphs varies strongly based on the type of graphs. The package provides functions to perform two-sample hypothesis testing using various estimators and various kernels. The package further allows the estimatation of graphs from a real valued data matrix using the graph lasso method.

# Statement of need

Graph two-sample hypothesis testing is a problem that frequently arises in variuous disciplines, for example in bioinformatics [@bassett2008hierarchical], community detection [@girvan2002community], and risk management [@carreno2017identifying]. The code was orignally used in a paper CITE OUR PAPER to compare  pairwise asset return
process relationships to study and understand risk and return in portfolio management practice. This allows one to statistically test for significance of any detected differences in portfolio diversification between any portfolio investment strategy when applying differing investment screening criteria or optimal investment strategies. Until now, there is no package which allows one to estimate graphs from real valued data matrices and perform hypothesis testing in a flexible manner. The flexibility comes from the fact that the package uses graph kernels to perform the two-sample tests. Kernel two-sample hypothesis was introduced by [@Gretton2012], and the method can be used on any object where a kernel function has been defined. Luckily, there is a vast literature on graph kernels [@Kriege2020] [@Nikolentzos2019].

# The MMDGraph package


There exists a python package called  GraKel [@JMLR:v21:18-370] which is dedicated to calculating various graph kernels. The package is very user-friendly so the MMDGraph user can use all graph kernels available in the Grakel package. We then allow users to use other kernels not available in GraKel such as: Fast random walk kernels based on ideas from [@Kang2012] along with an additional fast random walk kernel for edge-labeled graphs, 2) The Wasserstein Weisfeiler-Lehman Graph kernel [@Togninalli2019] whose original code was adjusted for the package needs, 3) The Deep Graph kernel @[DK], and 4) The Graph neural tangent kernel [@Du2019] whose original code was adjusted for the package needs. The package assumes that the graphs passed are a networkx object.

The MMDGraph package allows the user to use 4 different estimates of the MMD, namely the unbiased version $\text{MMD}_u$, the biased version  $\text{MMD}_b$, the unbised computationally cheaper estimate $ \text{MMD}_l$, and a robust estimate $\text{MONK}$. The MONK estimator was developed by $\text{MONK}$ and they do provide the code online and in a package environment. However, we have adjusted the code slighlty to allows for robust comparion of samples of different sizes. The MMDGraph then estimates the $p$-value of test by using a bootstrap or a Permutation sampling scheme.


The package also allows for estimating graphs using sklearn's graphical lasso [@sklearn_api]. Additionall preprocessing can be done by using the nonparanormal transform [@liu2009nonparanormal]. The best graph is found by using the EBIC criterio [@Orzechowski2019]





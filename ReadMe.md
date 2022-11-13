# MMDGraph

## What is this package for?

This package contains code to perform kernel two-sample hypothesis testing on samples of graphs. The code additionally allows for estimation of graphs from a real data matrix.


## How to install

## Usage

We will go through the case when the user has it own networkx graphs, and when they are estimated from data matricies. We will go thorugh multiple scenariros


```python
import numpy as np
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt
import MMDGraph as mg
```

### Fit when H1 true, different edge probability

Start by creating sample graphs


```python

n1 = n2 = 50
g1 = [nx.fast_gnp_random_graph(30,0.3,seed = 42) for _ in range(n1)]
g2 = [nx.fast_gnp_random_graph(30,0.2,seed = 50) for _ in range(n2)]
for j in range(len(g1)):
    nx.set_node_attributes(g1[j],  {key: str(value) for key, value in dict(g1[j].degree).items()} , 'label')
for j in range(len(g2)):
    nx.set_node_attributes(g2[j], {key: str(value) for key, value in dict(g2[j].degree).items()}, 'label')

```

Performd mmd test using various kernels


```python
# Random Walk
MMD_out = mg.MMD()
MMD_out.fit(G1 = g1, G2 = g2, kernel = 'RW_ARKU_plus', mmd_estimators = 'MMD_u', r = 2, c = 0.001)
print(f" RW_ARKU_plus {MMD_out.p_values}")
```

    C:\Users\User\Code\MMDGraph\MMDGraph\kernels\RandomWalk.py:937: FutureWarning: adjacency_matrix will return a scipy.sparse array instead of a matrix in Networkx 3.0.
      return scipy.sparse.csr_matrix(nx.adjacency_matrix(G ,weight=edge_attr), dtype=np.float64)
    

     RW_ARKU_plus {'MMD_u': 0.0}
    


```python
# GNTK kernel
MMD_out = mg.MMD()
MMD_out.fit(G1 = g1, G2 = g2, kernel = 'GNTK', mmd_estimators = 'MMD_u', num_layers = 2, num_mlp_lauers = 2, jk = True, scale = 'uniform')
print(f" GNTK {MMD_out.p_values}")
```

    100%|███████████████████████████████████████████████████████████████████████████| 5050/5050.0 [00:05<00:00, 969.28it/s]

     GNTK {'MMD_u': 0.0}
    

    
    


```python
# WWL kernel
MMD_out = mg.MMD()
MMD_out.fit(G1 = g1, G2 = g2, kernel = 'WWL', mmd_estimators = 'MMD_u', discount = 0.1, h = 2, node_label = 'label')
print(f" WWL {MMD_out.p_values}")
```

    Using label as node labels
     WWL {'MMD_u': 0.0}
    


```python
# Deep Kernel without the deepness
MMD_out = mg.MMD()
MMD_out.fit(G1 = g1, G2 = g2, kernel = 'DK', mmd_estimators = 'MMD_u', type = 'wl', wl_it = 4, node_label = 'label')
print(f" ML DK {MMD_out.p_values}")
```

    Using label as node labels
     ML DK {'MMD_u': 0.0}
    


```python

# Deep kernel with deepness, user has to install gensim, this might take some time, can try to increase number of workers
MMD_out = mg.MMD()
MMD_out.fit(G1 = g1, G2 = g2, kernel = 'DK', mmd_estimators = 'MMD_u', type = 'wl', wl_it = 4, opt_type = 'word2vec', node_label = 'label', workers = 10)
print(f" Deep DK {MMD_out.p_values}")
```

    Using label as node labels
     Deep DK {'MMD_u': 0.0}
    


```python
# Grakel kernel example
kernel = [{"name": "weisfeiler_lehman", "n_iter": 1}, {"name": "vertex_histogram"}]
MMD_out = mg.MMD()
MMD_out.fit(G1 = g1, G2 = g2, kernel = kernel, mmd_estimators = 'MMD_u', node_label = 'label')
print(f" WL {MMD_out.p_values}")
```

    Using label as node labels
    label
     WL {'MMD_u': 0.0}
    


```python
# Grakel propagation
kernel = [ {"name":"propagation", 't_max':5, 'w':0.1, 'M':"TV"}]
MMD_out = mg.MMD()
MMD_out.fit(G1 = g1, G2 = g2, kernel = kernel, mmd_estimators = 'MMD_u', node_label = 'label')
print(f" propagation {MMD_out.p_values}")

```

    Using label as node labels
    label
     propagation {'MMD_u': 0.0}
    

### H1 true different weights


```python
n1 = n2 = 50
g1_weights = [nx.fast_gnp_random_graph(30,0.3, seed = 42) for _ in range(n1)]
g2_weight = [nx.fast_gnp_random_graph(30,0.3, seed = 50) for _ in range(n2)]


for j in range(len(g1_weights)):
    nx.set_node_attributes(g1_weights[j],  {key: str(value) for key, value in dict(g1_weights[j].degree).items()} , 'label')
for j in range(len(g2_weight)):
    nx.set_node_attributes(g2_weight[j], {key: str(value) for key, value in dict(g2_weight[j].degree).items()}, 'label')


def edge_dist(loc, scale ):
    from scipy.stats import uniform
    return np.random.normal(loc = loc, scale = scale)# uniform.rvs(size=1,  loc = loc , scale = scale)[0]
def add_weight(G, loc, scale ):
    edge_w = dict()
    for e in G.edges():
        edge_w[e] = edge_dist(loc, scale)
    return edge_w


for G in g1_weights:
    nx.set_edge_attributes(G, add_weight(G, loc = 0.5, scale = 1), "weight")
for G in g2_weight:
    nx.set_edge_attributes(G, add_weight(G, loc = 0.5, scale = 4), "weight")

```


```python
# Random Walk
MMD_out = mg.MMD()
MMD_out.fit(G1 = g1_weights, G2 = g2_weight, kernel = 'RW_ARKU_plus', mmd_estimators = 'MMD_u', r = 2, c = 0.001, edge_attr = 'weight')
print(f" RW_ARKU_plus {MMD_out.p_values}")
```

    Using weight as edge attributes
    

    C:\Users\User\Code\MMDGraph\MMDGraph\kernels\RandomWalk.py:937: FutureWarning: adjacency_matrix will return a scipy.sparse array instead of a matrix in Networkx 3.0.
      return scipy.sparse.csr_matrix(nx.adjacency_matrix(G ,weight=edge_attr), dtype=np.float64)
    

     RW_ARKU_plus {'MMD_u': 0.0}
    


```python
# Random Walk weights ignored, should not reject
MMD_out = mg.MMD()
MMD_out.fit(G1 = g1_weights, G2 = g2_weight, kernel = 'RW_ARKU_plus', mmd_estimators = 'MMD_u', r = 2, c = 0.001, edge_attr = None)
print(f" propagation {MMD_out.p_values}")
```

     propagation {'MMD_u': 0.0}
    


```python
# Grakel pyramid
kernel = [{"name": "pyramid_match", "L": 6, "d":6, 'with_labels':False}]
MMD_out = mg.MMD()
MMD_out.fit(G1 = g1_weights, G2 = g2_weight, kernel = kernel, mmd_estimators = 'MMD_u', edge_attr = 'weight')
print(f" pyramid_match {MMD_out.p_values}")
```

    Using weight as edge attributes
    None
     pyramid_match {'MMD_u': 0.0}
    


```python
# propagation, needs node attribute or label 
kernel = [ {"name":"propagation", 't_max':5, 'w':0.01, 'M':"TV"}]
MMD_out = mg.MMD()
MMD_out.fit(G1 = g1_weights, G2 = g2_weight, kernel = kernel, mmd_estimators = 'MMD_u', edge_attr = 'weight', node_label = 'label')
print(f" propagation {MMD_out.p_values}")
```

    Using weight as edge attributes
    Using label as node labels
    label
     propagation {'MMD_u': 0.0}
    

 ### H1 true different attributes


```python

n1 = n2 = 50
g1_attr = [nx.fast_gnp_random_graph(30,0.2,seed=42) for _ in range(n1)]
g2_attr = [nx.fast_gnp_random_graph(30,0.2,seed=50) for _ in range(n2)]
for j in range(len(g1_attr)):
    nx.set_node_attributes(g1_attr[j], dict( ( (i, np.random.normal(loc = 0, scale = 0.1, size = (1,))) for i in range(len(g1_attr[j])) ) ), 'attr')
for j in range(len(g2_attr)):
    nx.set_node_attributes(g2_attr[j], dict( ( (i, np.random.normal(loc = 0.1, scale = 0.1, size = (1,))) for i in range(len(g2_attr[j])) ) ), 'attr')

```


```python
# Random Walk with weights and node attributes
MMD_out = mg.MMD()
MMD_out.fit(G1 = g1_attr, G2 = g2_attr, kernel = 'RW_ARKU_plus', mmd_estimators = 'MMD_u', r = 4, c = 0.01, node_attr = 'attr')
print(f" RW_ARKU_plus {MMD_out.p_values}")
```

    Using attr as node attributes
     RW_ARKU_plus {'MMD_u': 0.0}
    


```python
# GNTK with node attributes
MMD_out = mg.MMD()
MMD_out.fit(G1 = g1_attr, G2 = g2_attr, kernel = 'GNTK', mmd_estimators = 'MMD_u', num_layers = 2, num_mlp_lauers = 2, jk = True, scale = 'uniform', node_attr = 'attr')
print(f" GNTK {MMD_out.p_values}")
```

    Using attr as node attributes
    

    100%|██████████████████████████████████████████████████████████████████████████| 5050/5050.0 [00:03<00:00, 1274.32it/s]

     GNTK {'MMD_u': 0.0}
    

    
    


```python
# Grakel propagation
kernel = [ {"name":"propagation", 't_max':5, 'w':0.1, 'M':"L1",'with_attributes':True}]
MMD_out = mg.MMD()
MMD_out.fit(G1 = g1_attr, G2 = g2_attr, kernel = kernel, mmd_estimators = 'MMD_u', discount = 0.1, h = 2, node_attr = 'attr')
print(f" Propagation {MMD_out.p_values}")
```

    Using attr as node attributes
    attr
     Propagation {'MMD_u': 0.0}
    

### Two data matrices different structure
It is possible to estimate graphs from data matrices


```python
G = nx.fast_gnp_random_graph(11, 0.25, seed = 42)
assert nx.is_connected(G)

for e in G.edges():
    if np.random.uniform() <0.1:
        w = np.random.uniform(low = 0.1, high = 0.3)
        G.edges[e[0], e[1]]['weight'] = -w
    else:
        w = np.random.uniform(low = 0.1, high = 0.3)
        G.edges[e[0], e[1]]['weight'] = w

A = np.array(nx.adjacency_matrix(G).todense())
np.fill_diagonal(A, np.sum(np.abs(A), axis = 1)+0.1)

A_s = A.copy()
A_s[7,4] = 0
A_s[4,7] = 0
A_s[5,2] = 0
A_s[2,5] = 0


X1 = np.random.multivariate_normal(np.zeros(11),np.linalg.inv(A), size = 5000)
X2 = np.random.multivariate_normal(np.zeros(11),np.linalg.inv(A_s), size = 5000)
```

    C:\Users\User\AppData\Local\Temp\ipykernel_16868\3751381554.py:12: FutureWarning: adjacency_matrix will return a scipy.sparse array instead of a matrix in Networkx 3.0.
      A = np.array(nx.adjacency_matrix(G).todense())
    


```python
# window size = 200 so 5000/200 = 25 graphs in each sample
# Random Walk
MMD_out = mg.MMD()
MMD_out.estimate_graphs(X1,X2,window_size=200, alpha = np.exp(np.linspace(-5,-2,100)),beta = 0.5, nonparanormal=False,scale = False)
MMD_out.fit( kernel = 'RW_ARKU_plus', mmd_estimators = 'MMD_u', r = 5, c = 0.1, edge_attr = 'weight')
print(MMD_out.p_values)


```

    Using weight as edge attributes
    

    C:\Users\User\Code\MMDGraph\MMDGraph\kernels\RandomWalk.py:937: FutureWarning: adjacency_matrix will return a scipy.sparse array instead of a matrix in Networkx 3.0.
      return scipy.sparse.csr_matrix(nx.adjacency_matrix(G ,weight=edge_attr), dtype=np.float64)
    

    {'MMD_u': 0.059}
    


```python
# We can set node labels as degree (or define our own labelling, see below)
MMD_out = mg.MMD()
kernel = [{"name": "weisfeiler_lehman", "n_iter": 2}, {"name": "vertex_histogram"}]
MMD_out.estimate_graphs(X1,X2,window_size=400, alpha = np.exp(np.linspace(-5,-2,100)),beta = 0.5, nonparanormal=False,scale = False, set_labels="degree")
MMD_out.fit( kernel = kernel, mmd_estimators = 'MMD_u',  edge_attr = 'weight')
print(MMD_out.p_values)
```

    Using weight as edge attributes
    Using label as node labels
    label
    {'MMD_u': 0.036}
    


```python
np.fill_diagonal(A_s,0)
fig, ax = plt.subplots(2,2,figsize = (10,10))
pos = nx.kamada_kawai_layout(G, weight = None)
nx.draw(G, pos = pos, ax = ax[0,0])
nx.draw(nx.from_numpy_array(A_s), pos = pos, ax = ax[0,1])
nx.draw(MMD_out.G1[3], pos = pos, ax = ax[1,0])
nx.draw(MMD_out.G2[3], pos = pos, ax = ax[1,1])
ax[0,0].set_title("Sample 1 true precision structure")
ax[0,1].set_title("Sample 2 true precision structure")
ax[1,0].set_title("One estimated precision structure from sample 1")
ax[1,1].set_title("One estimated precision structure from sample 2")
```




    Text(0.5, 1.0, 'One estimated precision structure from sample 2')




    
![png](README_files/README_32_1.png)
    


### Two data matrices same structure different attributes
It is possible to estimate the graphs beforehand and apply a function to get node attributes


```python
G = nx.fast_gnp_random_graph(11, 0.25, seed = 42)
assert nx.is_connected(G)

for e in G.edges():
    if np.random.uniform() <0.1:
        w = np.random.uniform(low = 0.1, high = 0.3)
        G.edges[e[0], e[1]]['weight'] = -w
    else:
        w = np.random.uniform(low = 0.1, high = 0.3)
        G.edges[e[0], e[1]]['weight'] = w

A = np.array(nx.adjacency_matrix(G).todense())
np.fill_diagonal(A, np.sum(np.abs(A), axis = 1)+0.1)


X1 = np.random.multivariate_normal(np.zeros(11),np.linalg.inv(A), size = 10000)
X2 = np.random.multivariate_normal(np.ones(11),np.linalg.inv(A), size = 10000)
```

    C:\Users\User\AppData\Local\Temp\ipykernel_16868\3274406453.py:12: FutureWarning: adjacency_matrix will return a scipy.sparse array instead of a matrix in Networkx 3.0.
      A = np.array(nx.adjacency_matrix(G).todense())
    


```python
# Random Walk, with attributes, should reject. , the class will use the node label name 'attr'
def attr_function(X):
    return np.expand_dims(np.mean(X,axis = 0),axis=1)

MMD_out = mg.MMD()
MMD_out.estimate_graphs(X1,X2,window_size=400, alpha = np.exp(np.linspace(-5,-2,100)),beta = 0.5, nonparanormal=False,scale = False, set_attributes = attr_function)
MMD_out.fit( kernel = 'RW_ARKU_plus', mmd_estimators = 'MMD_u', r = 5, c = 0.1, edge_attr = 'weight', node_attr = 'attr')
print(MMD_out.p_values)
```

    Using weight as edge attributes
    Using attr as node attributes
    

    C:\Users\User\Code\MMDGraph\MMDGraph\kernels\RandomWalk.py:937: FutureWarning: adjacency_matrix will return a scipy.sparse array instead of a matrix in Networkx 3.0.
      return scipy.sparse.csr_matrix(nx.adjacency_matrix(G ,weight=edge_attr), dtype=np.float64)
    

    {'MMD_u': 0.0}
    


```python
# If we do not give attributes in this case the test should not reject
MMD_out_no_attr = mg.MMD()
MMD_out_no_attr.fit(G1= MMD_out.G1, G2 = MMD_out.G2, kernel = 'RW_ARKU_plus', mmd_estimators = 'MMD_u', r = 5, c = 0.1, edge_attr = 'weight')
print(MMD_out_no_attr.p_values)
```

    Using weight as edge attributes
    {'MMD_u': 0.66}
    


```python
# We can also try to make a label function, has to be a dictionary, the class will use the node label name 'label'
def label_function(X):
    m = np.mean(X,axis = 0)
    return {i:str(np.round(m[i],1)) for i in range(len(m))}

kernel = [{"name": "weisfeiler_lehman", "n_iter": 2}, {"name": "vertex_histogram"}]
MMD_out = mg.MMD()
MMD_out.estimate_graphs(X1,X2,window_size=400, alpha = np.exp(np.linspace(-5,-2,100)),beta = 0.5, nonparanormal=False,scale = False, set_labels= label_function)
MMD_out.fit(kernel = kernel, mmd_estimators = 'MMD_u', node_label = 'label')
print(MMD_out.p_values)
```

    Using label as node labels
    label
    {'MMD_u': 0.0}
    


```python
# We can also define labels using a dict
label_dict = {'1':{j:i for j,i in enumerate(['a']*6 + ['b']*5)}, 
              '2':{j:i for j,i in enumerate(['a']*4 + ['b']*7)}}
kernel = [{"name": "weisfeiler_lehman", "n_iter": 2}, {"name": "vertex_histogram"}]
MMD_out = mg.MMD()
MMD_out.estimate_graphs(X1,X2,window_size=400, alpha = np.exp(np.linspace(-5,-2,100)),beta = 0.5, nonparanormal=False,scale = False, set_labels= label_dict)
MMD_out.fit(kernel = kernel, mmd_estimators = 'MMD_u', node_label = 'label')
print(MMD_out.p_values)
```

    Using label as node labels
    label
    {'MMD_u': 0.0}
    

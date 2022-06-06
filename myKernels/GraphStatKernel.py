







from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
import networkx as nx

class G_stat_kernel():


        def __init__(self) -> None:
            pass

        def fit(self, Gs, method,  kernel_width = None, normalize = 0):
            Z = []
            if method == 'degree':
                for G in Gs:
                    Z.append(list(dict(G.degree()).values()))
            
            elif method == 'sp':
                for G in Gs:
                    Z.append([len(nx.shortest_path(G, source=i, target=j, weight=None, method='dijkstra')) for i in range(len(G)) for j in range(i+1,len(G))])
            else:
                raise ValueError(f'{method} method unknown')


            D2 = euclidean_distances(Z, squared=True)
            if kernel_width is None:
                upper = D2[np.triu_indices_from(D2, k=1)]
                kernel_width = np.median(upper, overwrite_input=True)
                bandwidth = np.sqrt(kernel_width / 2)
                kernel_width = 2 * bandwidth**2
                K = np.exp(-D2 * (1/kernel_width))
            else:
                K = np.exp(-D2 * (1/kernel_width))


            if normalize:
                K = self.normalize_gram_matrix(K)

        
            return K


        @staticmethod
        def normalize_gram_matrix(x):
            k = np.reciprocal(np.sqrt(np.diag(x)))
            k = np.resize(k, (len(k), 1))
            return np.multiply(x, np.outer(k,k))










                


















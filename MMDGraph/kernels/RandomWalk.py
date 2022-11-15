"""

fast Random walk kernels: See http://www.cs.cmu.edu/~ukang/papers/fast_rwgk.pdf
"""
import numpy as np
from numpy.linalg import eigh, inv

from sklearn.utils.extmath import randomized_svd
from scipy.sparse.linalg import eigsh
import scipy
import networkx as nx
import tqdm


def zero_div(x, y):
    return float(y) and float(x)/ float(y) or 0.0


class RandomWalk():


    def __init__(self, X, c, r, edge_attr = None, normalize = False, node_attr=None, edge_label = None, 
                    unique_edge_labels = None, unique_node_labels = None, node_label = None) -> None:
        """
        Parameters
        ---------------
        X: list,
            list of size N, of networkx graphs
        c: float,
            scalar
        edge_attr: str,
            Name of edge attributes, usually weight if the graphs are weighted.
        r: int, 
            number of eigenvalues used in approximation
        normalize: bool,
            Should the kernel be normalized
        node_attr: str,
            Name of node attribute, if not node attribute set as None
        edge_label:, str
            Name of edge labels, only used in ARKU_edge
        unique_edge_labels: list,
            Unique edge labels encountered in the graphs, only used in ARKU_edge
        unique_node_labels:list,
            Unique nodel labels encountered in the graphs, only used in ARKL
        node_label: str,
            Name of node labels
        

        
        """
        self.X = X
        self.c = c
        self.normalize = normalize
        self.node_attr = node_attr


        self.N = len(X)

        self.r = r

        self.edge_attr = edge_attr

        self.edge_label = edge_label
        self.unique_edge_labels = unique_edge_labels

        self.node_label = node_label
        self.unique_node_labels = unique_node_labels

        
        self.K = np.zeros((self.N, self.N))
        self.p = [None] * self.N 
        self.q = [None] * self.N 

    
    def fit(self,calc_type,  verbose = False, check_psd = True) -> None:
        """
        Fit a RW kernel

        Parameters
        ------------------------------
        calc_type:str,
            The type of RW kernel to use
        check_psd: bool,
        Check if Kernel matrix K is psd by inspecting the eigenvalues.
        
        """


        if verbose:
            pbar = tqdm.tqdm(disable=(verbose is False), total=self.N*(self.N+1)/2)
        else:
            pbar = None

        # Create lists to store caclulations
        self.U_list = [None] * self.N  # left SVD matrix of each adj matrix
        self.Lamda_list = [None] * self.N  # eigenvalues of each adj matrix
        self.Vt_list = [None] * self.N  # right transposed SVD matrix of each adj matrix

        self.all_A = [None] * self.N

        if calc_type == 'ARKU_edge':
            self.U_list = [[None] * len(self.unique_edge_labels)] * self.N  # left SVD matrix of each adj matrix
            self.Lamda_list = [[None] * len(self.unique_edge_labels)] * self.N  # eigenvalues of each adj matrix
            self.Vt_list = [[None] * len(self.unique_edge_labels)] * self.N  # right transposed SVD matrix of each adj matrix
            self.all_A = [[None] * len(self.unique_edge_labels)] * self.N


        if calc_type == "ARKL":
            self.Ls = [None] * self.N
            for i in range(self.N):
                self.Ls[i] = self._get_node_label_vectors(self.X[i], self.unique_node_labels, self.node_label)
        
        # Calculate kernel
        for i in range(self.N):
            for j in range(i,self.N):
                if calc_type == "ARKU":
                    (i,j,value) = self.fit_arku_ij(i,j,pbar)

                elif calc_type == "ARKU_plus":
                    (i,j,value) = self.fit_ARKU_plus_ij(i,j,pbar)

                elif calc_type == "ARKU_edge":
                    if self.edge_label is None:
                        raise ValueError("edge_label can not be None when using ARKU_edge")
                    if self.unique_edge_labels is None:
                        raise ValueError("unique_edge_labels can not be None when using ARKU_edge")
                    (i,j,value) = self.fit_ARKU_edge_ij(i,j,pbar)

                elif calc_type == "ARKL":
                    if self.unique_node_labels is None:
                        raise ValueError("unique_node_labels can not be None when using ARKL")
                    if self.node_label is None:
                        raise ValueError("edge_label can not be None when using ARKL")

                    (i,j,value) = self.fit_ARKL_plus_ij(i,j,pbar)

                else:
                    raise ValueError(f"{calc_type} not recognized")

                self.K[i,j] = value
                self.K[j,i] = value
                    

        if self.normalize:
            self.K = self.normalize_gram_matrix(self.K)

        if check_psd:
            v,_ = np.linalg.eigh(self.K)
            if np.any(v < -10e-12):
                raise ValueError(" Kernel not psd. Try to lower the constant c. Strange results may come from the mmd test if performed.")
        


    def fit_arku_ij(self, i,j, pbar:tqdm.tqdm = None):
        """
        Cacluate kernel value at index i,j

        Parameters
        ---------------
        i:int
        j:int
        pbar: tqdm.tqdm progressbar
        """
                
        # Row normalize the adjacency matrix
        if self.U_list[i] is None:
            W_row_normalize = self._row_normalized_adj(self.X[i])
            self.U_list[i], self.Lamda_list[i], self.Vt_list[i] = randomized_svd(W_row_normalize.T, n_components= self.r, random_state=None)

        if self.U_list[j] is None:
            W_row_normalize = self._row_normalized_adj(self.X[j])
            self.U_list[j], self.Lamda_list[j], self.Vt_list[j] = randomized_svd(W_row_normalize.T, n_components= self.r, random_state=None)

        if self.p[i] is None:
            if self.node_attr is None:
                self.p[i] = np.ones((self.X[i].number_of_nodes())) / float(self.X[i].number_of_nodes())
                self.q[i] = np.ones((self.X[i].number_of_nodes())) / float(self.X[i].number_of_nodes())
            else:
                self.p[i] = [(k[0]) for k in nx.get_node_attributes(self.X[i], self.node_attr).values()]
                self.q[i] = [(k[0]) for k in nx.get_node_attributes(self.X[i], self.node_attr).values()]
        
        if self.p[j] is None:
            if self.node_attr is None:
                self.p[j] = np.ones((self.X[j].number_of_nodes())) / float(self.X[j].number_of_nodes())
                self.q[j] = np.ones((self.X[j].number_of_nodes())) / float(self.X[j].number_of_nodes())
            else:
                self.p[j] = [(k[0]) for k in nx.get_node_attributes(self.X[j], self.node_attr).values()]
                self.q[j] = [(k[0]) for k in nx.get_node_attributes(self.X[j], self.node_attr).values()]

            

        value = self.ARKU(self.U_list[i], self.Lamda_list[i], self.Vt_list[i], self.U_list[j], self.Lamda_list[j], self.Vt_list[j], self.r, self.p[i], self.p[j], self.q[i], self.q[j])

        if pbar is not None:
            pbar.update()

        return (i,j,value)


    def ARKU(self, u1, w1, v1t, u2, w2, v2t, r, p1, p2, q1, q2):
        """
        Calculate Kernel value between G1 and G2

        Parameters
        ----------------
        u1, u2 - 2d array, Left SVD matrix of each adjacency matrix of G1, G2
        w1, w2 - 1d array, Eigenvalues of each adjacency matrix of G1, G2
        v1t, v2t - 2d array, Right transposed SVD matrix of each adjacency matrix of G1, G2
        r - int how many eigenvalues?
        p1, p2 - initial probabilities
        q1, q2 - stopping probabilities
        """

        if r < 1:
            raise ValueError('r has to 1 or bigger')


        diag_inverse =  np.kron(np.diag(np.reciprocal(w1)), np.diag(np.reciprocal(w2)))
        Lamda = inv(diag_inverse - self.c * np.matmul(np.kron(v1t, v2t), np.kron(u1, u2)))
        L = np.kron(np.matmul(q1.T, u1), np.matmul(q2.T, u2))
        R = np.kron(np.matmul(v1t, p1), np.matmul(v2t, p2))

        return np.inner(q1,p1)*np.inner(q2,p2) + self.c*np.dot(L, Lamda).dot(R)


    def fit_ARKU_plus_ij(self, i,j, pbar:tqdm.tqdm = None) -> None:
        """
        Cacluate kernel value at index i,j

        Parameters
        ---------------
        i:int
        j:int
        pbar: tqdm.tqdm progressbar
        """ 
        
        # Row normalize the adjacency matrix
        if self.Lamda_list[i] is None:
            A_i = self._get_adj_matrix(self.X[i], edge_attr = self.edge_attr)
            self.Lamda_list[i], self.U_list[i] = eigsh(A_i.T, k = self.r)
        if self.Lamda_list[j] is None:
            A_j = self._get_adj_matrix(self.X[j], edge_attr = self.edge_attr)
            self.Lamda_list[j], self.U_list[j] = eigsh(A_j.T, k = self.r)

        if self.p[i] is None:
            if self.node_attr is None:
                self.p[i] = np.ones((self.X[i].number_of_nodes())) / float(self.X[i].number_of_nodes())
                self.q[i] = np.ones((self.X[i].number_of_nodes())) / float(self.X[i].number_of_nodes())
            else:
                self.p[i] = np.array([(k[0]) for k in nx.get_node_attributes(self.X[i], self.node_attr).values()])
                self.q[i] = np.array([(k[0]) for k in nx.get_node_attributes(self.X[i], self.node_attr).values()])
        
        if self.p[j] is None:
            if self.node_attr is None:
                self.p[j] = np.ones((self.X[j].number_of_nodes())) / float(self.X[j].number_of_nodes())
                self.q[j] = np.ones((self.X[j].number_of_nodes())) / float(self.X[j].number_of_nodes())
            else:
                self.p[j] = np.array([(k[0]) for k in nx.get_node_attributes(self.X[j], self.node_attr).values()])
                self.q[j] = np.array([(k[0]) for k in nx.get_node_attributes(self.X[j], self.node_attr).values()])


        value = self.ARKU_plus(self.U_list[i], self.Lamda_list[i], self.U_list[j], self.Lamda_list[j], self.r, self.p[i], self.p[j], self.q[i], self.q[j])

        if pbar is not None:
            pbar.update()

        return (i,j,value)


    def fit_ARKU_edge_ij(self, i,j, pbar:tqdm.tqdm = None) -> None:
        """
        Cacluate kernel value at index i,j

        Parameters
        ---------------
        i:int
        j:int
        pbar: tqdm.tqdm progressbar
        """ 

        

                
        if self.Lamda_list[i][0] is None:
            self.all_A[i] = self._get_label_adj(self.X[i].copy(), edge_labels = self.unique_edge_labels, edge_attr = self.edge_attr, edge_labels_tag = self.edge_label)
            self.Lamda_list[i], self.U_list[i] = self._eigen_decomp(self.all_A[i], self.r)
        if self.Lamda_list[j][0] is None:
            self.all_A[j] = self._get_label_adj(self.X[j].copy(), edge_labels = self.unique_edge_labels, edge_attr = self.edge_attr, edge_labels_tag = self.edge_label)
            self.Lamda_list[j], self.U_list[j] = self._eigen_decomp(self.all_A[j], self.r)

        if self.p[i] is None:
            if self.node_attr is None:
                self.p[i] = np.ones((self.X[i].number_of_nodes())) / float(self.X[i].number_of_nodes())
                self.q[i] = np.ones((self.X[i].number_of_nodes())) / float(self.X[i].number_of_nodes())
            else:
                self.p[i] = np.array([(k[0]) for k in nx.get_node_attributes(self.X[i], self.node_attr).values()])
                self.q[i] = np.array([(k[0]) for k in nx.get_node_attributes(self.X[i], self.node_attr).values()])
        
        if self.p[j] is None:
            if self.node_attr is None:
                self.p[j] = np.ones((self.X[j].number_of_nodes())) / float(self.X[j].number_of_nodes())
                self.q[j] = np.ones((self.X[j].number_of_nodes())) / float(self.X[j].number_of_nodes())
            else:
                self.p[j] = np.array([(k[0]) for k in nx.get_node_attributes(self.X[j], self.node_attr).values()])
                self.q[j] = np.array([(k[0]) for k in nx.get_node_attributes(self.X[j], self.node_attr).values()])


        value = self.ARKU_edge(self.U_list[i], self.Lamda_list[i], self.p[i], self.q[i], self.U_list[j], self.Lamda_list[j], self.p[j], self.q[j] )

        if pbar is not None:
            pbar.update()

        return (i,j,value)

    
    def ARKU_edge(self, u1, w1, p1, q1, u2, w2, p2, q2 ):
        """
        Fast Random walk kernel for symmetric (weight) matrices

        Parameters
        ----------------
        u1, u2 - 2d array, eigenvector matrix of Adjacency matrix of G1, G2
        w1, w2 - 1d array, eigenvalues of Adjacency matrix of G1, G2
        p1, p2 - initial probabilities
        q1, q2 - stopping probabilities
        """

        nr_labels = len(u1)

        # Create the label eigenvalue (block) diagonal matrix
        diag_inverse = [] 
        for i in range(nr_labels):
            diag_inverse.append(np.diag(np.kron(np.diag(np.reciprocal(w1[i])), np.diag(np.reciprocal(w2[i])))))
        diag_inverse =  np.concatenate(diag_inverse)
        Lamda = np.diag(np.reciprocal(diag_inverse - self.c ))

        L = []
        for i in range(nr_labels):
            L.append(np.kron(np.matmul(q1.T, u1[i]), np.matmul(q2.T, u2[i])))
        L = np.concatenate(L)

        R = [] 
        for i in range(nr_labels):
            R.append(np.kron(np.matmul(u1[i].T, p1), np.matmul(u2[i].T, p2)))
        R = np.concatenate(R)

        return np.inner(q1,p1)*np.inner(q2,p2) + self.c*np.dot(L, Lamda).dot(R)


    def fit_ARKL_plus_ij(self, i,j, pbar:tqdm.tqdm = None) -> None:
        """
        Cacluate kernel value at index i,j

        Parameters
        ---------------
        i:int
        j:int
        pbar: tqdm.tqdm progressbar
        """ 



        if self.U_list[i] is None:
            self.U_list[i], self.Lamda_list[i], self.Vt_list[i] = randomized_svd(self._get_adj_matrix(self.X[i], edge_attr = self.edge_attr).T, n_components= self.r, random_state=None)
            self.Ls[i] = self._get_node_label_vectors(self.X[i], self.unique_node_labels, self.node_label)
        if self.U_list[j] is None:
            self.U_list[j], self.Lamda_list[j], self.Vt_list[j] = randomized_svd(self._get_adj_matrix(self.X[j], edge_attr = self.edge_attr).T, n_components= self.r, random_state=None)
            self.Ls[j] = self._get_node_label_vectors(self.X[j], self.unique_node_labels, self.node_label)


        if self.p[i] is None:
            if self.node_attr is None:
                self.p[i] = np.ones((self.X[i].number_of_nodes())) / float(self.X[i].number_of_nodes())
                self.q[i] = np.ones((self.X[i].number_of_nodes())) / float(self.X[i].number_of_nodes())
            else:
                self.p[i] = np.array([(k[0]) for k in nx.get_node_attributes(self.X[i], self.node_attr).values()])
                self.q[i] = np.array([(k[0]) for k in nx.get_node_attributes(self.X[i], self.node_attr).values()])
        
        if self.p[j] is None:
            if self.node_attr is None:
                self.p[j] = np.ones((self.X[j].number_of_nodes())) / float(self.X[j].number_of_nodes())
                self.q[j] = np.ones((self.X[j].number_of_nodes())) / float(self.X[j].number_of_nodes())
            else:
                self.p[j] = np.array([(k[0]) for k in nx.get_node_attributes(self.X[j], self.node_attr).values()])
                self.q[j] = np.array([(k[0]) for k in nx.get_node_attributes(self.X[j], self.node_attr).values()])


        value = self.ARKL(self.U_list[i], self.Lamda_list[i], self.Vt_list[i], self.U_list[j], self.Lamda_list[j], self.Vt_list[j], 
                            self.r, self.Ls[i], self.Ls[j], self.p[i], self.p[j], self.q[i], self.q[j])

        if pbar is not None:
            pbar.update()

        return (i,j,value)

    def fit_ARKL(self, r, label_list, normalize_adj = False, row_normalize_adj = False, edge_attr =None, verbose = True, label_name = 'label'):
        """
        Fit approximate label node random walk kernel.

        Parameters
        ----------------------------
        r - int, number of eigenvalues
        label_list - array with labels
        normalize_adj - bool, Should the adj matrix normalized? D^{-1/2}AD^{-1/2} where A is adj matrix and D is degree matrix.
        row_normalize_adj - bool, Should the adj matrix be row normalized? AD^{-1/2} where A is adj matrix and D is degree matrix.
        verbose - bool, print progress bar?
        label_name - str, what is the name of labels

        Returns 
        ------------------
        K - np.array, N x N, kernel matrix, N number of graphs
        
        """

        if normalize_adj and row_normalize_adj:
            raise ValueError("Can not have both row normalized and normalized adj") 

        
        all_A = [None] * self.N
        U_list = [None] * self.N  # left SVD matrix of each adj matrix
        Lamda_list = [None] * self.N  # eigenvalues of each adj matrix
        Vt_list = [None] * self.N  # Right transposed SVD matrix of each adj matrix
        K = np.zeros((self.N, self.N))

        if verbose:
            pbar = tqdm.tqdm(disable=(verbose is False), total=self.N*(self.N+1)/2)

        # get label matrix/vector of all graphs
        Ls = [None] * self.N
        for i in range(self.N):
            Ls[i] = self._get_node_label_vectors(self.X[i], label_list, label_name)

        for i in range(self.N):
            for j in range(i,self.N):


                if row_normalize_adj:
                    if all_A[i] is None:
                        all_A[i] = self._row_normalized_adj(self.X[i], edge_attr = edge_attr)
                        U_list[i], Lamda_list[i], Vt_list[i] = randomized_svd(all_A[i].T, n_components= r)
                    if all_A[j] is None:
                        all_A[j] = self._row_normalized_adj(self.X[j], edge_attr = edge_attr)
                        U_list[j], Lamda_list[j], Vt_list[j] = randomized_svd(all_A[j].T, n_components= r)
                else:
                    if all_A[i] is None:
                        all_A[i] = self._get_adj_matrix(self.X[i], edge_attr = edge_attr)
                        U_list[i], Lamda_list[i], Vt_list[i] = randomized_svd(all_A[i].T, n_components= r)
                    if all_A[j] is None:
                        all_A[j] = self._get_adj_matrix(self.X[j], edge_attr = edge_attr)
                        U_list[j], Lamda_list[j], Vt_list[j] = randomized_svd(all_A[j].T, n_components= r)
                


                p1 = np.ones((self.X[i].number_of_nodes())) / float(self.X[i].number_of_nodes())
                p2 = np.ones((self.X[j].number_of_nodes())) / float(self.X[j].number_of_nodes())
                q1 = np.ones((self.X[i].number_of_nodes())) / float(self.X[i].number_of_nodes())
                q2 = np.ones((self.X[j].number_of_nodes())) / float(self.X[j].number_of_nodes())
    

                K[i,j] = self.ARKL(U_list[i], Lamda_list[i], Vt_list[i], U_list[j], Lamda_list[j], Vt_list[j], r, Ls[i], Ls[j], p1, p2, q1, q2)

                if verbose:
                    pbar.update()

        if verbose:
            pbar.close()
            
        K = np.triu(K) + np.triu(K, 1).T

        if self.normalize:
            K = self.normalize_gram_matrix(K)

        return K


    def ARKU_plus(self, u1, w1, u2, w2, r, p1, p2, q1, q2):
        """
        Fast Random walk kernel for symmetric (weight) matrices

        Parameters
        ----------------
        u1, u2 - 2d array, eigenvector matrix of Adjacency matrix of G1, G2
        w1, w2 - 1d array, eigenvalues of Adjacency matrix of G1, G2
        r - int how many eigenvalues?
        p1, p2 - initial probabilities
        q1, q2 - stoppoing probabilities
        """

        if r < 1:
            raise ValueError('r has to 1 or bigger')
 
        diag_inverse =  np.kron(np.diag(np.reciprocal(w1)), np.diag(np.reciprocal(w2)))
        Lamda = inv(diag_inverse - self.c * np.identity(diag_inverse.shape[0]))
        L = np.kron(np.matmul(q1.T, u1), np.matmul(q2.T, u2))
        R = np.kron(np.matmul(u1.T, p1), np.matmul(u2.T, p2))

        return np.inner(q1,p1)*np.inner(q2,p2) + self.c*np.dot(L, Lamda).dot(R)

    def ARKL(self, u1, w1, v1t, u2, w2, v2t, r, L1, L2, p1, p2, q1, q2):
        """
        Fit an approximation to node labeled graphs

        Parameters
        -------------------------
        u1, u2 - 2d array, Left SVD matrix of each adjacency matrix of G1, G2
        w1, w2 - 1d array, Eigenvalues of each adjacency matrix of G1, G2
        v1t, v2t - 2d array, Right transposed SVD matrix of each adjacency matrix of G1, G2
        r - int how many eigenvalues?
        L1, L2 - list containing vectors. Each vector corresponds to a label and a element nr i is 1 if node i has the label 0 otherwise
        p1, p2 - initial probabilities
        q1, q2 - stoppoing probabilities
        """

        if r < 1:
            raise ValueError('r has to 1 or bigger')


        diag_inverse =  np.kron(np.diag(np.reciprocal(w1)), np.diag(np.reciprocal(w2)))
        Lamda = inv(diag_inverse - self.c * np.kron(np.matmul(np.matmul(v1t, np.diag(np.sum(L1, axis=0))), u1), np.matmul(np.matmul(v2t, np.diag(np.sum(L2, axis=0))), u2)))
        L = np.sum([np.kron(np.matmul(np.matmul(q1.T, np.diag(L1[i])), u1), np.matmul(np.matmul(q2.T, np.diag(L2[i])), u2)) for i in range(len(L1))], axis=0)
        R = np.sum([np.kron(np.matmul(np.matmul(v1t, np.diag(L1[i])), p1), np.matmul(np.matmul(v2t, np.diag(L2[i])), p2)) for i in range(len(L1))], axis=0)

        return np.inner(q1,p1)*np.inner(q2,p2) + self.c*np.dot(L, Lamda).dot(R)

    def _row_normalized_adj(self, G):
        """
        Get row normalized adjacency matrix

        Parameters
        --------------------------
        G - networkx graph

        Returns
        ---------------
        sparse csr matrix

        """

        # A = nx.linalg.adjacency_matrix(G, dtype = float)
        A = nx.adjacency_matrix(G ,weight=self.edge_attr)# scipy.sparse.csr_matrix(nx.adjacency_matrix(G ,weight=edge_attr), dtype=np.float64)
        if type(self.X[0]) == nx.classes.digraph.DiGraph:
            D_inv = scipy.sparse.dia_matrix(([zero_div(1.0, d[1]) for d in G.out_degree()], 0), shape = (A.shape[0], A.shape[0]))
        else:
            D_inv = scipy.sparse.dia_matrix(([zero_div(1.0, d[1]) for d in G.degree()], 0), shape = (A.shape[0], A.shape[0]))

        return A.dot(D_inv)




    def _get_adj_matrix(self, G, edge_attr = None):
        """
        Get adjacency matrix

        Parameters
        --------------------------
        G - networkx graph

        """
        return scipy.sparse.csr_matrix(nx.adjacency_matrix(G ,weight=edge_attr), dtype=np.float64)

    def _get_node_label_vectors(self, G, label_list, label_name = 'label'):
        """
        Get node label vectors

        1-D arrays are returned as we only need the diagonal information

        Parameters
        ---------------
        G - networkx graph
        label_list - list with all potential labels
        label_name - str, label name


        Returns
        -------------------------
        L - list of 1-d arrays

        
        """

        L = [] 

        for idx, label in enumerate(label_list):

            get_nodes_with_label = np.array(list(nx.get_node_attributes(G, label_name).values())) == label

            L.append(np.array(get_nodes_with_label, dtype = np.float64))

        return L


    def _get_label_adj(self, G, edge_labels, edge_labels_tag = 'sign', edge_attr = 'weight'):
        """

        Filter the adjacency matrix according to each label in edge_labels, w if edges have same label, 0 otherwise

        Parameter
        -------------------------
        G - networkx graph
        edge_labels - array with the label vocabulary
        edge_labels_tag - name of the edge label
        edge_attr - str, edge weight. None assumes binary


        Returns
        --------------
        A - list of sparse matrices representing filtered adjacency matrices according to the labels in edge_labels
        
        """

        edge_attrs = nx.get_edge_attributes(G, edge_labels_tag)

        A = [None] * len(edge_labels)  # Store filtered adjacency matrices 

        edges = [k for k, v in edge_attrs.items() if v == 1]
        edges
        for idx, label in enumerate(edge_labels):
            G_tmp = G.copy()
            for k, v in edge_attrs.items():
                if v == label:
                    G_tmp.remove_edge(k[0], k[1])

            A[idx] = scipy.sparse.csr_matrix(nx.adjacency_matrix(G_tmp, weight=edge_attr), dtype=np.float64)
        
        return A

    def _eigen_decomp(self, A, r):
        """
        Perform r eigenvalue decomposition on each matrix in A
        
        """

        nr_A = len(A)

        U = [None] * nr_A
        w = [None] * nr_A

        for i in range(nr_A):
            w[i], U[i] = eigsh(A[i].T, k = r)

        return w, U


    @staticmethod
    def normalize_gram_matrix(x):
        k = np.reciprocal(np.sqrt(np.diag(x)))
        k = np.resize(k, (len(k), 1))
        return np.multiply(x, np.outer(k,k))










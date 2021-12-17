"""

fast Random walk kernels: See http://www.cs.cmu.edu/~ukang/papers/fast_rwgk.pdf
"""


from networkx.classes.function import get_node_attributes
import numpy as np
from numpy.linalg import eigh, inv
from numpy.random import exponential
from sklearn.utils.extmath import randomized_svd
from scipy.sparse.linalg import eigsh
from scipy.sparse.linalg import inv as sparse_inv
from scipy.sparse import kron as sparse_kron
import scipy
import networkx as nx
import tqdm
from scipy.sparse import kron


class RandomWalk():


    def __init__(self, X, c, p = None, q = None) -> None:
        """
        Parameters
        ---------------
        X: list of size N, of networkx graphs
        p: list of size size N containing initial probabilities for each graph. If None then a uniform probabilites are used.
        p: list of size size N containing stopping probabilities for each graph. If None then a uniform probabilites are used.
        c: scalar
        
        """

        self.X = X
        self.p = p
        self.q = q
        self.c = c

        self.N = len(X)
    

    def fit_ARKU(self, r, verbose = True):
        """
        Approximate random walk kernel for unlabeled nodes and asymmetric W where W is the (weighted) adjacency matrix.

        Parameters
        --------------------------
        r - int, number of eigenvalues used in approximation
        verbose - bool, print progress bar?
        """



        W_row_normalize = [None] * self.N
        K = np.zeros((self.N, self.N))

        if verbose:
            pbar = tqdm.tqdm(disable=(verbose is False), total=self.N*(self.N+1)/2)
        #disable=(disp is False), total=(13 * len(_null_models) * len(mat_types))


        for i in range(self.N):
            for j in range(i,self.N):
                
                # Row normalize the adjacency matrix
                if W_row_normalize[i] is None:
                    W_row_normalize[i] = self._row_normalized_adj(self.X[i])
                if W_row_normalize[j] is None:
                    W_row_normalize[j] = self._row_normalized_adj(self.X[j])

                if (self.p is None) and (self.q is None):
                    p1 = np.ones((self.X[i].number_of_nodes())) / float(self.X[i].number_of_nodes())
                    p2 = np.ones((self.X[j].number_of_nodes())) / float(self.X[j].number_of_nodes())
                    q1 = np.ones((self.X[i].number_of_nodes())) / float(self.X[i].number_of_nodes())
                    q2 = np.ones((self.X[j].number_of_nodes())) / float(self.X[j].number_of_nodes())
                    K[i,j] = self.ARKU(W_row_normalize[i].T, W_row_normalize[j].T, r, p1, p2, q1, q2)
                elif (self.p is None) and ~(self.q is None):
                    p1 = np.ones((self.X[i].number_of_nodes())) / float(self.X[i].number_of_nodes())
                    p2 = np.ones((self.X[j].number_of_nodes())) / float(self.X[j].number_of_nodes())
                    K[i,j] = self.ARKU(W_row_normalize[i].T, W_row_normalize[j].T, r, p1, p2, self.q[i], self.q[j])
                elif ~(self.p is None) and (self.q is None):
                    q1 = np.ones((self.X[i].number_of_nodes())) / float(self.X[i].number_of_nodes())
                    q2 = np.ones((self.X[j].number_of_nodes())) / float(self.X[j].number_of_nodes())
                    K[i,j] = self.ARKU(W_row_normalize[i].T, W_row_normalize[j].T, r, self.p[i], self.p[j], q1, q2)
                else:
                    K[i,j] = self.ARKU(W_row_normalize[i].T, W_row_normalize[j].T, r, self.p[i], self.p[j], self.q[i], self.q[j])

                if verbose:
                    pbar.update()


        pbar.close()
        K = K + K.T
        np.fill_diagonal(K, np.diag(K) / 2)


        return K


    def fit_ARKU_plus(self, r, normalize_adj = False, verbose = True):
        """
        Approximate random walk kernel for unlabeled nodes and asymmetric W where W is the (weighted) adjacency matrix.

        Parameters
        --------------------------
        r - int, number of eigenvalues used in approximation
        verbose - bool, print progress bar?
        normalize_adj - bool, Should the adj matrix normalized? D^{-1/2}AD^{-1/2} where A is adj matrix and D is degree matrix.

        """
        all_A = [None] * self.N

        K = np.zeros((self.N, self.N))

        if verbose:
            pbar = tqdm.tqdm(disable=(verbose is False), total=self.N*(self.N+1)/2)
        #disable=(disp is False), total=(13 * len(_null_models) * len(mat_types))


        for i in range(self.N):
            for j in range(i,self.N):

                if normalize_adj:
                    if all_A[i] is None:
                        all_A[i] = self._normalized_adj(self.X[i])
                    if all_A[j] is None:
                        all_A[j] = self._normalized_adj(self.X[j])
                else:
                    if all_A[i] is None:
                        all_A[i] = self._get_adj_matrix(self.X[i])
                    if all_A[j] is None:
                        all_A[j] = self._get_adj_matrix(self.X[j])


                if (self.p is None) and (self.q is None):
                    p1 = np.ones((self.X[i].number_of_nodes())) / float(self.X[i].number_of_nodes())
                    p2 = np.ones((self.X[j].number_of_nodes())) / float(self.X[j].number_of_nodes())
                    q1 = np.ones((self.X[i].number_of_nodes())) / float(self.X[i].number_of_nodes())
                    q2 = np.ones((self.X[j].number_of_nodes())) / float(self.X[j].number_of_nodes())
                    K[i,j] = self.ARKU_plus(all_A[i].T, all_A[j].T, r, p1, p2, q1, q2)
                elif (self.p is None) and ~(self.q is None):
                    p1 = np.ones((self.X[i].number_of_nodes())) / float(self.X[i].number_of_nodes())
                    p2 = np.ones((self.X[j].number_of_nodes())) / float(self.X[j].number_of_nodes())
                    K[i,j] = self.ARKU(all_A[i].T, all_A[j].T, r, p1, p2, self.q[i], self.q[j])
                elif ~(self.p is None) and (self.q is None):
                    q1 = np.ones((self.X[i].number_of_nodes())) / float(self.X[i].number_of_nodes())
                    q2 = np.ones((self.X[j].number_of_nodes())) / float(self.X[j].number_of_nodes())
                    K[i,j] = self.ARKU(all_A[i].T, all_A[j].T, r, self.p[i], self.p[j], q1, q2)
                else:
                    K[i,j] = self.ARKU(all_A[i].T, all_A[j].T, r, self.p[i], self.p[j], self.q[i], self.q[j])

                if verbose:
                    pbar.update()


        pbar.close()
        K = K + K.T
        np.fill_diagonal(K, np.diag(K) / 2)
        
        return K


    def fit_ARKL(self, r, label_list, normalize_adj = False, row_normalize_adj = False, verbose = True, label_name = 'label'):
        """
        Fit approximate label node random walk kernel.

        Parameters
        ----------------------------
        r - int, number of eigenvalues
        label_list - array with labels
        normalize_adj - bool, Should the adj matrix normalized? D^{-1/2}AD^{-1/2} where A is adj matrix and D is degree matrix.
        row_normalize_adj - bool, Should the adj matrixb be row normalized? D^{-1/2}AD^{-1/2} where A is adj matrix and D is degree matrix.
        verbose - bool, print progress bar?
        label_name - str, what is the name of labels

        Returns 
        ------------------
        K - np.array, N x N, kernel matrix, N number of graphs
        
        """

        if normalize_adj and row_normalize_adj:
            raise ValueError("Can not have both row normalized and normalized adj") 

        
        all_A = [None] * self.N
        K = np.zeros((self.N, self.N))

        if verbose:
            pbar = tqdm.tqdm(disable=(verbose is False), total=self.N*(self.N+1)/2)
        #disable=(disp is False), total=(13 * len(_null_models) * len(mat_types))

        # get label matrix/vector of all graphs
        Ls = [None] * self.N
        for i in range(self.N):
            Ls[i] = self._get_node_label_vectors(self.X[i], label_list, label_name)

        for i in range(self.N):
            for j in range(i,self.N):

                if normalize_adj:
                    if all_A[i] is None:
                        all_A[i] = self._normalized_adj(self.X[i])
                    if all_A[j] is None:
                        all_A[j] = self._normalized_adj(self.X[j])
                elif row_normalize_adj:
                    if all_A[i] is None:
                        all_A[i] = self._row_normalized_adj(self.X[i])
                    if all_A[j] is None:
                        all_A[j] = self._row_normalized_adj(self.X[j])
                else:
                    if all_A[i] is None:
                        all_A[i] = self._get_adj_matrix(self.X[i])
                    if all_A[j] is None:
                        all_A[j] = self._get_adj_matrix(self.X[j])
                

                if (self.p is None) and (self.q is None):
                    p1 = np.ones((self.X[i].number_of_nodes())) / float(self.X[i].number_of_nodes())
                    p2 = np.ones((self.X[j].number_of_nodes())) / float(self.X[j].number_of_nodes())
                    q1 = np.ones((self.X[i].number_of_nodes())) / float(self.X[i].number_of_nodes())
                    q2 = np.ones((self.X[j].number_of_nodes())) / float(self.X[j].number_of_nodes())
                    K[i,j] = self.ARKL(all_A[i].T, all_A[j].T, Ls[i], Ls[j], p1, p2, q1, q2, r)
                elif (self.p is None) and ~(self.q is None):
                    p1 = np.ones((self.X[i].number_of_nodes())) / float(self.X[i].number_of_nodes())
                    p2 = np.ones((self.X[j].number_of_nodes())) / float(self.X[j].number_of_nodes())
                    K[i,j] = self.ARKL(all_A[i].T, all_A[j].T, Ls[i], Ls[j], p1, p2, self.q[i], self.q[j], r)
                elif ~(self.p is None) and (self.q is None):
                    q1 = np.ones((self.X[i].number_of_nodes())) / float(self.X[i].number_of_nodes())
                    q2 = np.ones((self.X[j].number_of_nodes())) / float(self.X[j].number_of_nodes())
                    K[i,j] = self.ARKL(all_A[i].T, all_A[j].T, Ls[i], Ls[j], self.p[i], self.p[j], q1, q2, r)
                else:
                    K[i,j] = self.ARKL(all_A[i].T, all_A[j].T, Ls[i], Ls[j], self.p[i], self.p[j], self.q[i], self.q[j], r)

                if verbose:
                    pbar.update()

        pbar.close()
        K = K + K.T
        np.fill_diagonal(K, np.diag(K) / 2)

        return K

    def fit_exponential(self, r = None, normalize_adj = False, row_normalize_adj = False, verbose = True):
        """
        Perform an infnite exponential random walk. Does not work for labelled graphs

        Parameters
        ---------------------------------------
        r - int, number of eigenvalues, if None full eigenvalue decomposition used which will be slow.
        normalize_adj - bool, Should the adj matrix normalized? D^{-1/2}AD^{-1/2} where A is adj matrix and D is degree matrix.
        row_normalize_adj - bool, Should the adj matrixb be row normalized? D^{-1/2}AD^{-1/2} where A is adj matrix and D is degree matrix.
        verbose - bool, print progress bar?
        
        Returns
        --------------------------------
        K - N n N kernel matrix 

        """

        if normalize_adj and row_normalize_adj:
            raise ValueError("Can not have both row normalized and normalized adj") 

        
        all_A = [None] * self.N
        K = np.zeros((self.N, self.N))

        if verbose:
            pbar = tqdm.tqdm(disable=(verbose is False), total=self.N*(self.N+1)/2)
        #disable=(disp is False), total=(13 * len(_null_models) * len(mat_types))


        for i in range(self.N):
            for j in range(i,self.N):

                if normalize_adj:
                    if all_A[i] is None:
                        all_A[i] = self._normalized_adj(self.X[i])
                    if all_A[j] is None:
                        all_A[j] = self._normalized_adj(self.X[j])
                elif row_normalize_adj:
                    if all_A[i] is None:
                        all_A[i] = self._row_normalized_adj(self.X[i])
                    if all_A[j] is None:
                        all_A[j] = self._row_normalized_adj(self.X[j])
                else:
                    if all_A[i] is None:
                        all_A[i] = self._get_adj_matrix(self.X[i])
                    if all_A[j] is None:
                        all_A[j] = self._get_adj_matrix(self.X[j])
                

                if (self.p is None) and (self.q is None):
                    p1 = np.ones((self.X[i].number_of_nodes())) / float(self.X[i].number_of_nodes())
                    p2 = np.ones((self.X[j].number_of_nodes())) / float(self.X[j].number_of_nodes())
                    q1 = np.ones((self.X[i].number_of_nodes())) / float(self.X[i].number_of_nodes())
                    q2 = np.ones((self.X[j].number_of_nodes())) / float(self.X[j].number_of_nodes())
                    K[i,j] = self.rw_exponential(all_A[i].T, all_A[j].T, p1, p2, q1, q2, r )
                elif (self.p is None) and ~(self.q is None):
                    p1 = np.ones((self.X[i].number_of_nodes())) / float(self.X[i].number_of_nodes())
                    p2 = np.ones((self.X[j].number_of_nodes())) / float(self.X[j].number_of_nodes())
                    K[i,j] = self.rw_exponential(all_A[i].T, all_A[j].T, p1, p2, self.q[i], self.q[i], r )
                elif ~(self.p is None) and (self.q is None):
                    q1 = np.ones((self.X[i].number_of_nodes())) / float(self.X[i].number_of_nodes())
                    q2 = np.ones((self.X[j].number_of_nodes())) / float(self.X[j].number_of_nodes())
                    K[i,j] = self.rw_exponential(all_A[i].T, all_A[j].T, self.p[i], self.p[j], q1, q2, r )
                else:
                    K[i,j] = self.rw_exponential(all_A[i].T, all_A[j].T, self.p[i], self.p[j], self.q[i], self.q[i], r )

                if verbose:
                    pbar.update()

        pbar.close()
        K = K + K.T
        np.fill_diagonal(K, np.diag(K) / 2)

        return K

    def rw_exponential(self, A1, A2, p1, p2, q1, q2, r):
        """
        Perform p-random walks. Symmetric matrix

        Parameters
        ---------------------------------------
        A1 - np array representing weight/adj matrix
        A2 - np array representing weight/adj matrix
        p1, p2 - initial probabilities
        q1, q2 - stopping probabilities
        r - int, If passed then an approximation is used by eigen decomposition
        
        Returns
        --------------------------------
        float kernel value between W1,W2

        """

        if (r is not None):
            if r < 1:
                raise ValueError('r has to 1 or bigger')

        if r is None:
            p1 = np.expand_dims(p1, axis = 1)
            p2 = np.expand_dims(p2, axis = 1)
            q1 = np.expand_dims(q1, axis = 1)
            q2 = np.expand_dims(q2, axis = 1)
            w1, u1 = eigh(A1)
            w2, u2 = eigh(A2)
        else:
            w1, u1 = eigsh(A1, k = r)
            w2, u2 = eigsh(A2, k = r)

        w = np.array(np.concatenate([w*w2 for w in w1]))

        stop_part = np.kron(np.matmul(q1.T, u1), np.matmul(q2.T, u2))
        start_part = np.kron(np.matmul(u1.T, p1), np.matmul(u2.T, p2))
        return np.matmul(np.matmul(stop_part, np.diag(np.exp(w))), start_part)



    def fit_random_walk(self, mu_vec, k, r = None, normalize_adj = False, row_normalize_adj = False, verbose = True):
        """
        Perform p-random walks. Symmetric matrix

        Parameters
        ---------------------------------------
        k - int, Nr. random walks
        mu_vec - array of size p, containing RW weight/discount.
        r - int, number of eigenvalues, if None full eigenvalue decomposition used which will be slow.
        normalize_adj - bool, Should the adj matrix normalized? D^{-1/2}AD^{-1/2} where A is adj matrix and D is degree matrix.
        row_normalize_adj - bool, Should the adj matrixb be row normalized? D^{-1/2}AD^{-1/2} where A is adj matrix and D is degree matrix.
        verbose - bool, print progress bar?
        
        Returns
        --------------------------------
        K - N n N kernel matrix 

        """

        if normalize_adj and row_normalize_adj:
            raise ValueError("Can not have both row normalized and normalized adj") 

        
        all_A = [None] * self.N
        K = np.zeros((self.N, self.N))

        if verbose:
            pbar = tqdm.tqdm(disable=(verbose is False), total=self.N*(self.N+1)/2)
        #disable=(disp is False), total=(13 * len(_null_models) * len(mat_types))


        for i in range(self.N):
            for j in range(i,self.N):

                if normalize_adj:
                    if all_A[i] is None:
                        all_A[i] = self._normalized_adj(self.X[i])
                    if all_A[j] is None:
                        all_A[j] = self._normalized_adj(self.X[j])
                elif row_normalize_adj:
                    if all_A[i] is None:
                        all_A[i] = self._row_normalized_adj(self.X[i])
                    if all_A[j] is None:
                        all_A[j] = self._row_normalized_adj(self.X[j])
                else:
                    if all_A[i] is None:
                        all_A[i] = self._get_adj_matrix(self.X[i])
                    if all_A[j] is None:
                        all_A[j] = self._get_adj_matrix(self.X[j])
                

                if (self.p is None) and (self.q is None):
                    p1 = np.ones((self.X[i].number_of_nodes())) / float(self.X[i].number_of_nodes())
                    p2 = np.ones((self.X[j].number_of_nodes())) / float(self.X[j].number_of_nodes())
                    q1 = np.ones((self.X[i].number_of_nodes())) / float(self.X[i].number_of_nodes())
                    q2 = np.ones((self.X[j].number_of_nodes())) / float(self.X[j].number_of_nodes())
                    K[i,j] = self.p_rw_symmetric(all_A[i], all_A[j], k, mu_vec, p1, p2, q1, q2, r )
                elif (self.p is None) and ~(self.q is None):
                    p1 = np.ones((self.X[i].number_of_nodes())) / float(self.X[i].number_of_nodes())
                    p2 = np.ones((self.X[j].number_of_nodes())) / float(self.X[j].number_of_nodes())
                    K[i,j] = self.p_rw_symmetric(all_A[i], all_A[j], k, mu_vec, p1, p2, self.q[i], self.q[i], r )
                elif ~(self.p is None) and (self.q is None):
                    q1 = np.ones((self.X[i].number_of_nodes())) / float(self.X[i].number_of_nodes())
                    q2 = np.ones((self.X[j].number_of_nodes())) / float(self.X[j].number_of_nodes())
                    K[i,j] = self.p_rw_symmetric(all_A[i], all_A[j], k, mu_vec, self.p[i], self.p[j], q1, q2, r )
                else:
                    K[i,j] = self.p_rw_symmetric(all_A[i], all_A[j], k, mu_vec, self.p[i], self.p[j], self.q[i], self.q[i], r )

                if verbose:
                    pbar.update()

        pbar.close()
        K = K + K.T
        np.fill_diagonal(K, np.diag(K) / 2)

        return K


    def p_rw_symmetric(self, W1, W2, k, mu_vec, p1, p2, q1, q2, r = None):
        """
        Perform p-random walks. Symmetric matrix

        Parameters
        ---------------------------------------
        W1 - np array representing weight/adj matrix
        W2 - np array representing weight/adj matrix
        k - int, Nr. random walks
        mu_vec - array of size p, containing RW weight/discount.
        p1, p2 - initial probabilities
        q1, q2 - stoppoing probabilities
        r - int, If passed then an approximation is used by eigen decomposition

        
        Returns
        --------------------------------
        float kernel value between W1,W2

        """

        if (r is not None):
            if r < 1:
                raise ValueError('r has to 1 or bigger')

        if k < 1:
            raise ValueError('k has to 1 or bigger')

        if r is None:
            p1 = np.expand_dims(p1, axis = 1)
            p2 = np.expand_dims(p2, axis = 1)
            q1 = np.expand_dims(q1, axis = 1)
            q2 = np.expand_dims(q2, axis = 1)
            w1, u1 = eigh(W1)
            w2, u2 = eigh(W2)
        else:
            w1, u1 = eigsh(W1, k = r)
            w2, u2 = eigsh(W2, k = r)

        w = np.array(np.concatenate([w*w2 for w in w1]))

        D = np.ones(shape=(len(w)))*mu_vec[0]

        for i in range(1,k+1):
            D = D + np.power(w,i)*mu_vec[i]

        stop_part = np.kron(np.matmul(q1.T, u1), np.matmul(q2.T, u2))
        start_part = np.kron(np.matmul(u1.T, p1), np.matmul(u2.T, p2))
        return np.matmul(np.matmul(stop_part, np.diag(D)), start_part)



    def ARKU(self, W1, W2, r, p1, p2, q1, q2):

        """
        Calculate Kernel value between G1 and G2

        Parameters
        ----------------
        G1, G2 - np array representing the Adjacency matrix. Can be weighted and/or directed
        r - int how many eigenvalues?
        p1, p2 - initial probabilities
        q1, q2 - stoppoing probabilities
        """

        if r < 1:
            raise ValueError('r has to 1 or bigger')

        u1, w1, v1t = randomized_svd(W1, n_components= r)
        u2, w2, v2t = randomized_svd(W2, n_components= r)
        diag_inverse =  np.kron(np.diag(np.reciprocal(w1)), np.diag(np.reciprocal(w2)))
        Lamda = inv(diag_inverse - self.c * np.matmul(np.kron(v1t, v2t), np.kron(u1, u2)))
        L = np.kron(np.matmul(q1.T, u1), np.matmul(q2.T, u2))
        R = np.kron(np.matmul(v1t, p1), np.matmul(v2t, p2))

        return np.inner(q1,p1)*np.inner(q2,p2) + self.c*np.dot(L, Lamda).dot(R)


    def ARKU_plus(self, A1, A2, r, p1, p2, q1, q2):
        """
        Fast Random walk kernel for symmetric (weight) matrices

        Parameters
        ----------------
        A1, A2 - np array representing the Adjacency matrix
        r - int how many eigenvalues?
        p1, p2 - initial probabilities
        q1, q2 - stoppoing probabilities
        """

        if r < 1:
            raise ValueError('r has to 1 or bigger')
 
        w1, u1 = eigsh(A1, k = r)
        w2, u2 = eigsh(A2, k = r)

        # w1_diag = scipy.sparse.dia_matrix((np.reciprocal(w1), 0), shape = (len(w1), len(w1)))
        # w2_diag = scipy.sparse.dia_matrix((np.reciprocal(w2), 0), shape = (len(w2), len(w2)))
        # diag_inverse = sparse_kron(w1_diag, w2_diag)
        # Lamda = diag_inverse - self.c * scipy.sparse.identity(diag_inverse.shape[0])
        # Lamda = sparse_inv(Lamda)
        diag_inverse =  np.kron(np.diag(np.reciprocal(w1)), np.diag(np.reciprocal(w2)))
        diag_inverse = diag_inverse - self.c * np.identity(diag_inverse.shape[0])
        Lamda = np.diag(np.reciprocal(np.diag(diag_inverse)))
        L = np.kron(np.matmul(q1.T, u1), np.matmul(q2.T, u2))
        R = np.kron(np.matmul(u1.T, p1), np.matmul(u1.T, p2))

        return np.inner(q1,p1)*np.inner(q2,p2) + self.c*np.dot(L, Lamda).dot(R)

    def ARKL(self, W1, W2, L1, L2, p1, p2, q1, q2, r):
        """

        Fit an approximation to node labeled graphs

        
        Parameters
        -------------------------
        W1, W2 - np array representing the Adjacency matrix
        L1, L2 - list containing vectors. Each vector corresponds to a label and a element nr i is 1 if node i has the label 0 otherwise
        r - int how many eigenvalues?
        p1, p2 - initial probabilities
        q1, q2 - stoppoing probabilities

        
        """

        if r < 1:
            raise ValueError('r has to 1 or bigger')

        u1, w1, v1t = randomized_svd(W1, n_components= r)
        u2, w2, v2t = randomized_svd(W2, n_components= r)
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

        """

        A = nx.linalg.adjacency_matrix(G, dtype = float)
        if type(self.X[0]) == nx.classes.digraph.DiGraph:
            D_inv = scipy.sparse.dia_matrix(([1/float(d[1]) for d in G.out_degree()], 0), shape = (A.shape[0], A.shape[0]))
        else:
            D_inv = scipy.sparse.dia_matrix(([1/float(d[1]) for d in G.degree()], 0), shape = (A.shape[0], A.shape[0]))

        return A.dot(D_inv)


    def _normalized_adj(self, G):
        """
        Get normalized adjacency matrix

        Parameters
        --------------------------
        G - networkx graph

        """

        A = nx.linalg.adjacency_matrix(G, dtype = float)
        D_sq_inv = scipy.sparse.dia_matrix(([1/ np.sqrt(float(d[1])) for d in G.degree()], 0), shape = (A.shape[0], A.shape[0]))

        return D_sq_inv.dot(A).dot(D_sq_inv)

    def _get_adj_matrix(self, G):
        """
        Get adjacency matrix

        Parameters
        --------------------------
        G - networkx graph

        """

        return nx.linalg.adjacency_matrix(G, dtype = np.float64)

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










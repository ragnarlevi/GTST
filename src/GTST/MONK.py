# Code taken from https://bitbucket.org/TimotheeMathieu/monk-mmd/src/master/, see https://arxiv.org/abs/1802.04784


import numpy as np
import inspect
import warnings
from numpy.linalg import eigh


def my_cholesky(K):
    """
    Cholesky decomposition of matrices, that are ill-conditioned
    """

    l, u = eigh(K)
    l[ l<=0] = 0
    _, r = np.linalg.qr(np.dot(u, np.diag(np.sqrt(l))).T)

    return r.T



class MMD_MONK():

    ''' Estimate the MMD (maximum mean discrepancy) using MOM (median of means) estimate. The result is robust to outliers.

    Parameters
    ----------

    Q : int, odd
        number of blocks to use for MOM. Usually, the larger Q is, the faster the algorithm will be and the more robust it will be. 
        Conversely, if Q is large, the efficiency of the estimation will suffer.

    K :  np.array,
        Kernel matrix
    
    maxiter : int, default = 100
        number of iterations.

    epsilon : float, default=1e-5
        coefficient of the L2 regularisation used in order to make the Gram matrix inversible
    
    perte : boolean, default = False
        wether or not to save the mmd at each iteration in the list self.pertes

    beta : float, default=1e-3
        parameter for the step size for gradient method.

    solver : string in {'BCD'}
        solver to use to estimator MMD. Currently only BCD included
    

    Returns 
    -------
    
    float, estimate of MMD.
    
    '''
    


    def __init__(self,Q=11,K=None,maxiter=100,epsilon=1e-5,perte=False,beta=0.001, solver = 'BCD'):

        # Even so that the median can be calculated by picking the middle point
        if Q % 2 == 0:
            raise ValueError('Q should be odd')
        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")
        for arg, val in values.items():
            setattr(self, arg, val)

    def estimate(self,y1,y2):
        if self.solver == 'BCD' :
            return self._estimate_bcd(y1,y2)
        else:
            raise ValueError('There is no solver with that name implemented.')

    @staticmethod
    def blockMOM(Q,n):
        # Code taken from https://bitbucket.org/TimotheeMathieu/monk-mmd/src/master/, see https://arxiv.org/abs/1802.04784

        '''Sample the indices of K blocks for data x using a random permutation

        Parameters
        ----------

        Q : int
            number of blocks

        n : aint, length = n_sample
            The size of the sample we want to do blocks for.

        Returns 
        -------

        list of size K containing the lists of the indices of the blocks, the size of the lists are contained in [n_sample/K,2n_sample/K]
        '''
        b=int(np.floor(n/Q))
        nb=Q-(n-b*Q)
        nbpu=n-b*Q
        perm=np.random.permutation(n)
        blocks=[[(b+1)*g+f for f in range(b+1) ] for g in range(nbpu)]
        blocks+=[[nbpu*(b+1)+b*g+f for f in range(b)] for g in range(nb)]
        return [perm[b] for  b in blocks]


    def _estimate_bcd(self,n1,n2):
        '''
        Estimate MONK using a block coardinate descent

        Parameters
        ----------
        n1: int.
            number of observations in sample 1
        n2: int
            number of observations in sample 2

        Returns
        ----------
        float,
            robust estimate of MMD
        
        '''


        l = my_cholesky(self.K).T
        c = np.ones(n1 + n2)
        c = c/np.sqrt(c.dot(self.K).dot(c))

        self.pertes=[]
        for t in range(self.maxiter):
            pas=1/(1+self.beta*t)

            # Sample the blocks
            if n1 == n2:
                blocks1 = self.blockMOM(self.Q, n1)
                blocks2 = blocks1.copy()
            else:
                blocks1 = self.blockMOM(self.Q, n1)
                blocks2 = self.blockMOM(self.Q, n2)
        
            # Compute the block which realize the median of the cost
            perte = [np.sum(self.K[blocks1[q]]/len(blocks1[q]),axis=0).dot(c)-np.sum(self.K[blocks2[q]+n1]/len(blocks2[q]),axis=0).dot(c) for q in range(self.Q)]
            if self.perte:
                self.pertes+=[np.median(perte)]
            bmed = np.argsort(np.array(perte).ravel())[int(np.floor(self.Q/2))]

            # do the maximization on the median block
            c_tilde = np.zeros(n1+n2)
            c_tilde[blocks1[bmed]] = 1
            c_tilde[blocks2[bmed]+n1] = -1
            c_tilde = c_tilde/np.linalg.norm( l[:,np.hstack([blocks1[bmed],blocks2[bmed]+n1])].dot(np.hstack([np.ones(len(blocks1[bmed])),-np.ones(len(blocks2[bmed]))])))
            c =  c + pas*(c_tilde-c)

        return perte[bmed]

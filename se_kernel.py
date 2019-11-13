import numpy as np
import scipy.linalg as spl
from collections import namedtuple

pars = namedtuple('se_pars', ['std_dev', 'cor_len', 'std_dev_noise'])

class SEKernel(object):

    def __init__(self, std_dev=1.0, cor_len=0.15, std_dev_noise=0.0):
        self.std_dev = std_dev
        self.cor_len = cor_len
        self.std_dev_noise = std_dev_noise

    def kernel_fun(self, x, xp):
        if x.ndim > 1:
            k = self.std_dev**2 * np.exp(-spl.norm(x - xp, axis=1)**2 / (2 * self.cor_len**2))
        else:
            k = self.std_dev**2 * np.exp(-(x - xp)**2 / (2 * self.cor_len**2))
        return k

    def covar(self, x, xp):
        nx  = x.shape[0]
        nxp = xp.shape[0]
        K = np.zeros((nx, nxp))
        for i in range(nxp):
            K[:, i] = self.kernel_fun(x, xp[i])
        di  = np.diag_indices_from(K)
        K[di] = K[di] + self.std_dev_noise**2
        return K

    def sample(self, K, u):
        tol = 1e-12
        di  = np.diag_indices_from(K)
        K[di] = K[di] + tol
        L = spl.cholesky(K, lower=True)
        f = L.dot(u)
        return f

import numpy as np
import numpy.linalg as npl
import numpy.random as npr
import scipy.linalg as spl
import scipy.optimize as spo
import scipy.sparse as sps
from time import time
from geom import Geom
from bc import BC
from darcy import DarcyExp
from dasa import DASAExpLM
from se_kernel import SEKernel

def compute_Lreg(geom):
    Nc        = geom.cells.num
    Nfint     = np.sum(geom.faces.is_interior)
    neighbors = geom.faces.neighbors[:,geom.faces.is_interior]
    rows      = np.concatenate((np.arange(Nfint), np.arange(Nfint)))
    cols      = np.concatenate((neighbors[0], neighbors[1]))
    vals      = np.concatenate((np.full(Nfint, -1.0), np.full(Nfint, 1.0)))
    return sps.coo_matrix((vals, (rows, cols)), shape=(Nfint, Nc))

class LossVec(object):

    def __init__(self, iuobs, uobs, iYobs, Yobs, gamma, L):
        self.iuobs = iuobs
        self.uobs  = uobs
        self.iYobs = iYobs
        self.Yobs  = Yobs
        self.gamma = gamma
        self.L     = L

    def val(self, u, Y):
        Ly  = self.L.dot(Y)
        vec = np.concatenate(((self.uobs - u[self.iuobs]).reshape(-1), (self.Yobs - Y[self.iYobs]).reshape(-1), np.sqrt(self.gamma) * Ly.reshape(-1)))
        return vec

    def grad_u(self, u, Y):
        Nus  = self.iuobs.size
        Nu   = u.size
        cols = self.iuobs
        rows = np.arange(Nus)
        vals = np.full(Nus, 1.0)
        Hu   = sps.coo_matrix((vals, (rows, cols)), shape=(Nus, Nu))
        return -Hu

    def grad_Y(self, u, Y):
        NYs  = self.iYobs.size
        NY   = Y.size
        cols = self.iYobs
        rows = np.arange(NYs)
        vals = np.full(NYs, 1.0)
        Hy   = sps.coo_matrix((vals, (rows, cols)), shape=(NYs, NY))
        return sps.vstack([-Hy, np.sqrt(self.gamma) * self.L])

if __name__ == '__main__':

    L = np.array([1.0, 1.0])
    N = np.array([32,  32])

    g = Geom(L, N)
    g.calculate()

    ul = 2.0
    ur = 1.0
    bc = BC(g)
    bc.dirichlet(g, "left", ul)
    bc.dirichlet(g, "right", ur)

    # Hydraulic conductivity
    npr.seed(0)
    se = SEKernel(std_dev=1.0, cor_len=0.15, std_dev_noise=0.0)
    CY = se.covar(g.cells.centroids.T, g.cells.centroids.T)
    Nc = np.prod(N)
    Y  = se.sample(CY, npr.randn(Nc))
    K  = np.exp(Y)

    # Problem
    prob = DarcyExp(g, bc)
    timer = time()
    u = prob.solve(Y)
    print("Elapsed time: {:g} s".format(time() - timer))

    # Measurements
    Nuobs = 20
    iuobs = npr.choice(Nc, Nuobs, replace=False)
    uobs  = u[iuobs]

    NYobs = 20
    iYobs = npr.choice(Nc, NYobs, replace=False)
    Yobs  = Y[iYobs]

    # Regularizer
    gamma = 1e-6
    Lreg  = compute_Lreg(g)

    # Sensitivity
    loss = LossVec(iuobs, uobs, iYobs, Yobs, gamma, Lreg) # H1 regularization
    # loss = LossVec(iuobs, uobs, iYobs, Yobs, 1e-6, spl.inv(spl.cholesky(CY, lower=True))) # GP prior regularization
    dasa = DASAExpLM(loss.val, loss.grad_u, loss.grad_Y, prob.solve, prob.residual_sens_u, prob.residual_sens_Y)

    # print(dasa.grad(Y))

    # Y0 = Y
    Y0    = np.full(Nc, 0.0)
    Yest  = np.full(Nc, 0.0)
    timer = time()
    res   = spo.leastsq(dasa.obj, Y0, Dfun=dasa.grad)
    print("Elapsed time: {:g} s".format(time() - timer))

    np.savez('test_sdfs_est_lm', centroids=g.cells.centroids, L=L, N=N, u=u, iuobs=iuobs, uobs=uobs, Y=Y, iYobs=iYobs, Yobs=Yobs, gamma=gamma, Yest=res[0], status=res[1])

import numpy as np
import scipy.sparse.linalg as spl
from tpfa import TPFA

class DarcyExp(object):

    def __init__(self, geom, bc):

        self.geom = geom
        self.bc   = bc

    def solve(self, Y):

        K    = np.exp(Y)
        A, b = TPFA.ops(self.geom, self.bc, K)
        return spl.spsolve(A.tocsc(), b)

    def residual(self, u, Y):

        K    = np.exp(Y)
        A, b = TPFA.ops(self.geom, self.bc, K)
        return A.dot(u) - b

    def residual_sens_Y(self, u, Y):

        Nc    = self.geom.cells.num
        K     = np.exp(Y)
        rsens = np.zeros((Nc, Nc))

        for idx in range(0, Nc):
            Asens, bsens = TPFA.sens(self.geom, self.bc, K, idx)
            rsens[idx]   = Asens.dot(u) - bsens

        return rsens * K[:,np.newaxis]

    def residual_sens_u(self, u, Y):

        K = np.exp(Y)
        return TPFA.ops(self.geom, self.bc, K)[0]

        

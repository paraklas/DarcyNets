import numpy as np
import scipy.sparse.linalg as spl

class DASAExp(object):

    def __init__(self, objfun, obj_sens_state, obj_sens_param, solvefun, res_sens_state, res_sens_param):
        
        self.objfun   = objfun
        self.solvefun = solvefun
        self.obj_sens_state = obj_sens_state
        self.obj_sens_param = obj_sens_param
        self.res_sens_state = res_sens_state
        self.res_sens_param = res_sens_param

    def obj(self, p):
        
        u = self.solvefun(p)
        return self.objfun(u, p)

    def grad(self, p):

        u = self.solvefun(p)
        dhdu = self.obj_sens_state(u, p)
        dhdp = self.obj_sens_param(u, p)
        dLdu = self.res_sens_state(u, p)
        dLdp = self.res_sens_param(u, p)
        adj  = -spl.spsolve(dLdu.T.tocsc(), dhdu)
        sens = dLdp.dot(adj)
        sens = sens + dhdp
        return sens
    
class DASAExpLM(object):

    def __init__(self, objfun, obj_sens_state, obj_sens_param, solvefun, res_sens_state, res_sens_param):
        
        self.objfun   = objfun
        self.solvefun = solvefun
        self.obj_sens_state = obj_sens_state
        self.obj_sens_param = obj_sens_param
        self.res_sens_state = res_sens_state
        self.res_sens_param = res_sens_param

    def obj(self, p):
        
        u = self.solvefun(p)
        return self.objfun(u, p)

    def grad(self, p):

        u = self.solvefun(p)
        dhdu = self.obj_sens_state(u, p)
        dhdp = self.obj_sens_param(u, p)
        dLdu = self.res_sens_state(u, p)
        dLdp = self.res_sens_param(u, p)
        adj  = -spl.spsolve(dLdu.T.tocsc(), dhdu.T.toarray())
        sens = dLdp.dot(adj)
        sens = np.concatenate((sens.T, dhdp.toarray()), axis=0)
        return sens

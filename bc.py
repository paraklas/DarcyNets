import numpy as np

class BC(object):

    def __init__(self, geom):
        self.kind = np.empty(geom.faces.num, dtype="str")
        self.val  = np.zeros(geom.faces.num)

    def side_gf_idx(self, geom, side):

        if side is "left":
            idx = np.logical_and(geom.faces.neighbors[0,:] < 0, geom.faces.normals[0,:] > 0)
        elif side is "right":
            idx = np.logical_and(geom.faces.neighbors[1,:] < 0, geom.faces.normals[0,:] > 0)
        elif side is "bottom":
            idx = np.logical_and(geom.faces.neighbors[0,:] < 0, geom.faces.normals[1,:] > 0)
        elif side is "top":
            idx = np.logical_and(geom.faces.neighbors[1,:] < 0, geom.faces.normals[1,:] > 0)

        return idx

    def dirichlet(self, geom, side, val):

        idx = self.side_gf_idx(geom, side)
        self.kind[idx] = "D"
        self.val[idx]  = val

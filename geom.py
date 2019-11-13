import numpy as np
from collections import namedtuple

class Geom(object):

    def __init__(self, L, N):
        
        self.L = L
        self.N = N
        # Cartesian coordinates
        self.d = self.L / (1.0 * self.N)
        self.x = np.linspace(0, self.L[0] - self.d[0], self.N[0]) + 0.5 * self.d[0]
        self.y = np.linspace(0, self.L[1] - self.d[1], self.N[1]) + 0.5 * self.d[1]
        # Substructures
        self.faces = namedtuple('faces', ['num', 'centroids', 'to_hf', 'areas', 'neighbors', 'normals', 'is_interior'])
        self.cells = namedtuple('cells', ['num', 'centroids', 'to_hf', 'volumes'])

    def calculate(self):

        ### Faces
        self.faces.num = (self.N[0] + 1) * self.N[1] + (self.N[1] + 1) * self.N[0]
        
        ## Centroids
        sx = self.N[0] + 1
        sy = self.N[1] + 1
        # Faces orthogonal to 0-direction (x)
        xcv = np.arange(sx) * self.d[0]
        ycv = np.arange(self.N[1]) * self.d[1] + 0.5 * self.d[1]
        xc, yc = np.meshgrid(xcv, ycv)
        fx_centroids = np.concatenate((xc.reshape((1, -1)), yc.reshape((1, -1))))
        # Faces orthogonal to 1-direction (y)
        xcv = np.arange(self.N[0]) * self.d[0] + 0.5 * self.d[0]
        ycv = np.arange(sy) * self.d[1]
        xc, yc = np.meshgrid(xcv, ycv)
        fy_centroids = np.concatenate((xc.reshape((1, -1)), yc.reshape((1, -1))))
        self.faces.centroids = np.concatenate((fx_centroids, fy_centroids), axis=1)

        ## Global faces to half faces
        nfx = sx * self.N[1]
        nfy = sy * self.N[0]
        fx  = np.arange(nfx).reshape((self.N[1], sx))
        fy  = np.arange(nfy).reshape((sy, self.N[0])) + nfx
        f1  = fx[:,0:self.N[0]].reshape((1, -1))
        f2  = fx[:,1:sx].reshape((1, -1))
        f3  = fy[0:self.N[1]].reshape((1, -1))
        f4  = fy[1:sy].reshape((1, -1))
        self.faces.to_hf = np.concatenate((f1, f2, f3, f4)).T.reshape((1, -1)).flatten()

        # Areas
        self.faces.areas = np.concatenate((np.tile(self.d[1], (1, nfx)), np.tile(self.d[0], (1, nfy))), axis=1).flatten()
        
        ### Cells
        self.cells.num = np.prod(self.N)
        
        ## Centroids
        xc, yc = np.meshgrid(self.x, self.y)
        self.cells.centroids = np.concatenate((xc.reshape((1, -1)), yc.reshape((1, -1))))
        
        ## Volumes
        self.cells.volumes = np.prod(self.d) * np.ones((1, self.cells.num)).flatten()

        ## Cells to half faces
        self.cells.to_hf = np.tile(np.arange(self.cells.num), (4, 1)).T.reshape((1, -1)).flatten()

        ### Neighbors
        C = np.full((sy + 1, sx + 1), -1) # -1 indicates that there's no neighbor
        C[1 : sy, 1 : sx] = np.arange(self.cells.num).reshape((self.N[1], -1))
        nx1 = C[1:sy, 0:sx].reshape((1, -1))
        nx2 = C[1:sy, 1:  ].reshape((1, -1))
        ny1 = C[0:sy, 1:sx].reshape((1, -1))
        ny2 = C[1:,   1:sx].reshape((1, -1))
        self.faces.neighbors = np.concatenate((np.concatenate((nx1, ny1), axis=1), np.concatenate((nx2, ny2), axis=1)))

        ### Normals
        normals_x = np.concatenate((np.full((1, nfx), self.d[1]), np.full((1, nfx), 0)))
        normals_y = np.concatenate((np.full((1, nfy), 0),         np.full((1, nfy), self.d[0])))
        self.faces.normals = np.concatenate((normals_x, normals_y), axis=1)

        # Interior face identification
        # self.faces.is_interior = np.logical_and(np.logical_not(np.isnan(self.faces.neighbors[0])), np.logical_not(np.isnan(self.faces.neighbors[1]))).flatten()
        self.faces.is_interior = np.logical_and(self.faces.neighbors[0] >= 0, self.faces.neighbors[1] >= 0).flatten()





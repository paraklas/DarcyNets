#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 21:14:04 2018

@author: Paris

python main_BCs.py gamma seed uobs kobs collobs
"""
import sys
sys.path.append('../../david_experiment/sdfs/')
import csv
import numpy as np
import matplotlib.pyplot as plt
from pyDOE import lhs
from scipy.interpolate import griddata

#from models_tf import DarcyNet2D_BCs
from test_sdfs_est_lm import *

import tensorflow as tf

tf.set_random_seed(int(sys.argv[-4]))
np.random.seed(int(sys.argv[-4]))


if __name__ == "__main__": 

    dataset = np.load('../test.npz')
    X = dataset['coo']
    K = dataset['k']
    U = dataset['u']
    
    idx = 11
    X_star = X
    k_star = K[idx:idx+1,:].T
    u_star = U[idx:idx+1,:].T

    N_u = int(sys.argv[-3])
    N_k = int(sys.argv[-2])
    N_f = int(sys.argv[-1])
    N = u_star.shape[0]  
    res = int(N**(1/2))

    lb, ub = X.min(0), X.max(0)
    
    #Latin Hypercube sampling
    #make obs deterministic for different net initialization seeds and 
    # increasing as you add more points
    np.random.seed(32)
    
    samples=1
    idx_us = [np.floor(N*lhs(1, N)).astype(int).flatten()[:N_u] ]*samples
    idx_ks = [np.floor(N*lhs(1, N)).astype(int).flatten()[:N_k] ]*samples
#    idx_us = [np.floor(N*lhs(1, N)).astype(int).flatten()[:N_u] for _ in 
#              range(samples)]
#    idx_ks = [np.floor(N*lhs(1, N)).astype(int).flatten()[:N_k] for _ in 
#              range(samples)]
    idx_fs = [np.floor(N*lhs(1, N)).astype(int).flatten()[:N_f] ]*samples
#    idx_fs = [np.floor(N*lhs(1, N)).astype(int).flatten()[:N_f] for _ in 
#              range(samples)]

#idx_us = np.apply_along_axis(lambda r : r[0]+32*r[1], 
#                             axis=1, 
#                             arr=np.floor(res*lhs(2)).astype(int))[:N_u]
    #reset seed
    np.random.seed(int(sys.argv[-4]))

    errors_k = []
    errors_u = []

    for idx_u, idx_k, idx_f in zip(idx_us,idx_ks, idx_fs): 

        # Training data
#    idx_k = np.random.choice(N, N_k)
        Y_y = np.log(k_star[idx_k,:]).flatten()
        
#    idx_u = np.random.choice(N, N_u)
        Y_u = u_star[idx_u,:].flatten()

        L = np.array([1.0, 1.0])
        N = np.array([32,  32])

        g = Geom(L, N)
        g.calculate()

        ul = 1.0
        ur = 0.0
        bc = BC(g)
        bc.dirichlet(g, "left", ul)
        bc.dirichlet(g, "right", ur)

       
        se = SEKernel(std_dev=1.0, cor_len=0.15, std_dev_noise=1e-4)
        CY = se.covar(g.cells.centroids.T, g.cells.centroids.T)

        # Create model
        
        prob = DarcyExp(g, bc)

        gamma = float(sys.argv[-5])
        Lreg  = compute_Lreg(g)

#        loss = LossVec(idx_u, Y_u, idx_k, Y_y, gamma, spl.inv(spl.cholesky(CY, lower=True)))
        loss = LossVec(idx_u, Y_u, idx_k, Y_y, gamma, Lreg)

        dasa = DASAExpLM(loss.val, loss.grad_u, loss.grad_Y, prob.solve, prob.residual_sens_u, prob.residual_sens_Y)



        Y0 = np.full(g.cells.num, 0.0)

        res = spo.leastsq(dasa.obj, Y0, Dfun=dasa.grad)
        k_pred = np.exp(res[0]).reshape(-1,1)
        # Predict at test points

        # Relative L2 error
        error_k = np.linalg.norm(k_star - k_pred, 2)/np.linalg.norm(k_star, 2)
#        error_u = np.linalg.norm(u_star - u_pred, 2)/np.linalg.norm(u_star, 2)

        errors_k.append(error_k)
#        errors_u.append(error_u)

       # Plot
        X_k = X[idx_k,:]
        Y_k = k_star[idx_k,:]

        nn = 200
        x = np.linspace(lb[0], ub[0], nn)
        y = np.linspace(lb[1], ub[1], nn)
        XX, YY = np.meshgrid(x,y)

        K_plot = griddata(X_star, k_pred.flatten(), (XX, YY), method='cubic')
#        U_plot = griddata(X_star, u_pred.flatten(), (XX, YY), method='cubic')
        
        K_error = griddata(X_star, np.abs(k_star-k_pred).flatten(), (XX, YY), method='cubic')
#        U_error = griddata(X_star, np.abs(u_star-u_pred).flatten(), (XX, YY), method='cubic')

        fig = plt.figure(1)
        plt.pcolor(XX, YY, K_plot, cmap='viridis')
        plt.plot(X_k[:,0], X_k[:,1], 'ro', markersize = 1)
        plt.clim(np.min(k_star), np.max(k_star))
        plt.colorbar()
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlabel('$x_1$', fontsize=16)
        plt.ylabel('$x_2$', fontsize=16)
        plt.title('$k(x_1,x_2)$', fontsize=16)
        fig.tight_layout() 
        fig.savefig('./plots/map/map_k_sample_'+str(sys.argv[-5])+'_u_'+sys.argv[-3]+'_k_'+sys.argv[-2]+'_c_'+sys.argv[-1]+'_pred.png')
        fig.clf()

        fig = plt.figure(3)
        plt.pcolor(XX, YY, K_error, cmap='viridis')
        plt.colorbar()
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlabel('$x_1$', fontsize=16)
        plt.ylabel('$x_2$', fontsize=16)
        plt.title('Absolute error', fontsize=16) 
        fig.tight_layout()
        fig.savefig('./plots/map/map_k_sample_'+str(sys.argv[-5])+'_u_'+sys.argv[-3]+'_k_'+sys.argv[-2]+'_c_'+sys.argv[-1]+'_errors.png')
        fig.clf()
 
        samples=samples-1

        #completely reset tensorflow
        tf.reset_default_graph()

    with open("./errors/map/map_k_loss_u_"+sys.argv[-3]+"_k_"+sys.argv[-2]+"_c_"+sys.argv[-1]+".csv", 
              "a") as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(errors_k)
    f.close()

#    with open("./errors/map_u_loss_u_"+sys.argv[-3]+"_k_"+sys.argv[-2]+"_c_"+sys.argv[-1]+".csv", 
#              "a") as f:
#        writer = csv.writer(f, delimiter=',')
#        writer.writerow(errors_u)
#    f.close()
#
#        
#    #    filename = sys.argv[-1].replace('/','.').split('.')[-2]
#        fig.savefig('./plots/'+sys.argv[-4]+'_u_'+sys.argv[-1]+'_k_'+sys.argv[-2]+'_c_'+sys.argv[-3]+'_test.png')


#        with open('./errors/'+sys.argv[-4]+'_u_'+sys.argv[-1]+'_k_'+sys.argv[-2]+'_c_'+sys.argv[-3]+'_k_error_test.txt', 'a') as losses_file:
#            print(error_k, file=losses_file)
#        losses_file.close()

#        with open('./errors/'+sys.argv[-4]+'_u_'+sys.argv[-1]+'_k_'+sys.argv[-2]+'_c_'+sys.argv[-3]+'_u_error_test.txt', 'a') as losses_file:
#            print(error_u, file=losses_file)
#        losses_file.close()

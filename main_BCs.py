#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
python main_BCs.py runs seed uobs kobs collobs

example
python main_BCs.py 2 16 20 20 1024
"""
import sys
import csv
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from pyDOE import lhs
from scipy.interpolate import griddata

from models_tf import DarcyNet2D_BCs

import tensorflow as tf

tf.set_random_seed(int(sys.argv[-4]))
np.random.seed(int(sys.argv[-4]))

if __name__ == "__main__": 

    dataset = np.load('./test.npz')
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
    
    # Specify input domain bounds
    lb, ub = X.min(0), X.max(0)

    #Latin Hypercube sampling
    #make obs deterministic for different net initialization seeds and 
    # increasing as you add more points
    np.random.seed(32)
    
    samples=int(sys.argv[-5])
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
        X_k = X[idx_k,:]
        Y_k = k_star[idx_k,:]
        
#    idx_u = np.random.choice(N, N_u)
        X_u = X[idx_u,:]
        Y_u = u_star[idx_u,:]
     
#    idx_f = np.random.choice(N, N_f)
        X_f = X_star[idx_f,:]
        Y_f = np.zeros((N_f, 1))
        
        # Dirichlet boundaries
        x_res = int(N**(1/2))
        b0, u0 = X[::x_res,:], u_star[::x_res,:]
        b1, u1 = X[::-x_res,:], u_star[::-x_res,:]    
        X_ubD = np.concatenate([b0, b1], axis = 0)
        Y_ubD = np.concatenate([u0, u1], axis = 0)
        
        # Neumann boundaries
        b2 = X[:x_res,:]
        b3 = X[-x_res:,:]
        X_ubN = np.concatenate([b2, b3], axis = 0)
        Y_ubN = np.zeros((X_ubN.shape[0], 1))   
        n2 = np.tile(np.array([-1.0, 0.0]), (b2.shape[0],1))
        n3 = np.tile(np.array([1.0, 0.0]),  (b3.shape[0],1))
        normal_vec = np.concatenate([n2, n3], axis = 0)

#        plt.figure(1)
#        plt.plot(X[:,0], X[:,1], 'ko')
#        plt.plot(b0[:,0], b0[:,1], 'ro')
#        plt.plot(b1[:,0], b1[:,1], 'go')
#        plt.plot(b2[:,0], b2[:,1], 'bo')
#        plt.plot(b3[:,0], b3[:,1], 'mo')
#        plt.show()
       
        # Create model
        layers_u = [2,50,50,50,1]
        layers_k = [2,50,50,50,1]
        model = DarcyNet2D_BCs(X_k, Y_k, X_u, Y_u, X_f, Y_f, 
                               X_ubD, Y_ubD, X_ubN, Y_ubN, normal_vec,
                               layers_k, layers_u, lb, ub)
        
        # Train
        model.train()
        
        # Predict at test points
        k_pred = model.predict_k(X_star)
        u_pred = model.predict_u(X_star)
        
        # Relative L2 error
        error_k = np.linalg.norm(k_star - k_pred, 2)/np.linalg.norm(k_star, 2)
        error_u = np.linalg.norm(u_star - u_pred, 2)/np.linalg.norm(u_star, 2)

        errors_k.append(error_k)
        errors_u.append(error_u)

        #completely reset tensorflow
        tf.reset_default_graph()

        nn = 200
        x = np.linspace(lb[0], ub[0], nn)
        y = np.linspace(lb[1], ub[1], nn)
        XX, YY = np.meshgrid(x,y)

        K_orig = griddata(X_star, k_star.flatten(), (XX, YY), method='cubic')
        U_orig = griddata(X_star, u_star.flatten(), (XX, YY), method='cubic')
        K_plot = griddata(X_star, k_pred.flatten(), (XX, YY), method='cubic')
        U_plot = griddata(X_star, u_pred.flatten(), (XX, YY), method='cubic')
        
        K_error = griddata(X_star, np.abs(k_star-k_pred).flatten(), (XX, YY), method='cubic')
        U_error = griddata(X_star, np.abs(u_star-u_pred).flatten(), (XX, YY), method='cubic')

        fig = plt.figure(10)
        plt.pcolor(XX, YY, K_orig, cmap='viridis')
        plt.clim(np.min(k_star), np.max(k_star))
        plt.colorbar()
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlabel('$x_1$', fontsize=16)
        plt.ylabel('$x_2$', fontsize=16)  
        plt.title('$k(x_1,x_2)$', fontsize=16)
        fig.tight_layout()
        fig.savefig('./plots/collocation/orginal_k_field.png')
        fig.clf()

        fig = plt.figure(11)
        plt.pcolor(XX, YY, U_orig, cmap='viridis')
        plt.clim(np.min(u_star), np.max(u_star))
        plt.colorbar()
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlabel('$x_1$', fontsize=16)
        plt.ylabel('$x_2$', fontsize=16)  
        plt.title('$u(x_1,x_2)$', fontsize=16)
        fig.tight_layout()
        fig.savefig('./plots/collocation/orginal_u_field.png')
        fig.clf()


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
        fig.savefig('./plots/collocation/kfield_sample_'+str(samples)+'_seed_'+str(sys.argv[-4])+'_u_'+sys.argv[-3]+'_k_'+sys.argv[-2]+'_c_'+sys.argv[-1]+'_pred.png')
        fig.clf()

        fig = plt.figure(2)
        plt.pcolor(XX, YY, U_plot, cmap='viridis')
        plt.plot(X_u[:,0], X_u[:,1], 'ro', markersize = 1)    
        plt.plot(X_ubD[:,0], X_ubD[:,1], 'ro', markersize = 1)   
        plt.plot(X_ubN[:,0], X_ubN[:,1], 'ro', markersize = 1)   
        plt.clim(np.min(u_star), np.max(u_star))
        plt.colorbar()
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlabel('$x_1$', fontsize=16)
        plt.ylabel('$x_2$', fontsize=16)  
        plt.title('$u(x_1,x_2)$', fontsize=16) 
        fig.tight_layout()
        fig.savefig('./plots/collocation/ufield_sample_'+str(samples)+'_seed_'+str(sys.argv[-4])+'_u_'+sys.argv[-3]+'_k_'+sys.argv[-2]+'_c_'+sys.argv[-1]+'_pred.png')
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
        fig.savefig('./plots/collocation/kfield_sample_'+str(samples)+'_seed_'+str(sys.argv[-4])+'_u_'+sys.argv[-3]+'_k_'+sys.argv[-2]+'_c_'+sys.argv[-1]+'_error.png')
        fig.clf()
        
        fig = plt.figure(4)
        plt.pcolor(XX, YY, U_error, cmap='viridis')
        plt.colorbar()
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlabel('$x_1$', fontsize=16)
        plt.ylabel('$x_2$', fontsize=16)  
        plt.title('Absolute error', fontsize=16)
        fig.tight_layout()
        fig.savefig('./plots/collocation/ufield_sample_'+str(samples)+'_seed_'+str(sys.argv[-4])+'_u_'+sys.argv[-3]+'_k_'+sys.argv[-2]+'_c_'+sys.argv[-1]+'_error.png')
        fig.clf()

        plt.close('all')

        #use to label plots
        samples=samples-1


    with open("./errors/collocation/k_loss_u_"+sys.argv[-3]+"_k_"+sys.argv[-2]+"_c_"+sys.argv[-1]+".csv", 
              "a") as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(errors_k)
    f.close()

    with open("./errors/collocation/u_loss_u_"+sys.argv[-3]+"_k_"+sys.argv[-2]+"_c_"+sys.argv[-1]+".csv", 
              "a") as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(errors_u)
    f.close()
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

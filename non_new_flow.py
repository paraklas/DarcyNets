#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 18:24:02 2018

@author: Paris
"""

import sys
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from pyDOE import lhs

from models_tf import DarcyNet_toy1D_non_newtonian

np.random.seed(int(sys.argv[-1]))
tf.set_random_seed(int(sys.argv[-1]))

if __name__ == "__main__": 
    
    #Load Data
    df = pd.read_excel('./non_Newtonian_flow.xlsx', header=None)


    num_obs = df[[0]][8:].shape[0]
    N_u = num_obs
    N_f = num_obs

    lb = 0.0
    ub = 1.0

    # Training data

    u_idx = np.cast[int](num_obs*lhs(1, N_u)).ravel()
    X_u = np.cast[np.float64](df[[0]][8:].values[u_idx])
    Y_u = np.cast[np.float64](df[[1]][8:].values[u_idx])
    
    f_idx = np.cast[int](num_obs*lhs(1, N_f)).ravel()
    X_f = np.cast[np.float64](df[[0]][8:].values[f_idx])
    Y_f = -1*np.ones((N_f, 1))
    
    # Dirichlet boundaries   
    X_ubD = np.array([[lb]])
    Y_ubD = np.array([[0.0]])
    
    # Neumann boundaries
    X_ubN = np.array([[ub]])
    Y_ubN = np.array([[0.0]])
    
    # Test data
    X_star = np.cast[np.float64](df[[0]][8:].values)
    u_star = np.cast[np.float64](df[[1]][8:].values)
#    k_star = k(u_star)
    
    # Create model
    layers_k = [1,50,50,1]
    layers_u = [1,50,50,1]
    model = DarcyNet_toy1D_non_newtonian(X_u, Y_u, X_f, Y_f, 
                                    X_ubD, Y_ubD, X_ubN, Y_ubN,
                                    layers_u, layers_k, lb, ub)
    
    # Train
    model.train()
    
    # Predict at test points
    u_pred = model.predict_u(X_star)
    k_pred = model.predict_k(u_pred)
    
    # Relative L2 error
    error_u = np.linalg.norm(u_star - u_pred, 2)/np.linalg.norm(u_star, 2)
#    error_k = np.linalg.norm(k_star - k_pred, 2)/np.linalg.norm(k_star, 2)
    
    print('Error u: %e' % (error_u))   
#    print('Error k: %e' % (error_k))   
    
    
    # Plot
    plt.figure(1)
    # u(x) #
    plt.subplot(1,2,1)
    plt.plot(X_star, u_star, 'b', linewidth = 2)
    plt.plot(X_star, u_pred, 'r--', linewidth = 2)
    plt.plot(X_u, Y_u, 'bo', alpha = 0.5)
    plt.xlabel('$x$')
    plt.ylabel('$u(x)$')  
    plt.title('Error u: %e' % (error_u))
    # k(u) #
    plt.subplot(1,2,2)
#    plt.plot(u_star, k_star, 'b', linewidth = 2)
    plt.plot(u_pred, k_pred, 'r--', linewidth = 2)
    plt.xlabel('$u$')
    plt.ylabel('$k(u)$')  
    plt.savefig('./plots/seed_'+sys.argv[-1]+'.png')
    
    

    

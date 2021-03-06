#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 21:14:04 2018

@author: Paris
"""

import numpy as np
import matplotlib.pyplot as plt
from pyDOE import lhs
from scipy.interpolate import griddata

from models_tf import DarcyNet2D

np.random.seed(1234)

if __name__ == "__main__": 
    
    dataset = np.load('./test.npz')
    X = dataset['coo']
    K = dataset['k']
    U = dataset['u']
    
    idx = 11
    X_star = X
    k_star = K[idx:idx+1,:].T
    u_star = U[idx:idx+1,:].T

    N_u = 100
    N_k = 250
    N = u_star.shape[0]  
    
    # Specify input domain bounds
    lb, ub = X.min(0), X.max(0)
    
    # Training data
    idx_k = np.random.choice(N, N_k)
    X_k = X[idx_k,:]
    Y_k = k_star[idx_k,:]
    
    idx_u = np.random.choice(N, N_u)
    X_u = X[idx_u,:]
    Y_u = u_star[idx_u,:]
    
    X_f = X_star
    Y_f = np.zeros((N, 1))
    
    # Create model
    layers_u = [2,50,50,50,1]
    layers_k = [2,50,50,50,1]
    model = DarcyNet2D(X_k, Y_k, X_u, Y_u, X_f, Y_f, 
                      layers_k, layers_u, lb, ub)
    
    # Train
    model.train()
    
    # Predict at test points
    k_pred = model.predict_k(X_star)
    u_pred = model.predict_u(X_star)
    
    # Relative L2 error
    error_k = np.linalg.norm(k_star - k_pred, 2)/np.linalg.norm(k_star, 2)
    error_u = np.linalg.norm(u_star - u_pred, 2)/np.linalg.norm(u_star, 2)
    
    # Plot
    nn = 200
    x = np.linspace(lb[0], ub[0], nn)
    y = np.linspace(lb[1], ub[1], nn)
    XX, YY = np.meshgrid(x,y)
    
    K_plot = griddata(X_star, k_pred.flatten(), (XX, YY), method='cubic')
    U_plot = griddata(X_star, u_pred.flatten(), (XX, YY), method='cubic')
    
    K_error = griddata(X_star, np.abs(k_star-k_pred).flatten(), (XX, YY), method='cubic')
    U_error = griddata(X_star, np.abs(u_star-u_pred).flatten(), (XX, YY), method='cubic')

    fig = plt.figure(1)
    plt.subplot(2,2,1)
    plt.pcolor(XX, YY, K_plot, cmap='viridis')
    plt.plot(X_k[:,0], X_k[:,1], 'ro', markersize = 1)
    plt.colorbar()
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')  
    plt.title('$k(x_1,x_2)$')
    
    plt.subplot(2,2,2)
    plt.pcolor(XX, YY, U_plot, cmap='viridis')
    plt.plot(X_u[:,0], X_u[:,1], 'ro', markersize = 1)    
    plt.colorbar()
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')  
    plt.title('$u(x_1,x_2)$')
    
    plt.subplot(2,2,3)
    plt.pcolor(XX, YY, K_error, cmap='viridis')
    plt.colorbar()
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')  
    plt.title('Absolute error')
    
    plt.subplot(2,2,4)
    plt.pcolor(XX, YY, U_error, cmap='viridis')
    plt.colorbar()
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')  
    plt.title('Absolute error')
    

    

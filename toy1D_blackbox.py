#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 18:24:02 2018

@author: Paris
"""

import numpy as np
import matplotlib.pyplot as plt
from pyDOE import lhs

from models_tf import DarcyNet_toy1D_blackbox

np.random.seed(1234)

if __name__ == "__main__": 
    
    N_u = 8
    N_f = 400
    
    scaling_factor = 0.00007
    q = -7.0e-5/scaling_factor
    lambda_1 = 3.6e-4/scaling_factor
    lambda_2 = 2.0
    u0 = 0.0
    
    lb = 0.0
    ub = 3.0     
    
    # Exact solution
    def k(u):
        k = lambda_1*np.exp(lambda_2*u)
        return k
    
    def u(x):
        u = (1.0/lambda_2)*np.log(-q/lambda_1 + np.exp(-lambda_2*x)*(np.exp(lambda_2*u0) + q/lambda_1))
        return u
    
    # Training data
    X_u = lb + (ub-lb)*lhs(1, N_u)
    Y_u = u(X_u)
    
    X_f = lb + (ub-lb)*lhs(1, N_f)
    Y_f = np.zeros((N_f, 1))
    
    # Dirichlet boundaries   
    X_ubD = np.array([[lb]])
    Y_ubD = np.array([[0.0]])
    
    # Neumann boundaries
    X_ubN = np.array([[ub]])
    Y_ubN = np.array([[q]])
    
    # Test data
    N_star = 400
    X_star = np.linspace(lb,ub,N_star)[:,None]
    u_star = u(X_star)
    k_star = k(u_star)
    
    # Create model
    layers_k = [1,50,50,1]
    layers_u = [1,50,50,1]
    model = DarcyNet_toy1D_blackbox(X_u, Y_u, X_f, Y_f, 
                                    X_ubD, Y_ubD, X_ubN, Y_ubN,
                                    layers_u, layers_k, lb, ub)
    
    # Train
    model.train()
    
    # Predict at test points
    u_pred = model.predict_u(X_star)
    k_pred = model.predict_k(u_pred)
    
    # Relative L2 error
    error_u = np.linalg.norm(u_star - u_pred, 2)/np.linalg.norm(u_star, 2)
    error_k = np.linalg.norm(k_star - k_pred, 2)/np.linalg.norm(k_star, 2)
    
    print('Error u: %e' % (error_u))   
    print('Error k: %e' % (error_k))   
    
    
    # Plot
    plt.figure(1)
    # u(x) #
    plt.subplot(1,2,1)
    plt.plot(X_star, u_star, 'b', linewidth = 2)
    plt.plot(X_star, u_pred, 'r--', linewidth = 2)
    plt.plot(X_u, Y_u, 'bo', alpha = 0.5)
    plt.xlabel('$x$')
    plt.ylabel('$u(x)$')  
    # k(u) #
    plt.subplot(1,2,2)
    plt.plot(u_star, k_star, 'b', linewidth = 2)
    plt.plot(u_pred, k_pred, 'r--', linewidth = 2)
    plt.xlabel('$u$')
    plt.ylabel('$k(u)$')  
    
    

    
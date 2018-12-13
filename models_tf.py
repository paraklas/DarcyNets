#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 13:20:11 2018

@author: Paris
"""

import tensorflow as tf
import numpy as np



class DarcyNet_toy1D_non_newtonian:
    # Initialize the class
    def __init__(self, X_u, Y_u, X_f, Y_f, 
                 X_ubD, Y_ubD, X_ubN, Y_ubN,
                 layers_u, layers_k, lb, ub):
     
        self.lb = lb
        self.ub = ub
        X_u = (X_u - lb) - 0.5*(ub - lb)
        X_f = (X_f - lb) - 0.5*(ub - lb)
        X_ubD = (X_ubD - lb) - 0.5*(ub - lb)
        X_ubN = (X_ubN - lb) - 0.5*(ub - lb)

        #Center outputs
        self.Ymin, self.Ymax = Y_u.min(0), Y_u.max(0)
        Y_u = (Y_u - self.Ymin) - 0.5*(self.Ymax - self.Ymin)
        Y_ubD = (Y_ubD - self.Ymin) - 0.5*(self.Ymax - self.Ymin)
        Y_ubN = (Y_ubN - self.Ymin) - 0.5*(self.Ymax - self.Ymin)
        
        
        self.X_u = X_u
        self.Y_u = Y_u
        
        self.X_f = X_f
        self.Y_f = Y_f
        
        self.X_ubD = X_ubD
        self.Y_ubD = Y_ubD
        
        self.X_ubN = X_ubN
        self.Y_ubN = Y_ubN
                
        self.layers_u = layers_u
        self.layers_k = layers_k

        # Initialize network weights and biases 
        self.weights_k, self.biases_k = self.initialize_NN(layers_k)        
        self.weights_u, self.biases_u = self.initialize_NN(layers_u)
        
        # Define Tensorflow session
        self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        
        # Define placeholders and computational graph
        self.Xu_tf = tf.placeholder(tf.float32, shape=(None, self.X_u.shape[1]))
        self.Yu_tf = tf.placeholder(tf.float32, shape=(None, self.Y_u.shape[1]))
        
        self.Xf_tf = tf.placeholder(tf.float32, shape=(None, self.X_f.shape[1]))
        self.Yf_tf = tf.placeholder(tf.float32, shape=(None, self.Y_f.shape[1]))
        
        self.XubD_tf = tf.placeholder(tf.float32, shape=(None, self.X_ubD.shape[1]))
        self.YubD_tf = tf.placeholder(tf.float32, shape=(None, self.Y_ubD.shape[1]))
        
        self.XubN_tf = tf.placeholder(tf.float32, shape=(None, self.X_ubN.shape[1]))
        self.YubN_tf = tf.placeholder(tf.float32, shape=(None, self.Y_ubN.shape[1]))
        
        # Evaluate prediction
        self.u_pred = self.net_u(self.Xu_tf)
        self.f_pred = self.net_f(self.Xf_tf)
        
        self.ubD_pred = self.net_ubD(self.XubD_tf)
        self.ubN_pred = self.net_ubN(self.XubN_tf)
        
        # Evaluate loss
        self.loss = tf.losses.mean_squared_error(self.Yu_tf, self.u_pred) + \
                    tf.losses.mean_squared_error(self.Yf_tf, self.f_pred) + \
                    tf.losses.mean_squared_error(self.YubD_tf, self.ubD_pred) + \
                    tf.losses.mean_squared_error(self.YubN_tf, self.ubN_pred)
        
        self.k_pred = self.net_k(self.Yu_tf)

        # Define optimizer (use L-BFGS for better accuracy)       
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss, 
                                                                method = 'L-BFGS-B', 
                                                                options = {'maxiter': 50000,
                                                                           'maxfun': 50000,
                                                                           'maxcor': 50,
                                                                           'maxls': 50,
                                                                           'ftol' : 1.0 * np.finfo(float).eps})

        self.optimizer_Adam = tf.train.AdamOptimizer()
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)
        
        # Initialize Tensorflow variables
        init = tf.global_variables_initializer()
        self.sess.run(init)

    
    # Initialize network weights and biases using Xavier initialization
    def initialize_NN(self, layers): 
        # Custom initialization of 1st layer
        def custom_init(size):
            in_dim = size[0]
            out_dim = size[1]
            custom_stddev = np.sqrt(0.5*(1.0/3.0)/self.X_f.var(0, keepdims = True))
            return tf.Variable(np.random.randn(in_dim, out_dim) * custom_stddev.T, dtype=tf.float32)
    
        # Xavier initialization
        def xavier_init(size):
            in_dim = size[0]
            out_dim = size[1]
            xavier_stddev = 1. / np.sqrt((in_dim + out_dim) / 2.)
            return tf.Variable(tf.random_normal([in_dim, out_dim], dtype=tf.float32) * xavier_stddev, dtype=tf.float32)   
        
        weights = []
        biases = []
        num_layers = len(layers) 
        W = custom_init(size=[layers[0], layers[1]])
        b = tf.Variable(tf.zeros([1,layers[1]], dtype=tf.float32), dtype=tf.float32)
        weights.append(W)
        biases.append(b)            
        for l in range(1,num_layers-1):
            W = xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)        
        return weights, biases
       
           
    # Evaluates the forward pass
    def forward_pass(self, H, layers, weights, biases):
        num_layers = len(layers)
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        H = tf.add(tf.matmul(H, W), b)
        return H
    
    
   
    # Forward pass for u
    def net_k(self, u):
        k = self.forward_pass(u,
                              self.layers_k,
                              self.weights_k, 
                              self.biases_k)
        return k
    
    
    # Forward pass for u
    def net_u(self, x):
        u = self.forward_pass(x,
                              self.layers_u,
                              self.weights_u, 
                              self.biases_u)
        return u
    
    
    # Forward pass for f
    def net_f(self, x):
        u = self.net_u(x)
        k = self.net_k(u)
        u_x = tf.gradients(u, x)[0]
        f = tf.gradients(k*u_x, x)[0]
        return f
    
    
    def net_ubD(self, x):
        u = self.net_u(x)
        return u
    
   #need to check if this is how you impose Direclet everywhere. 
    def net_ubN(self, x):
        u = self.net_u(x)
#        k = self.net_k(u)
#        u_x = tf.gradients(u + x, x)[0]
#        g = -k*u_x
#        return g
        return u
       
    
    # Callback to print the loss at every optimization step
    def callback(self, loss):
        print('Loss: %e' % (loss))
       
        

    # Trains the model by minimizing the loss using L-BFGS
    def train(self):         
        # Define a dictionary for associating placeholders with data
        tf_dict = {self.Xu_tf: self.X_u, self.Yu_tf: self.Y_u, 
                   self.Xf_tf: self.X_f, self.Yf_tf: self.Y_f,
                   self.XubD_tf: self.X_ubD, self.YubD_tf: self.Y_ubD,
                   self.XubN_tf: self.X_ubN, self.YubN_tf: self.Y_ubN}
        
        # Call SciPy's L-BFGS otpimizer
        self.optimizer.minimize(self.sess, 
                                feed_dict = tf_dict,         
                                fetches = [self.loss], 
                                loss_callback = self.callback)

    
    # Evaluates predictions at test points           
    def predict_k(self, U_star): 
        # Predict
        tf_dict = {self.Yu_tf: U_star}    
        k_star = self.sess.run(self.k_pred, tf_dict) 
        return k_star
    
    # Evaluates predictions at test points           
    def predict_u(self, X_star): 
        # Center around the origin
        X_star = (X_star - self.lb) - 0.5*(self.ub - self.lb)
        # Predict
        tf_dict = {self.Xu_tf: X_star}    
        u_star = self.sess.run(self.u_pred, tf_dict) 
        #de-normalize outputs
        u_star = u_star + 0.5*(self.Ymax - self.Ymin) + self.Ymin
        return u_star
    
    # Evaluates predictions at test points           
    def predict_f(self, X_star): 
        # Center around the origin
        X_star = (X_star - self.lb) - 0.5*(self.ub - self.lb)
        # Predict
        tf_dict = {self.Xf_tf: X_star}    
        f_star = self.sess.run(self.f_pred, tf_dict) 
        return f_star
    





class DarcyNet_toy1D_blackbox:
    # Initialize the class
    def __init__(self, X_u, Y_u, X_f, Y_f, 
                 X_ubD, Y_ubD, X_ubN, Y_ubN,
                 layers_u, layers_k, lb, ub):
     
        self.lb = lb
        self.ub = ub
        X_u = (X_u - lb) - 0.5*(ub - lb)
        X_f = (X_f - lb) - 0.5*(ub - lb)
        X_ubD = (X_ubD - lb) - 0.5*(ub - lb)
        X_ubN = (X_ubN - lb) - 0.5*(ub - lb)
        
        self.X_u = X_u
        self.Y_u = Y_u
        
        self.X_f = X_f
        self.Y_f = Y_f
        
        self.X_ubD = X_ubD
        self.Y_ubD = Y_ubD
        
        self.X_ubN = X_ubN
        self.Y_ubN = Y_ubN
                
        self.layers_u = layers_u
        self.layers_k = layers_k

        # Initialize network weights and biases 
        self.weights_k, self.biases_k = self.initialize_NN(layers_k)        
        self.weights_u, self.biases_u = self.initialize_NN(layers_u)
        
        # Define Tensorflow session
        self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        
        # Define placeholders and computational graph
        self.Xu_tf = tf.placeholder(tf.float32, shape=(None, self.X_u.shape[1]))
        self.Yu_tf = tf.placeholder(tf.float32, shape=(None, self.Y_u.shape[1]))
        
        self.Xf_tf = tf.placeholder(tf.float32, shape=(None, self.X_f.shape[1]))
        self.Yf_tf = tf.placeholder(tf.float32, shape=(None, self.Y_f.shape[1]))
        
        self.XubD_tf = tf.placeholder(tf.float32, shape=(None, self.X_ubD.shape[1]))
        self.YubD_tf = tf.placeholder(tf.float32, shape=(None, self.Y_ubD.shape[1]))
        
        self.XubN_tf = tf.placeholder(tf.float32, shape=(None, self.X_ubN.shape[1]))
        self.YubN_tf = tf.placeholder(tf.float32, shape=(None, self.Y_ubN.shape[1]))
        
        # Evaluate prediction
        self.u_pred = self.net_u(self.Xu_tf)
        self.f_pred = self.net_f(self.Xf_tf)
        
        self.ubD_pred = self.net_ubD(self.XubD_tf)
        self.ubN_pred = self.net_ubN(self.XubN_tf)
        
        # Evaluate loss
        self.loss = tf.losses.mean_squared_error(self.Yu_tf, self.u_pred) + \
                    tf.losses.mean_squared_error(self.Yf_tf, self.f_pred) + \
                    tf.losses.mean_squared_error(self.YubD_tf, self.ubD_pred) + \
                    tf.losses.mean_squared_error(self.YubN_tf, self.ubN_pred)
        
        self.k_pred = self.net_k(self.Yu_tf)

        # Define optimizer (use L-BFGS for better accuracy)       
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss, 
                                                                method = 'L-BFGS-B', 
                                                                options = {'maxiter': 50000,
                                                                           'maxfun': 50000,
                                                                           'maxcor': 50,
                                                                           'maxls': 50,
                                                                           'ftol' : 1.0 * np.finfo(float).eps})

        self.optimizer_Adam = tf.train.AdamOptimizer()
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)
        
        # Initialize Tensorflow variables
        init = tf.global_variables_initializer()
        self.sess.run(init)

    
    # Initialize network weights and biases using Xavier initialization
    def initialize_NN(self, layers): 
        # Custom initialization of 1st layer
        def custom_init(size):
            in_dim = size[0]
            out_dim = size[1]
            custom_stddev = np.sqrt(0.5*(1.0/3.0)/self.X_f.var(0, keepdims = True))
            return tf.Variable(np.random.randn(in_dim, out_dim) * custom_stddev.T, dtype=tf.float32)
    
        # Xavier initialization
        def xavier_init(size):
            in_dim = size[0]
            out_dim = size[1]
            xavier_stddev = 1. / np.sqrt((in_dim + out_dim) / 2.)
            return tf.Variable(tf.random_normal([in_dim, out_dim], dtype=tf.float32) * xavier_stddev, dtype=tf.float32)   
        
        weights = []
        biases = []
        num_layers = len(layers) 
        W = custom_init(size=[layers[0], layers[1]])
        b = tf.Variable(tf.zeros([1,layers[1]], dtype=tf.float32), dtype=tf.float32)
        weights.append(W)
        biases.append(b)            
        for l in range(1,num_layers-1):
            W = xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)        
        return weights, biases
       
           
    # Evaluates the forward pass
    def forward_pass(self, H, layers, weights, biases):
        num_layers = len(layers)
        for l in range(0,num_layers-2):
            l = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        H = tf.add(tf.matmul(H, W), b)
        return H
    
    
   
    # Forward pass for u
    def net_k(self, u):
        k = self.forward_pass(u,
                              self.layers_k,
                              self.weights_k, 
                              self.biases_k)
        return k
    
    
    # Forward pass for u
    def net_u(self, x):
        u = self.forward_pass(x,
                              self.layers_u,
                              self.weights_u, 
                              self.biases_u)
        return u
    
    
    # Forward pass for f
    def net_f(self, x):
        u = self.net_u(x)
        k = self.net_k(u)
        u_x = tf.gradients(u + x, x)[0]
        f = tf.gradients(k*u_x, x)[0]
        return f
    
    
    def net_ubD(self, x):
        u = self.net_u(x)
        return u
    
    
    def net_ubN(self, x):
        u = self.net_u(x)
        k = self.net_k(u)
        u_x = tf.gradients(u + x, x)[0]
        g = -k*u_x
        return g
       
    
    # Callback to print the loss at every optimization step
    def callback(self, loss):
        print('Loss: %e' % (loss))
       
        

    # Trains the model by minimizing the loss using L-BFGS
    def train(self):         
        # Define a dictionary for associating placeholders with data
        tf_dict = {self.Xu_tf: self.X_u, self.Yu_tf: self.Y_u, 
                   self.Xf_tf: self.X_f, self.Yf_tf: self.Y_f,
                   self.XubD_tf: self.X_ubD, self.YubD_tf: self.Y_ubD,
                   self.XubN_tf: self.X_ubN, self.YubN_tf: self.Y_ubN}
        
        # Call SciPy's L-BFGS otpimizer
        self.optimizer.minimize(self.sess, 
                                feed_dict = tf_dict,         
                                fetches = [self.loss], 
                                loss_callback = self.callback)

    
    # Evaluates predictions at test points           
    def predict_k(self, U_star): 
        # Predict
        tf_dict = {self.Yu_tf: U_star}    
        k_star = self.sess.run(self.k_pred, tf_dict) 
        return k_star
    
    # Evaluates predictions at test points           
    def predict_u(self, X_star): 
        # Center around the origin
        X_star = (X_star - self.lb) - 0.5*(self.ub - self.lb)
        # Predict
        tf_dict = {self.Xu_tf: X_star}    
        u_star = self.sess.run(self.u_pred, tf_dict) 
        return u_star
    
    # Evaluates predictions at test points           
    def predict_f(self, X_star): 
        # Center around the origin
        X_star = (X_star - self.lb) - 0.5*(self.ub - self.lb)
        # Predict
        tf_dict = {self.Xf_tf: X_star}    
        f_star = self.sess.run(self.f_pred, tf_dict) 
        return f_star
    
    
    
class DarcyNet_toy1D_parametrized:
    # Initialize the class
    def __init__(self, X_u, Y_u, X_f, Y_f, 
                 X_ubD, Y_ubD, X_ubN, Y_ubN,
                 layers_u, lb, ub):
     
        self.lb = lb
        self.ub = ub
        X_u = (X_u - lb) - 0.5*(ub - lb)
        X_f = (X_f - lb) - 0.5*(ub - lb)
        X_ubD = (X_ubD - lb) - 0.5*(ub - lb)
        X_ubN = (X_ubN - lb) - 0.5*(ub - lb)
        
        self.X_u = X_u
        self.Y_u = Y_u
        
        self.X_f = X_f
        self.Y_f = Y_f
        
        self.X_ubD = X_ubD
        self.Y_ubD = Y_ubD
        
        self.X_ubN = X_ubN
        self.Y_ubN = Y_ubN
                
        self.layers_u = layers_u

        # Initialize network weights and biases  
        self.weights_u, self.biases_u = self.initialize_NN(layers_u)
        
        # Define Tensorflow session
        self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        
        # Define placeholders and computational graph
        self.Xu_tf = tf.placeholder(tf.float32, shape=(None, self.X_u.shape[1]))
        self.Yu_tf = tf.placeholder(tf.float32, shape=(None, self.Y_u.shape[1]))
        
        self.Xf_tf = tf.placeholder(tf.float32, shape=(None, self.X_f.shape[1]))
        self.Yf_tf = tf.placeholder(tf.float32, shape=(None, self.Y_f.shape[1]))
        
        self.XubD_tf = tf.placeholder(tf.float32, shape=(None, self.X_ubD.shape[1]))
        self.YubD_tf = tf.placeholder(tf.float32, shape=(None, self.Y_ubD.shape[1]))
        
        self.XubN_tf = tf.placeholder(tf.float32, shape=(None, self.X_ubN.shape[1]))
        self.YubN_tf = tf.placeholder(tf.float32, shape=(None, self.Y_ubN.shape[1]))
        
        # Initialize parameters (lambda_1 = 3.6e-4, lambda_2 = 2.0)
        self.lambda_1 = tf.Variable([1.0], dtype=tf.float32)
        self.lambda_2 = tf.Variable([1.0], dtype=tf.float32)
        
        # Evaluate prediction
        self.u_pred = self.net_u(self.Xu_tf)
        self.f_pred = self.net_f(self.Xf_tf)
        
        self.ubD_pred = self.net_ubD(self.XubD_tf)
        self.ubN_pred = self.net_ubN(self.XubN_tf)
        
        # Evaluate loss
        self.loss = tf.losses.mean_squared_error(self.Yu_tf, self.u_pred) + \
                    tf.losses.mean_squared_error(self.Yf_tf, self.f_pred) + \
                    tf.losses.mean_squared_error(self.YubD_tf, self.ubD_pred) + \
                    tf.losses.mean_squared_error(self.YubN_tf, self.ubN_pred)
        
        # Define optimizer (use L-BFGS for better accuracy)       
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss, 
                                                                method = 'L-BFGS-B', 
                                                                options = {'maxiter': 50000,
                                                                           'maxfun': 50000,
                                                                           'maxcor': 50,
                                                                           'maxls': 50,
                                                                           'ftol' : 1.0 * np.finfo(float).eps})

        self.optimizer_Adam = tf.train.AdamOptimizer()
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)
        
        # Initialize Tensorflow variables
        init = tf.global_variables_initializer()
        self.sess.run(init)

    
    # Initialize network weights and biases using Xavier initialization
    def initialize_NN(self, layers): 
        # Custom initialization of 1st layer
        def custom_init(size):
            in_dim = size[0]
            out_dim = size[1]
            custom_stddev = np.sqrt(0.5*(1.0/3.0)/self.X_f.var(0, keepdims = True))
            return tf.Variable(np.random.randn(in_dim, out_dim) * custom_stddev.T, dtype=tf.float32)
    
        # Xavier initialization
        def xavier_init(size):
            in_dim = size[0]
            out_dim = size[1]
            xavier_stddev = 1. / np.sqrt((in_dim + out_dim) / 2.)
            return tf.Variable(tf.random_normal([in_dim, out_dim], dtype=tf.float32) * xavier_stddev, dtype=tf.float32)   
        
        weights = []
        biases = []
        num_layers = len(layers) 
        W = custom_init(size=[layers[0], layers[1]])
        b = tf.Variable(tf.zeros([1,layers[1]], dtype=tf.float32), dtype=tf.float32)
        weights.append(W)
        biases.append(b)            
        for l in range(1,num_layers-1):
            W = xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)        
        return weights, biases
       
           
    # Evaluates the forward pass
    def forward_pass(self, H, layers, weights, biases):
        num_layers = len(layers)
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        H = tf.add(tf.matmul(H, W), b)
        return H
    
    
   
    # Forward pass for u
    def net_u(self, x):
        u = self.forward_pass(x,
                              self.layers_u,
                              self.weights_u, 
                              self.biases_u)
        return u
    
    
    # Forward pass for f
    def net_f(self, x):
        lambda_1 = self.lambda_1
        lambda_2 = self.lambda_2    
        u = self.net_u(x)
        k = lambda_1*tf.exp(lambda_2*u)
        u_x = tf.gradients(u + x, x)[0]
        f = tf.gradients(k*u_x, x)[0]
        return f
    
    
    def net_ubD(self, x):
        u = self.net_u(x)
        return u
    
    
    def net_ubN(self, x):
        lambda_1 = self.lambda_1
        lambda_2 = self.lambda_2      
        u = self.net_u(x)
        k = lambda_1*tf.exp(lambda_2*u)
        u_x = tf.gradients(u + x, x)[0]
        g = -k*u_x
        return g
       
    
    # Callback to print the loss at every optimization step
    def callback(self, loss, lambda_1, lambda_2):
        print('Loss: %e, l1: %.5f, l2: %.5f' % (loss, lambda_1, lambda_2))
       
        

    # Trains the model by minimizing the loss using L-BFGS
    def train(self):         
        # Define a dictionary for associating placeholders with data
        tf_dict = {self.Xu_tf: self.X_u, self.Yu_tf: self.Y_u, 
                   self.Xf_tf: self.X_f, self.Yf_tf: self.Y_f,
                   self.XubD_tf: self.X_ubD, self.YubD_tf: self.Y_ubD,
                   self.XubN_tf: self.X_ubN, self.YubN_tf: self.Y_ubN}
        
        # Call SciPy's L-BFGS otpimizer
        self.optimizer.minimize(self.sess, 
                                feed_dict = tf_dict,         
                                fetches = [self.loss, self.lambda_1, self.lambda_2], 
                                loss_callback = self.callback)

    
    # Evaluates predictions at test points           
    def predict_u(self, X_star): 
        # Center around the origin
        X_star = (X_star - self.lb) - 0.5*(self.ub - self.lb)
        # Predict
        tf_dict = {self.Xu_tf: X_star}    
        u_star = self.sess.run(self.u_pred, tf_dict) 
        return u_star
    
    # Evaluates predictions at test points           
    def predict_f(self, X_star): 
        # Center around the origin
        X_star = (X_star - self.lb) - 0.5*(self.ub - self.lb)
        # Predict
        tf_dict = {self.Xf_tf: X_star}    
        f_star = self.sess.run(self.f_pred, tf_dict) 
        return f_star
    


class DarcyNet2D:
    # Initialize the class
    def __init__(self, X_k, Y_k, X_u, Y_u, X_f, Y_f, 
                 layers_k, layers_u, lb, ub):
     
        self.lb = lb
        self.ub = ub
        X_k = (X_k - lb) - 0.5*(ub - lb)
        X_u = (X_u - lb) - 0.5*(ub - lb)
        X_f = (X_f - lb) - 0.5*(ub - lb)
        
        self.x1_k = X_k[:,0:1]
        self.x2_k = X_k[:,1:2]
        self.Y_k = Y_k
        
        self.x1_u = X_u[:,0:1]
        self.x2_u = X_u[:,1:2]
        self.Y_u = Y_u
        
        self.x1_f = X_f[:,0:1]
        self.x2_f = X_f[:,1:2]
        self.Y_f = Y_f
        
        self.X_f = X_f
        
        self.layers_k = layers_k
        self.layers_u = layers_u

        # Initialize network weights and biases  
        self.weights_k, self.biases_k = self.initialize_NN(layers_k)        
        self.weights_u, self.biases_u = self.initialize_NN(layers_u)
        
        # Define Tensorflow session
        self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        
        # Define placeholders and computational graph
        self.x1_k_tf = tf.placeholder(tf.float32, shape=(None, self.x1_k.shape[1]))
        self.x2_k_tf = tf.placeholder(tf.float32, shape=(None, self.x2_k.shape[1]))        
        self.Yk_tf = tf.placeholder(tf.float32, shape=(None, self.Y_k.shape[1]))
        
        self.x1_u_tf = tf.placeholder(tf.float32, shape=(None, self.x1_u.shape[1]))
        self.x2_u_tf = tf.placeholder(tf.float32, shape=(None, self.x2_u.shape[1]))        
        self.Yu_tf = tf.placeholder(tf.float32, shape=(None, self.Y_u.shape[1]))
        
        self.x1_f_tf = tf.placeholder(tf.float32, shape=(None, self.x1_f.shape[1]))
        self.x2_f_tf = tf.placeholder(tf.float32, shape=(None, self.x2_f.shape[1]))        
        self.Yf_tf = tf.placeholder(tf.float32, shape=(None, self.Y_f.shape[1]))
        
        # Evaluate prediction
        self.k_pred = self.net_k(self.x1_k_tf, self.x2_k_tf)
        self.u_pred = self.net_u(self.x1_u_tf, self.x2_u_tf)
        self.f_pred = self.net_f(self.x1_f_tf, self.x2_f_tf)
        
        # Evaluate loss
        self.loss = tf.losses.mean_squared_error(self.Yk_tf, self.k_pred) + \
                    tf.losses.mean_squared_error(self.Yu_tf, self.u_pred) + \
                    tf.losses.mean_squared_error(self.Yf_tf, self.f_pred)
        
        # Define optimizer (use L-BFGS for better accuracy)       
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss, 
                                                                method = 'L-BFGS-B', 
                                                                options = {'maxiter': 50000,
                                                                           'maxfun': 50000,
                                                                           'maxcor': 50,
                                                                           'maxls': 50,
                                                                           'ftol' : 1.0 * np.finfo(float).eps})

        
        # Initialize Tensorflow variables
        init = tf.global_variables_initializer()
        self.sess.run(init)

    
    # Initialize network weights and biases using Xavier initialization
    def initialize_NN(self, layers): 
        # Custom initialization of 1st layer
        def custom_init(size):
            in_dim = size[0]
            out_dim = size[1]
            custom_stddev = np.sqrt(0.5*(1.0/3.0)/self.X_f.var(0, keepdims = True))
            return tf.Variable(np.random.randn(in_dim, out_dim) * custom_stddev.T, dtype=tf.float32)
    
        # Xavier initialization
        def xavier_init(size):
            in_dim = size[0]
            out_dim = size[1]
            xavier_stddev = 1. / np.sqrt((in_dim + out_dim) / 2.)
            return tf.Variable(tf.random_normal([in_dim, out_dim], dtype=tf.float32) * xavier_stddev, dtype=tf.float32)   
        
        weights = []
        biases = []
        num_layers = len(layers) 
        W = custom_init(size=[layers[0], layers[1]])
        b = tf.Variable(tf.zeros([1,layers[1]], dtype=tf.float32), dtype=tf.float32)
        weights.append(W)
        biases.append(b)            
        for l in range(1,num_layers-1):
            W = xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)        
        return weights, biases
       
           
    # Evaluates the forward pass
    def forward_pass(self, H, layers, weights, biases):
        num_layers = len(layers)
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        H = tf.add(tf.matmul(H, W), b)
        return H
    
    
    # Forward pass for k
    def net_k(self, x1, x2):
        k = self.forward_pass(tf.concat([x1, x2], 1),
                              self.layers_k,
                              self.weights_k, 
                              self.biases_k)
        return k
    
    # Forward pass for u
    def net_u(self, x1, x2):
        u = self.forward_pass(tf.concat([x1, x2], 1),
                              self.layers_u,
                              self.weights_u, 
                              self.biases_u)
        return u
    
    
    # Forward pass for f
    def net_f(self, x1, x2):
        k = self.net_k(x1, x2)
        u = self.net_u(x1, x2)
        u_x1 = tf.gradients(u, x1)[0]
        u_x2 = tf.gradients(u, x2)[0]
        f_1 = tf.gradients(k*u_x1, x1)[0]
        f_2 = tf.gradients(k*u_x2, x2)[0]
        f = f_1 + f_2
        return f
    
    
    # Callback to print the loss at every optimization step
    def callback(self, loss):
        print('Loss:', loss)

       
    # Trains the model by minimizing the loss using L-BFGS
    def train(self): 
        
        # Define a dictionary for associating placeholders with data
        tf_dict = {self.x1_k_tf: self.x1_k, self.x2_k_tf: self.x2_k, self.Yk_tf: self.Y_k,
                   self.x1_u_tf: self.x1_u, self.x2_u_tf: self.x2_u, self.Yu_tf: self.Y_u, 
                   self.x1_f_tf: self.x1_f, self.x2_f_tf: self.x2_f, self.Yf_tf: self.Y_f}

        # Call SciPy's L-BFGS otpimizer
        self.optimizer.minimize(self.sess, 
                                feed_dict = tf_dict,         
                                fetches = [self.loss], 
                                loss_callback = self.callback)

                
    # Evaluates predictions at test points           
    def predict_k(self, X_star): 
        # Center around the origin
        X_star = (X_star - self.lb) - 0.5*(self.ub - self.lb)
        # Predict
        tf_dict = {self.x1_k_tf: X_star[:,0:1], self.x2_k_tf: X_star[:,1:2]}    
        k_star = self.sess.run(self.k_pred, tf_dict) 
        return k_star
    
    # Evaluates predictions at test points           
    def predict_u(self, X_star): 
        # Center around the origin
        X_star = (X_star - self.lb) - 0.5*(self.ub - self.lb)
        # Predict
        tf_dict = {self.x1_u_tf: X_star[:,0:1], self.x2_u_tf: X_star[:,1:2]}    
        u_star = self.sess.run(self.u_pred, tf_dict) 
        return u_star
    
    # Evaluates predictions at test points           
    def predict_f(self, X_star): 
        # Center around the origin
        X_star = (X_star - self.lb) - 0.5*(self.ub - self.lb)
        # Predict
        tf_dict = {self.x1_f_tf: X_star[:,0:1], self.x2_f_tf: X_star[:,1:2]}    
        f_star = self.sess.run(self.f_pred, tf_dict) 
        return f_star
    


class DarcyNet2D_BCs:
    # Initialize the class
    def __init__(self, X_k, Y_k, X_u, Y_u, X_f, Y_f, 
                 X_ubD, Y_ubD, X_ubN, Y_ubN, normal_vec,
                 layers_k, layers_u, lb, ub):
     
        self.lb = lb
        self.ub = ub
        X_k = (X_k - lb) - 0.5*(ub - lb)
        X_u = (X_u - lb) - 0.5*(ub - lb)
        X_f = (X_f - lb) - 0.5*(ub - lb)
        X_ubD = (X_ubD - lb) - 0.5*(ub - lb)
        X_ubN = (X_ubN - lb) - 0.5*(ub - lb)

#        Y_k = (Y_k - Y_k.min()) - 0.5*(Y_k.max() - Y_k.min())
#        Y_u = (Y_u - Y_u.min()) - 0.5*(Y_u.max() - Y_u.min())
#        Y_ubD = (Y_ubD - Y_ubD.min()) -\
#            0.5*(Y_ubD.max() - Y_ubD.min())

        self.x1_k = X_k[:,0:1]
        self.x2_k = X_k[:,1:2]
        self.Y_k = Y_k
        
        self.x1_u = X_u[:,0:1]
        self.x2_u = X_u[:,1:2]
        self.Y_u = Y_u
        
        self.x1_f = X_f[:,0:1]
        self.x2_f = X_f[:,1:2]
        self.Y_f = Y_f
        
        self.X_f = X_f
        
        self.x1_ubD = X_ubD[:,0:1]
        self.x2_ubD = X_ubD[:,1:2]
        self.Y_ubD = Y_ubD
        
        self.x1_ubN = X_ubN[:,0:1]
        self.x2_ubN = X_ubN[:,1:2]
        self.Y_ubN = Y_ubN
        
        self.normal_vec = normal_vec
        
        self.layers_k = layers_k
        self.layers_u = layers_u

        # Initialize network weights and biases  
        self.weights_k, self.biases_k = self.initialize_NN(layers_k)        
        self.weights_u, self.biases_u = self.initialize_NN(layers_u)
        
        # Define Tensorflow session
        self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        
        # Define placeholders and computational graph
        self.x1_k_tf = tf.placeholder(tf.float32, shape=(None, self.x1_k.shape[1]))
        self.x2_k_tf = tf.placeholder(tf.float32, shape=(None, self.x2_k.shape[1]))        
        self.Yk_tf = tf.placeholder(tf.float32, shape=(None, self.Y_k.shape[1]))
        
        self.x1_u_tf = tf.placeholder(tf.float32, shape=(None, self.x1_u.shape[1]))
        self.x2_u_tf = tf.placeholder(tf.float32, shape=(None, self.x2_u.shape[1]))        
        self.Yu_tf = tf.placeholder(tf.float32, shape=(None, self.Y_u.shape[1]))
        
        self.x1_f_tf = tf.placeholder(tf.float32, shape=(None, self.x1_f.shape[1]))
        self.x2_f_tf = tf.placeholder(tf.float32, shape=(None, self.x2_f.shape[1]))        
        self.Yf_tf = tf.placeholder(tf.float32, shape=(None, self.Y_f.shape[1]))
        
        self.x1_ubD_tf = tf.placeholder(tf.float32, shape=(None, self.x1_ubD.shape[1]))
        self.x2_ubD_tf = tf.placeholder(tf.float32, shape=(None, self.x2_ubD.shape[1]))        
        self.YubD_tf = tf.placeholder(tf.float32, shape=(None, self.Y_ubD.shape[1]))
        
        self.x1_ubN_tf = tf.placeholder(tf.float32, shape=(None, self.x1_ubN.shape[1]))
        self.x2_ubN_tf = tf.placeholder(tf.float32, shape=(None, self.x2_ubN.shape[1]))        
        self.YubN_tf = tf.placeholder(tf.float32, shape=(None, self.Y_ubN.shape[1]))
              
        self.normal_vec_tf = tf.placeholder(tf.float32, shape=(None, self.normal_vec.shape[1]))
        
        # Evaluate prediction
        self.k_pred = self.net_k(self.x1_k_tf, self.x2_k_tf)
        self.u_pred = self.net_u(self.x1_u_tf, self.x2_u_tf)
        self.f_pred = self.net_f(self.x1_f_tf, self.x2_f_tf)
        
        self.ubD_pred = self.net_ubD(self.x1_ubD_tf, self.x2_ubD_tf)
        self.ubN_pred = self.net_ubN(self.x1_ubN_tf, self.x2_ubN_tf, self.normal_vec_tf)
        
        # Evaluate loss
        self.loss = tf.losses.mean_squared_error(self.Yk_tf, self.k_pred) + \
                    tf.losses.mean_squared_error(self.Yu_tf, self.u_pred) + \
                    tf.losses.mean_squared_error(self.Yf_tf, self.f_pred) + \
                    tf.losses.mean_squared_error(self.YubD_tf, self.ubD_pred) + \
                    tf.losses.mean_squared_error(self.YubN_tf, self.ubN_pred)
        
        # Define optimizer (use L-BFGS for better accuracy)       
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss, 
                                                                method = 'L-BFGS-B', 
                                                                options = {'maxiter': 50000,
                                                                           'maxfun': 50000,
                                                                           'maxcor': 50,
                                                                           'maxls': 50,
                                                                           'ftol' : 1.0 * np.finfo(float).eps})

        
        # Initialize Tensorflow variables
        init = tf.global_variables_initializer()
        self.sess.run(init)

    
    # Initialize network weights and biases using Xavier initialization
    def initialize_NN(self, layers): 
        # Custom initialization of 1st layer
        def custom_init(size):
            in_dim = size[0]
            out_dim = size[1]
            custom_stddev = np.sqrt(0.5*(1.0/3.0)/self.X_f.var(0, keepdims = True))
            return tf.Variable(np.random.randn(in_dim, out_dim) * custom_stddev.T, dtype=tf.float32)
    
        # Xavier initialization
        def xavier_init(size):
            in_dim = size[0]
            out_dim = size[1]
            xavier_stddev = 1. / np.sqrt((in_dim + out_dim) / 2.)
            return tf.Variable(tf.random_normal([in_dim, out_dim], dtype=tf.float32) * xavier_stddev, dtype=tf.float32)   
        
        weights = []
        biases = []
        num_layers = len(layers) 
        W = custom_init(size=[layers[0], layers[1]])
        b = tf.Variable(tf.zeros([1,layers[1]], dtype=tf.float32), dtype=tf.float32)
        weights.append(W)
        biases.append(b)            
        for l in range(1,num_layers-1):
            W = xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)        
        return weights, biases
       
           
    # Evaluates the forward pass
    def forward_pass(self, H, layers, weights, biases):
        num_layers = len(layers)
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        H = tf.nn.softplus(tf.add(tf.matmul(H, W), b))
        return H
    
    
    # Forward pass for k
    def net_k(self, x1, x2):
        k = self.forward_pass(tf.concat([x1, x2], 1),
                              self.layers_k,
                              self.weights_k, 
                              self.biases_k)
        return k
    # Forward pass for u
    def net_u(self, x1, x2):
        u = self.forward_pass(tf.concat([x1, x2], 1),
                              self.layers_u,
                              self.weights_u, 
                              self.biases_u)
        return u
    
    
    # Forward pass for f
    def net_f(self, x1, x2):
        k = self.net_k(x1, x2)
        u = self.net_u(x1, x2)
        u_x1 = tf.gradients(u, x1)[0]
        u_x2 = tf.gradients(u, x2)[0]
        f_1 = tf.gradients(k*u_x1, x1)[0]
        f_2 = tf.gradients(k*u_x2, x2)[0]
        f = f_1 + f_2
        return f

    
    def net_ubD(self, x1, x2):
        u = self.net_u(x1, x2)
        return u
    
    
    def net_ubN(self, x1, x2, n):
        u = self.net_u(x1, x2)
        k = self.net_k(x1, x2)
        g_1 = k*tf.gradients(u, x1)[0]
        g_2 = k*tf.gradients(u, x2)[0]
        g = g_1*n[:,0:1] + g_2*n[:,1:2]
        return g
    
    
    # Callback to print the loss at every optimization step
    def callback(self, loss):
        print('Loss:', loss)

       
    # Trains the model by minimizing the loss using L-BFGS
    def train(self): 
        
        # Define a dictionary for associating placeholders with data
        tf_dict = {self.x1_k_tf: self.x1_k, self.x2_k_tf: self.x2_k, self.Yk_tf: self.Y_k,
                   self.x1_u_tf: self.x1_u, self.x2_u_tf: self.x2_u, self.Yu_tf: self.Y_u, 
                   self.x1_f_tf: self.x1_f, self.x2_f_tf: self.x2_f, self.Yf_tf: self.Y_f,
                   self.x1_ubD_tf: self.x1_ubD, self.x2_ubD_tf: self.x2_ubD, self.YubD_tf: self.Y_ubD,
                   self.x1_ubN_tf: self.x1_ubN, self.x2_ubN_tf: self.x2_ubN, self.YubN_tf: self.Y_ubN,
                   self.normal_vec_tf: self.normal_vec}

        # Call SciPy's L-BFGS otpimizer
        self.optimizer.minimize(self.sess, 
                                feed_dict = tf_dict,         
                                fetches = [self.loss], 
                                loss_callback = self.callback)

                
    # Evaluates predictions at test points           
    def predict_k(self, X_star): 
        # Center around the origin
        X_star = (X_star - self.lb) - 0.5*(self.ub - self.lb)
        # Predict
        tf_dict = {self.x1_k_tf: X_star[:,0:1], self.x2_k_tf: X_star[:,1:2]}    
        k_star = self.sess.run(self.k_pred, tf_dict) 
        return k_star
    
    # Evaluates predictions at test points           
    def predict_u(self, X_star): 
        # Center around the origin
        X_star = (X_star - self.lb) - 0.5*(self.ub - self.lb)
        # Predict
        tf_dict = {self.x1_u_tf: X_star[:,0:1], self.x2_u_tf: X_star[:,1:2]}    
        u_star = self.sess.run(self.u_pred, tf_dict) 
        return u_star
    
    # Evaluates predictions at test points           
    def predict_f(self, X_star): 
        # Center around the origin
        X_star = (X_star - self.lb) - 0.5*(self.ub - self.lb)
        # Predict
        tf_dict = {self.x1_f_tf: X_star[:,0:1], self.x2_f_tf: X_star[:,1:2]}    
        f_star = self.sess.run(self.f_pred, tf_dict) 
        return f_star
    
class DarcyNet2D_BCs_Y:
    # Initialize the class
    def __init__(self, X_k, Y_k, X_u, Y_u, X_f, Y_f, 
                 X_ubD, Y_ubD, X_ubN, Y_ubN, normal_vec,
                 layers_k, layers_u, lb, ub):
     
        self.lb = lb
        self.ub = ub
        X_k = (X_k - lb) - 0.5*(ub - lb)
        X_u = (X_u - lb) - 0.5*(ub - lb)
        X_f = (X_f - lb) - 0.5*(ub - lb)
        X_ubD = (X_ubD - lb) - 0.5*(ub - lb)
        X_ubN = (X_ubN - lb) - 0.5*(ub - lb)

#        Y_k = (Y_k - Y_k.min()) - 0.5*(Y_k.max() - Y_k.min())
#        Y_u = (Y_u - Y_u.min()) - 0.5*(Y_u.max() - Y_u.min())
#        Y_ubD = (Y_ubD - Y_ubD.min()) -\
#            0.5*(Y_ubD.max() - Y_ubD.min())

        self.x1_k = X_k[:,0:1]
        self.x2_k = X_k[:,1:2]
        self.Y_k = Y_k
        
        self.x1_u = X_u[:,0:1]
        self.x2_u = X_u[:,1:2]
        self.Y_u = Y_u
        
        self.x1_f = X_f[:,0:1]
        self.x2_f = X_f[:,1:2]
        self.Y_f = Y_f
        
        self.X_f = X_f
        
        self.x1_ubD = X_ubD[:,0:1]
        self.x2_ubD = X_ubD[:,1:2]
        self.Y_ubD = Y_ubD
        
        self.x1_ubN = X_ubN[:,0:1]
        self.x2_ubN = X_ubN[:,1:2]
        self.Y_ubN = Y_ubN
        
        self.normal_vec = normal_vec
        
        self.layers_k = layers_k
        self.layers_u = layers_u

        # Initialize network weights and biases  
        self.weights_k, self.biases_k = self.initialize_NN(layers_k)        
        self.weights_u, self.biases_u = self.initialize_NN(layers_u)
        
        # Define Tensorflow session
        self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        
        # Define placeholders and computational graph
        self.x1_k_tf = tf.placeholder(tf.float32, shape=(None, self.x1_k.shape[1]))
        self.x2_k_tf = tf.placeholder(tf.float32, shape=(None, self.x2_k.shape[1]))        
        self.Yk_tf = tf.placeholder(tf.float32, shape=(None, self.Y_k.shape[1]))
        
        self.x1_u_tf = tf.placeholder(tf.float32, shape=(None, self.x1_u.shape[1]))
        self.x2_u_tf = tf.placeholder(tf.float32, shape=(None, self.x2_u.shape[1]))        
        self.Yu_tf = tf.placeholder(tf.float32, shape=(None, self.Y_u.shape[1]))
        
        self.x1_f_tf = tf.placeholder(tf.float32, shape=(None, self.x1_f.shape[1]))
        self.x2_f_tf = tf.placeholder(tf.float32, shape=(None, self.x2_f.shape[1]))        
        self.Yf_tf = tf.placeholder(tf.float32, shape=(None, self.Y_f.shape[1]))
        
        self.x1_ubD_tf = tf.placeholder(tf.float32, shape=(None, self.x1_ubD.shape[1]))
        self.x2_ubD_tf = tf.placeholder(tf.float32, shape=(None, self.x2_ubD.shape[1]))        
        self.YubD_tf = tf.placeholder(tf.float32, shape=(None, self.Y_ubD.shape[1]))
        
        self.x1_ubN_tf = tf.placeholder(tf.float32, shape=(None, self.x1_ubN.shape[1]))
        self.x2_ubN_tf = tf.placeholder(tf.float32, shape=(None, self.x2_ubN.shape[1]))        
        self.YubN_tf = tf.placeholder(tf.float32, shape=(None, self.Y_ubN.shape[1]))
              
        self.normal_vec_tf = tf.placeholder(tf.float32, shape=(None, self.normal_vec.shape[1]))
        
        # Evaluate prediction
        self.k_pred = self.net_k(self.x1_k_tf, self.x2_k_tf)
        self.u_pred = self.net_u(self.x1_u_tf, self.x2_u_tf)
        self.f_pred = self.net_f(self.x1_f_tf, self.x2_f_tf)
        
        self.ubD_pred = self.net_ubD(self.x1_ubD_tf, self.x2_ubD_tf)
        self.ubN_pred = self.net_ubN(self.x1_ubN_tf, self.x2_ubN_tf, self.normal_vec_tf)
        
        # Evaluate loss
        self.loss = tf.losses.mean_squared_error(self.Yk_tf, self.k_pred) + \
                    tf.losses.mean_squared_error(self.Yu_tf, self.u_pred) + \
                    tf.losses.mean_squared_error(self.Yf_tf, self.f_pred) + \
                    tf.losses.mean_squared_error(self.YubD_tf, self.ubD_pred) + \
                    tf.losses.mean_squared_error(self.YubN_tf, self.ubN_pred)
        
        # Define optimizer (use L-BFGS for better accuracy)       
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss, 
                                                                method = 'L-BFGS-B', 
                                                                options = {'maxiter': 50000,
                                                                           'maxfun': 50000,
                                                                           'maxcor': 50,
                                                                           'maxls': 50,
                                                                           'ftol' : 1.0 * np.finfo(float).eps})

        
        # Initialize Tensorflow variables
        init = tf.global_variables_initializer()
        self.sess.run(init)

    
    # Initialize network weights and biases using Xavier initialization
    def initialize_NN(self, layers): 
        # Custom initialization of 1st layer
        def custom_init(size):
            in_dim = size[0]
            out_dim = size[1]
            custom_stddev = np.sqrt(0.5*(1.0/3.0)/self.X_f.var(0, keepdims = True))
            return tf.Variable(np.random.randn(in_dim, out_dim) * custom_stddev.T, dtype=tf.float32)
    
        # Xavier initialization
        def xavier_init(size):
            in_dim = size[0]
            out_dim = size[1]
            xavier_stddev = 1. / np.sqrt((in_dim + out_dim) / 2.)
            return tf.Variable(tf.random_normal([in_dim, out_dim], dtype=tf.float32) * xavier_stddev, dtype=tf.float32)   
        
        weights = []
        biases = []
        num_layers = len(layers) 
        W = custom_init(size=[layers[0], layers[1]])
        b = tf.Variable(tf.zeros([1,layers[1]], dtype=tf.float32), dtype=tf.float32)
        weights.append(W)
        biases.append(b)            
        for l in range(1,num_layers-1):
            W = xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)        
        return weights, biases
       
           
    # Evaluates the forward pass
    def forward_pass(self, H, layers, weights, biases):
        num_layers = len(layers)
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        H = tf.nn.softplus(tf.add(tf.matmul(H, W), b))
        return H
    
    
    # Forward pass for k
    def net_k(self, x1, x2):
        k = self.forward_pass(tf.concat([x1, x2], 1),
                              self.layers_k,
                              self.weights_k, 
                              self.biases_k)
        return tf.log(k)

    # Forward pass for u
    def net_u(self, x1, x2):
        u = self.forward_pass(tf.concat([x1, x2], 1),
                              self.layers_u,
                              self.weights_u, 
                              self.biases_u)
        return u
    
    
    # Forward pass for f
    def net_f(self, x1, x2):
        y = self.net_k(x1, x2)
        exp_y = tf.exp(y)
        grady_dot_gradu = tf.gradients(y, x1)[0]+tf.gradients(y, x2)[0]
        u = self.net_u(x1, x2)
        u_x1 = tf.gradients(u, x1)
        u_x2 = tf.gradients(u, x2)
        grady_dot_gradu = u_x1*tf.gradients(y, x1)[0]+\
            u_x2*tf.gradients(y, x2)[0]
        nabla_u = tf.linalg.trace(tf.hessians(u, [x1,x2]))
        f = exp_y*(nabla_u)+exp_y*grady_dot_gradu
        return f
    
    def net_ubD(self, x1, x2):
        u = self.net_u(x1, x2)
        return u
    
    
    def net_ubN(self, x1, x2, n):
        u = self.net_u(x1, x2)
        k = self.net_k(x1, x2)
        g_1 = k*tf.gradients(u, x1)[0]
        g_2 = k*tf.gradients(u, x2)[0]
        g = g_1*n[:,0:1] + g_2*n[:,1:2]
        return g
    
    
    # Callback to print the loss at every optimization step
    def callback(self, loss):
        print('Loss:', loss)

       
    # Trains the model by minimizing the loss using L-BFGS
    def train(self): 
        
        # Define a dictionary for associating placeholders with data
        tf_dict = {self.x1_k_tf: self.x1_k, self.x2_k_tf: self.x2_k, self.Yk_tf: self.Y_k,
                   self.x1_u_tf: self.x1_u, self.x2_u_tf: self.x2_u, self.Yu_tf: self.Y_u, 
                   self.x1_f_tf: self.x1_f, self.x2_f_tf: self.x2_f, self.Yf_tf: self.Y_f,
                   self.x1_ubD_tf: self.x1_ubD, self.x2_ubD_tf: self.x2_ubD, self.YubD_tf: self.Y_ubD,
                   self.x1_ubN_tf: self.x1_ubN, self.x2_ubN_tf: self.x2_ubN, self.YubN_tf: self.Y_ubN,
                   self.normal_vec_tf: self.normal_vec}

        # Call SciPy's L-BFGS otpimizer
        self.optimizer.minimize(self.sess, 
                                feed_dict = tf_dict,         
                                fetches = [self.loss], 
                                loss_callback = self.callback)

                
    # Evaluates predictions at test points           
    def predict_k(self, X_star): 
        # Center around the origin
        X_star = (X_star - self.lb) - 0.5*(self.ub - self.lb)
        # Predict
        tf_dict = {self.x1_k_tf: X_star[:,0:1], self.x2_k_tf: X_star[:,1:2]}    
        k_star = self.sess.run(self.k_pred, tf_dict) 
        return k_star
    
    # Evaluates predictions at test points           
    def predict_u(self, X_star): 
        # Center around the origin
        X_star = (X_star - self.lb) - 0.5*(self.ub - self.lb)
        # Predict
        tf_dict = {self.x1_u_tf: X_star[:,0:1], self.x2_u_tf: X_star[:,1:2]}    
        u_star = self.sess.run(self.u_pred, tf_dict) 
        return u_star
    
    # Evaluates predictions at test points           
    def predict_f(self, X_star): 
        # Center around the origin
        X_star = (X_star - self.lb) - 0.5*(self.ub - self.lb)
        # Predict
        tf_dict = {self.x1_f_tf: X_star[:,0:1], self.x2_f_tf: X_star[:,1:2]}    
        f_star = self.sess.run(self.f_pred, tf_dict) 
        return f_star
    

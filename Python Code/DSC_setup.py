#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import ot
import cvxpy as cp

# Supplementary Packages
#import scipy.stats as stats
#import seaborn as sns
#import scipy.special as sps
#import time as t


# ## Functions

# In[62]:


def baryc_proj(source, target, method):
    
    n1 = source.shape[0]
    n2 = target.shape[0]   
    p = source.shape[1]
    a_ones, b_ones = np.ones((n1,)) / n1, np.ones((n2,)) / n2
    
    M = ot.dist(source, target)
    M = M.astype('float64')
    M /= M.max()
    
    if method == 'emd':
        OTplan = ot.emd(a_ones, b_ones, M, numItermax = 1e7)
        
    elif method == 'entropic':
        OTplan = ot.bregman.sinkhorn_stabilized(a_ones, b_ones, M, reg = 5*1e-3)
    
    # initialization
    OTmap = np.empty((0, p))

    for i in range(n1):
        
        # normalization
        OTplan[i,:] = OTplan[i,:] / sum(OTplan[i,:])
    
        # conditional expectation
        OTmap = np.vstack([OTmap, (target.T @ OTplan[i,:])])
    
    OTmap = np.array(OTmap)
    
    return(OTmap)


# In[63]:


def DSCreplication(target, controls, method = 'emd'):
    
    n = target.shape[0]
    d = target.shape[1]
    J = len(controls)
    
    
    # Barycentric Projection
    G_list = []
    for i in range(len(controls)):
        G_list.append(baryc_proj(target, controls[i], method))
    
    
    # Function to optimize
    def to_optimize(lambdas):
                
        ans = lambdas[0] * (G_list[0] - target)
        for i in range(J-1):
            ans += lambdas[i+1] * (G_list[i+1] - target)
        
        return sum(sum(ans**2)) / n

    
    # Obtain optimal weights
    mylambda = cp.Variable(J)

    objective = cp.Minimize(to_optimize(mylambda))
    constraints = [mylambda >= 0, mylambda <= 1, cp.sum(mylambda) == 1]

    prob = cp.Problem(objective, constraints)
    prob.solve()

    
    weights = mylambda.value
    projection = weights[0]*G_list[0]
    for j in range(J-1):
        projection += weights[j+1]*G_list[j+1]
    
    
    return(weights, projection)



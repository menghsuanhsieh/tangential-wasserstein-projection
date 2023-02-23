#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import ot
import cvxpy as cp
import multiprocess

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
    
    OTmap = np.array(OTmap).astype('float32')
    
    return(OTmap)


# In[2]:


def tan_wass_proj(target, controls, method = 'emd'):
    
    n = target.shape[0]
    d = target.shape[1]
    J = len(controls)
    S = np.mean(target)*n*d*J # Stabilizer: to ground the optimization objective
    
    
    # Barycentric Projection
    G_list = []
    proj_list = []
    for i in range(len(controls)):
        temp = baryc_proj(target, controls[i], method)
        G_list.append(temp)
        proj_list.append(temp - target)
    
    
    # Obtain optimal weights
    mylambda = cp.Variable(J)

    objective = cp.Minimize(
                    cp.sum_squares(
                    cp.sum([a*b for a,b in zip(mylambda, proj_list)], axis = 0))/S
                    )
    
    constraints = [mylambda >= 0, mylambda <= 1, cp.sum(mylambda) == 1]

    prob = cp.Problem(objective, constraints)
    prob.solve()
    
    # outputs
    weights = mylambda.value
    replication = sum([a*b for a,b in zip(weights, G_list)])
    
    
    return(weights, replication)


# ### barycentric projection step parallelized

# In[3]:


def tan_wass_proj2(target, controls, method = 'emd', projtype = 'wass'):
    
    n = target.shape[0]
    d = target.shape[1]
    J = len(controls)
    S = np.mean(target)*n*d*J # Stabilizer: to ground the optimization objective
    
    
    # Barycentric Projection
    to_multiprocess = [(target, controls[i], method) for i in range(len(controls))]
    
    with multiprocess.Pool() as pool:
        G_list = pool.starmap(baryc_proj, to_multiprocess)
    proj_list = G_list - target
    
    
    # Obtain optimal weights
    mylambda = cp.Variable(J)

    objective = cp.Minimize(
                    cp.sum_squares(
                    cp.sum([a*b for a,b in zip(mylambda, proj_list)], axis = 0))/S
                    )
    
    constraints = [mylambda >= 0, mylambda <= 1, cp.sum(mylambda) == 1]

    prob = cp.Problem(objective, constraints)
    prob.solve()

    #outputs
    weights = mylambda.value
    projection = sum([a*b for a,b in zip(weights, G_list)])

    
    return(weights, projection)


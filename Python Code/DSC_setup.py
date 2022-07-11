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

# In[22]:


def baryc_proj(source, target, method = 'emd'):
    
    n1 = source.shape[0]
    n2 = target.shape[0]   
    p = source.shape[1]
    a_ones, b_ones = np.ones((n1,)) / n1, np.ones((n2,)) / n2
    
    M = ot.dist(source, target)
    M = M.astype('float64')
    M /= M.max()
    
    if method == 'emd':
        OTplan = ot.emd(a_ones, b_ones, M)
        
    elif method == 'entropic':
        OTplan = ot.bregman.sinkhorn(a_ones, b_ones, M, reg = 5*1e-3)

    
    # initialization
    OTmap = np.empty((0, p))

    for i in range(n1):
        
        # normalization
        OTplan[i,:] = OTplan[i,:] / sum(OTplan[i,:])

        # calculate conditional expectation for each sample in the source dist.
        OTmap = np.vstack([OTmap, (target.T @ OTplan[i,:])])
    
    
    OTmap = np.array(OTmap)
    full_sum = sum(sum((OTmap - source)**2)) # for optimization in later steps
    
    
    return(OTmap, full_sum)


# In[23]:


def DSCreplication(target, controls, method = 'emd'):
    
    n = target.shape[0]
    d = target.shape[1]
    J = len(controls)
    

    # Barycentric Projection
    distances = []
    G_list = []
    
    for i in range(len(controls)):
        distances.append(baryc_proj(target, controls[i], method)[1])
        G_list.append(baryc_proj(target, controls[i], method)[0])
    
    distances = np.array(distances)
    
    
    # Obtain Optimal Weights
    mylambda = cp.Variable(J)

    objective = cp.Minimize(distances@cp.square(mylambda) / n)
    constraints = [mylambda >= 0, mylambda <= 1, cp.sum(mylambda) == 1]
    prob = cp.Problem(objective, constraints)
    prob.solve()
    
    
    # Generate Replication
    weights = mylambda.value
    replication = weights[0]*G_list[0]
    for j in range(J-1):
        replication += weights[j+1]*G_list[j+1]
    
    
    return(weights, replication)


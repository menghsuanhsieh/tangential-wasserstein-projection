#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from numpy import random

import pandas as pd
import matplotlib.pylab as plt
import ot
import ot.plot
import cvxpy as cp

import scipy.stats as stats
import seaborn as sns
import scipy.special as sps
import time as t


# ## Functions

# In[2]:


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


# In[3]:
    

def DSCreplication(target, controls, method = 'emd'):
    
    n = target.shape[0]
    d = target.shape[1]
    J = len(controls)
    S = np.mean(target)*n*d*J
    
    
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
                    cp.sum([a*b for a,b in zip(mylambda, proj_list)], axis = 0)**2)/S
                    )
    
    constraints = [mylambda >= 0, mylambda <= 1, cp.sum(mylambda) == 1]

    prob = cp.Problem(objective, constraints)
    prob.solve()
    
    weights = mylambda.value
    projection = sum([a*b for a,b in zip(weights, G_list)])
    
    
    return(weights, projection)

# ## LEGO Blocks

# In[4]:


def reshaped_image(file_name, sample = False):
        
    image = plt.imread(file_name)
    
    if sample: 
        image = image[::10, ::10, :]
    
    p = image.shape[0]
    q = image.shape[1]
    r = image.shape[2]
    
    image = image.reshape((p*q, r))
    return(image)


# In[5]:

import os, glob
imglist = []

path = 'Users/pablomjlee/Documents/DSC/workingData'
for file in sorted(glob.glob("workingData/*.png")):
    imglist.append(reshaped_image(file))
    
    
blockT = imglist[0]
blockCs = imglist[1:]


# In[6]:


ts = t.time()

weightsb, replicationb = DSCreplication(blockT, blockCs, 'emd')

te = t.time() - ts


# In[7]:
print(te)
print(weightsb)

# In[9]:
from PIL import Image

replicated = (replicationb.reshape((200,200,4))*255).astype('uint8')
replicated = Image.fromarray(replicated, mode ='RGBA')

replicated.save('LEGOreplication.png')


# In[10]:



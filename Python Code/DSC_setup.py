# coding: utf-8

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


def baryc_proj(source, target, method):
    
    # This function obtains the barycentric projection of the transport plan
    
    ## Output
    #####
    # OTmap: barycentric projection of the transport plan;
    #        Each row is the data point to which the source data points are transported
    
    ## Parameters
    #####
    # source: source distribution(array-like)
    # target: target distribution(array-like)
    # method: default = 'emd'
    #   'emd': uses the earth mover's distance solution
    #   'entropic': uses the entropic regularized OT solution
    # Note: the entropic version may be unstable and very slow.
    #       the emd version is recommended.
    
    
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
    
        # obtain conditional expectation
        OTmap = np.vstack([OTmap, (target.T @ OTplan[i,:])])
    
    OTmap = np.array(OTmap).astype('float32')
    
    return(OTmap)




def DSCreplication(target, controls, method = 'emd'):
    
    # This function obtains the optimal weights and the projection.
    
    ## Output
    #####
    # weights: optimal weights(list object)
    # projection: replication of the target distribution(numpy array)
    
    ## Parameters
    #####
    # target: target distribution(array-like)
    # controls: control distributions(list of array-like objects)
    # method: default = 'emd', parameter for `baryc_proj`
    
    
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
                    cp.sum([a*b for a,b in zip(mylambda, proj_list)],
                            axis = 0)**2)/S
                    )
    
    constraints = [mylambda >= 0, mylambda <= 1, cp.sum(mylambda) == 1]

    prob = cp.Problem(objective, constraints)
    prob.solve()
    
    weights = mylambda.value
    projection = sum([a*b for a,b in zip(weights, G_list)])
    
    return(weights, projection)


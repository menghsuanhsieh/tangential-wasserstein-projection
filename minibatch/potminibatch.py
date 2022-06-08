#!/usr/bin/env python
# coding: utf-8

import numpy as np
from numpy import random

import pandas as pd
import matplotlib.pylab as plt
import ot
import ot.plot

import scipy.stats as stats
import seaborn as sns
import scipy.special as sps
import time as t


# minibatch functions

def mini_batch(data, weights, batch_size, N_data):

    id_batch = np.random.choice(N_data, batch_size, replace=False, p=weights)
    sub_weights = ot.unif(batch_size)
    return data[id_batch], sub_weights, id_batch


def update_gamma(gamma, gamma_mb, id_a, id_b):

    for i,i2 in enumerate(id_a):
        for j,j2 in enumerate(id_b):
            gamma[i2,j2] += gamma_mb[i][j]
    return gamma


def get_stoc_gamma(xs, xt, a, b, m1, m2, K, M, lambd=5*1e-3,
                         method='emd'):

    stoc_gamma = np.zeros((np.shape(xs)[0], np.shape(xt)[0]))
    Ns = np.shape(xs)[0]
    Nt = np.shape(xt)[0]
    
    for i in range(K):
        sub_xs, sub_weights_a, id_a = mini_batch(xs, a, m1, Ns)
        sub_xt, sub_weights_b, id_b = mini_batch(xt, b, m2, Nt)

        if method == 'emd':
            sub_M = M[id_a,:][:,id_b].copy()
            G0 = ot.emd(sub_weights_a, sub_weights_b, sub_M, numItermax = 1000000)

        elif method == 'entropic':
            sub_M = M[id_a, :][:, id_b]
            G0 = ot.sinkhorn(sub_weights_a, sub_weights_b, sub_M, lambd)
        
        elif method == 'stable':
            sub_M = M[id_a, :][:, id_b]
            G0 = ot.bregman.sinkhorn_stabilized(sub_weights_a, sub_weights_b, sub_M, lambd)

        stoc_gamma = update_gamma(stoc_gamma, G0, id_a, id_b)

    return (1/K) * stoc_gamma


# setup:
n1 = n2 = 50000
dim = 50

covmat = np.zeros((dim, dim))
np.fill_diagonal(covmat, 1)

covmat2 = np.full((dim, dim), 0.8)
np.fill_diagonal(covmat2, 1)

a_ones, b_ones = np.ones((n1,)) / n1, np.ones((n2,)) / n2

source = random.multivariate_normal(mean = [0]*dim, cov = covmat, size = n1)
target = random.multivariate_normal(mean = [10]*dim, cov = covmat2, size = n2)


# cost matrix
M = ot.dist(source, target)
M /= M.max()
#plt.imshow(M)


# minibatch sinkhorn
Gmb = get_stoc_gamma(source, target, a_ones, b_ones, 25000, 25000,
        1000, M, method = 'entropic')

plt.imsave('Gmb.png', Gmb[0:1000,0:1000])
# we cut out 1000x1000 to visually check the matchings -- if n > 10000, a matching in a nxn plot is not distinguishable



# actual - choose one: for comparison with the minibatch method
#G0 = ot.sinkhorn(a_ones, b_ones, M, reg = 5*1e-3)
#G0 = ot.emd(a_ones, b_ones, M, numItermax = 1000000)

#plt.imsave('G0.png', G0[0:1000,0:1000])




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


#1 : Plot, shows matchings from each point in source to distribution -- see pot website example "2D empirical"
#NOTE: This shows the plot when you run the code, but it does not show up on GreatLakes output - I need to figure out a way to save this image

ot.plot.plot2D_samples_mat(source[0:100,:], target[0:100,:], Gmb[0:100,0:100], c=[.5, .5, 1])
plt.plot(source[0:100,0], source[0:100,1], '+b', label='Source samples')
plt.plot(target[0:100,0], target[0:100,1], 'xr', label='Target samples')

ot.plot.plot2D_samples_mat(source[0:100,:], target[0:100,:], G0[0:100,0:100], c=[.5, .5, 1])
plt.plot(source[0:100,0], source[0:100,1], '+b', label='Source samples')
plt.plot(target[0:100,0], target[0:100,1], 'xr', label='Target samples')


#2: Generate Mixed Multivariate Gaussian distribution

mean = [0] * n
cov1 = np.zeros((n, dim))
np.fill_diagonal(cov1, 1)

def mixed_multi_gauss(mean1, mean2, cov1, cov2, samplesize, partition):
    
    gauss1 = random.multivariate_normal(mean = mean1, cov = cov1, size = samplesize * partition)
    gauss2 = random.multivariate_normal(mean = mean2, cov = cov2, size = samplesize * (1 - partition))
    
    mixed = np.concatenate((gauss1, gauss2), axis = 0)
    np.random.shuffle(mixed)
    
    return(mixed)







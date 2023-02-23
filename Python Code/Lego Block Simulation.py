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
from PIL import Image
import twp_utils as twp


#import scipy.stats as stats
#import seaborn as sns
#import scipy.special as sps
#import time as t



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

for file in sorted(glob.glob("Data/Lego_bricks/*.png")):
    imglist.append(reshaped_image(file))
    
    
blockT = imglist[0]
blockCs = imglist[1:]


# In[6]:


ts = t.time()

weightsb, replicationb = twp.tan_wass_proj(blockT, blockCs, 'emd')

te = t.time() - ts


# In[7]:
print(te)
print(weightsb)

# In[9]:
replicated = (replicationb.reshape((200,200,4))*255).astype('uint8')
replicated = Image.fromarray(replicated, mode ='RGBA')

replicated.save('LEGOreplication.png')


# In[10]:



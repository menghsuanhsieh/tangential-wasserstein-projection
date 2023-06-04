#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import ot
import cvxpy as cp
import seaborn as sns

from twp_utils import baryc_proj, tan_wass_proj

# Supplementary Packages
#import scipy.stats as stats
#import scipy.special as sps
#import time as t


# ## Functions

# ## Medicare Data

# In[4]:


def read_medicaid(file_name, columns, sample = True):
    
    df = pd.read_csv(file_name)[columns]
    
    if sample:
        df = df.sample(1500, random_state = 31)
    
    return(np.array(df))


# In[26]:


import os, glob
medidata = []

columns1 = ['HINSCAID','EMPSTAT','UHRSWORK','INCWAGE']
for file in sorted(glob.glob("workingData/medicaid/*.csv")):
    
    df = read_medicaid(file, columns1, sample = False)
    nrow = len(df)
    
    if nrow > 30000: # we limit the data size to 30k for computational issues.

        np.random.shuffle(df)
        df = df[:30000]
        
    medidata.append(df)

medidata.insert(0, medidata.pop(5)) # Move Montana to front of list
medi_target = medidata[0]
medi_controls = medidata[1:]


#for counterfactual exercise
mt_years = read_medicaid("workingData/medicaid/MT.csv", ['YEAR'])


# In[27]:


for i in range(len(medi_controls)):
    
    print(len(medi_controls[i]))


# ### Test Run

# In[6]:


medi_weights, medi_projection = tan_wass_proj(medi_target, medi_controls)

# round integer columns
medi_projection[:,0:2] = medi_projection[:,0:2].round(decimals = 0).astype('int32')


# In[7]:


medi_target[0:5,:]


# In[8]:


medi_projection[0:5,:]


# In[9]:


medi_weights




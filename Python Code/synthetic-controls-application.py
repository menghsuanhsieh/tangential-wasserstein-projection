# !pip install cvxpy # install this on first try
# !pip install pandas
# !pip install pot

import ot
import pandas as pd

import numpy as np
from numpy import random
import os
import glob

import matplotlib.pylab as plt
import ot.plot
import cvxpy as cp

import scipy.stats as stats
import seaborn as sns
import scipy.special as sps
import time as t

# importing stuff
path = os.getcwd()
csv_files = glob.glob(os.path.join(path, "*.csv"))
files_list = glob.glob('*.{}'.format("csv"))

files_names = [s.replace(".csv", "") for s in files_list]
files_names

files_dict = {}

for k in range(len(csv_files)):
    
    key = files_names[k]
    
    # read the csv file
    df = pd.read_csv(csv_files[k])
        
    # store in dictionary
    files_dict[key] = df

# defining target
target = files_dict["MT"][["HINSCAID", "EMPSTAT", "UHRSWORK", "INCWAGE"]]
target # view the target df

# defining controls
control1 = files_dict["SC"][["HINSCAID", "EMPSTAT", "UHRSWORK", "INCWAGE"]]
control2 = files_dict["FL"][["HINSCAID", "EMPSTAT", "UHRSWORK", "INCWAGE"]]
control3 = files_dict["TN"][["HINSCAID", "EMPSTAT", "UHRSWORK", "INCWAGE"]]
control4 = files_dict["NC"][["HINSCAID", "EMPSTAT", "UHRSWORK", "INCWAGE"]]
control5 = files_dict["WY"][["HINSCAID", "EMPSTAT", "UHRSWORK", "INCWAGE"]]
control6 = files_dict["GA"][["HINSCAID", "EMPSTAT", "UHRSWORK", "INCWAGE"]]
control7 = files_dict["MS"][["HINSCAID", "EMPSTAT", "UHRSWORK", "INCWAGE"]]
control8 = files_dict["AL"][["HINSCAID", "EMPSTAT", "UHRSWORK", "INCWAGE"]]
control9 = files_dict["WI"][["HINSCAID", "EMPSTAT", "UHRSWORK", "INCWAGE"]]
control10 = files_dict["SD"][["HINSCAID", "EMPSTAT", "UHRSWORK", "INCWAGE"]]
control11 = files_dict["KS"][["HINSCAID", "EMPSTAT", "UHRSWORK", "INCWAGE"]]
control12 = files_dict["TX"][["HINSCAID", "EMPSTAT", "UHRSWORK", "INCWAGE"]]

# code for implementation

## barycentric projection
def baryc_proj(source, target):
    
    n1 = source.shape[0]
    n2 = target.shape[0]   
    p = source.shape[1]
    a_ones, b_ones = np.ones((n1,)) / n1, np.ones((n2,)) / n2
    
    M = ot.dist(source, target)
    M = M.astype('float64')
    M /= M.max()
    OTplan = ot.bregman.sinkhorn_stabilized(a_ones, b_ones, M, reg = 5*1e-3)# try emd
    
    # initialization
    OTmap = np.empty((0, p))

    for i in range(n1):
        
        # normalization
        OTplan[i,:] = OTplan[i,:] / sum(OTplan[i,:])
    
        # conditional expectation
        OTmap = np.vstack([OTmap, (target.T @ OTplan[i,:])])
    
    OTmap = np.array(OTmap)
    
    return(OTmap)

## optimization routine

def to_optimize(lambdas):
    
    ans = []
    for i in range(J):
        temp = lambdas[i] * (G_list[i] - globtarget)
        ans.append(sum(sum(temp**2)))
    
    return sum(ans) / n

## optimal weights calculations

def get_optimal_weights(Glist):
        
    mylambda = cp.Variable(J)

    objective = cp.Minimize(to_optimize(mylambda))
    constraints = [mylambda >= 0, mylambda <= 1, cp.sum(mylambda) == 1]

    prob = cp.Problem(objective, constraints)
    prob.solve()

    weights = mylambda.value
    
    return(weights)

# full implementation

def DSCreplication(target, controls):
    
    global n, d, J, globtarget
    n = target.shape[0]
    d = target.shape[1]
    J = len(controls)
    globtarget = target

    # Barycentric Projection
    global G_list
    G_list = []
    for i in range(len(controls)):
        G_list.append(baryc_proj(target, controls[i]))
    
    # Obtain optimal weights(to_optimze needs to be pre-defined)
    weights = get_optimal_weights(G_list)
    projection = weights[0]*G_list[0]
    for j in range(J-1):
        projection += weights[j+1]*G_list[j+1]
    
    
    return(weights, projection)

def DSCreplicationV2(target, controls):
    
    n = target.shape[0]
    d = target.shape[1]
    J = len(controls)
    
    
    # Barycentric Projection
    G_list = []
    for i in range(len(controls)):
        G_list.append(baryc_proj(target, controls[i]))
    
    
    # Function to optimize
    def to_optimize(lambdas):
                
        ans = []
        for i in range(J):
            temp = lambdas[i] * (G_list[i] - target)
            ans.append(sum(sum(temp**2)))
    
        return sum(ans) / n

    
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

# testing below-------------------

n1 = n2 = 100
dim = 3

covmat = np.zeros((dim, dim))
np.fill_diagonal(covmat, 1)

covmat2 = np.full((dim, dim), 0.3)
np.fill_diagonal(covmat2, 1)

covmat3 = np.full((dim, dim), 0.5)
np.fill_diagonal(covmat3, 1)

covmat4 = np.full((dim, dim), 0.8)
np.fill_diagonal(covmat4, 1)

a_ones, b_ones = np.ones((n1,)) / n1, np.ones((n2,)) / n2

c1 = random.multivariate_normal(mean = [20, 20, 20], cov = covmat4, size = n1)
c2 = random.multivariate_normal(mean = [100, 100, 100], cov = covmat4, size = n1)
c3 = random.multivariate_normal(mean = [50, 50, 50], cov = covmat3, size = n1)
cs = [c1, c2, c3]

TARGET = random.multivariate_normal(mean = [25]*dim, cov = covmat, size = n1)

type(cs[1])

type(TARGET)

ts = t.time()

weights1, projection1 = DSCreplication(TARGET, cs)

print(t.time() - ts)
print(weights1)

# full implementation and estimation=====================

states_controls = [control1.iloc[0:2000,].to_numpy(), control2.iloc[0:1500,].to_numpy()]
type(states_controls)

target = target.iloc[0:1000,].to_numpy()

weights_s, projection_s = DSCreplicationV2(target, states_controls)

weights_s
projection_s
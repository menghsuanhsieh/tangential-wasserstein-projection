# -*- coding: utf-8 -*-
"""
=============================
OT for image color transport
=============================

Initial Date: May 2, 2022 Monday

@author: rexhsieh

"""

import os
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
import ot


##### "set seed" for the following simulation stuff
rng = np.random.RandomState(42)

def image_to_matrix(img):
    """Converts an image to matrix (one pixel per line)"""
    return img.reshape((img.shape[0] * img.shape[1], img.shape[2]))


def matrix_to_image(X, shape):
    """Converts back a matrix to an image"""
    return X.reshape(shape)


def minmax(img):
    return np.clip(img, 0, 1)


##############################################################################
# Generate data
# -------------

# Loading images
I1 = plt.imread('Dropbox (University of Michigan)/TWP/tangential-wasserstein-projection/Test Images/apple.jpeg').astype(np.float64) / 256
I2 = plt.imread('Dropbox (University of Michigan)/TWP/tangential-wasserstein-projection/Test Images/pear.jpeg').astype(np.float64) / 256

X1 = image_to_matrix(I1)
X2 = image_to_matrix(I2)

# training samples
nb = 500
idx1 = rng.randint(X1.shape[0], size=(nb,))
idx2 = rng.randint(X2.shape[0], size=(nb,))

Xs = X1[idx1, :]
Xt = X2[idx2, :]


##############################################################################
# Plot original image
# -------------------

plt.figure(1, figsize=(6.4, 3))

plt.subplot(1, 2, 1)
plt.imshow(I1)
plt.axis('off')
plt.title('Image 1')

plt.subplot(1, 2, 2)
plt.imshow(I2)
plt.axis('off')
plt.title('Image 2')


##############################################################################
# Scatter plot of colors
# ----------------------

plt.figure(2, figsize=(6.4, 3))

plt.subplot(1, 2, 1)
plt.scatter(Xs[:, 0], Xs[:, 2], c=Xs)
plt.axis([0, 1, 0, 1])
plt.xlabel('Red')
plt.ylabel('Blue')
plt.title('Image 1')

plt.subplot(1, 2, 2)
plt.scatter(Xt[:, 0], Xt[:, 2], c=Xt)
plt.axis([0, 1, 0, 1])
plt.xlabel('Red')
plt.ylabel('Blue')
plt.title('Image 2')
plt.tight_layout()


##############################################################################
# Instantiate the different transport algorithms and fit them
# -----------------------------------------------------------

# "Earth MOver" distance Transport
ot_emd = ot.da.EMDTransport()
ot_emd.fit(Xs=Xs, Xt=Xt)

# Sinkhorn / regularized Transport
ot_sinkhorn = ot.da.SinkhornTransport(reg_e=1e-1)
ot_sinkhorn.fit(Xs=Xs, Xt=Xt)

# implementation forthcoming: Tangential wwasserstein projection
## calculate transport plans with out implementation
## estimate optimal weights
## fit training values

# prediction between images
transp_Xs_emd = ot_emd.transform(Xs=X1)
transp_Xt_emd = ot_emd.inverse_transform(Xt=X2)

transp_Xs_sinkhorn = ot_sinkhorn.transform(Xs=X1)
transp_Xt_sinkhorn = ot_sinkhorn.inverse_transform(Xt=X2)

I1t = minmax(matrix_to_image(transp_Xs_emd, I1.shape))
I2t = minmax(matrix_to_image(transp_Xt_emd, I2.shape))

I1te = minmax(matrix_to_image(transp_Xs_sinkhorn, I1.shape))
I2te = minmax(matrix_to_image(transp_Xt_sinkhorn, I2.shape))


##############################################################################
# Plot new images
# ---------------

plt.figure(3, figsize=(8, 4))

plt.subplot(2, 3, 1)
plt.imshow(I1)
plt.axis('off')
plt.title('Image 1')

plt.subplot(2, 3, 2)
plt.imshow(I1t)
plt.axis('off')
plt.title('Image 1 Adapt')

plt.subplot(2, 3, 3)
plt.imshow(I1te)
plt.axis('off')
plt.title('Image 1 Adapt (reg)')

plt.subplot(2, 3, 4)
plt.imshow(I2)
plt.axis('off')
plt.title('Image 2')

plt.subplot(2, 3, 5)
plt.imshow(I2t)
plt.axis('off')
plt.title('Image 2 Adapt')

plt.subplot(2, 3, 6)
plt.imshow(I2te)
plt.axis('off')
plt.title('Image 2 Adapt (reg)')
plt.tight_layout()

plt.show()

##### export #######
plt.savefig('Dropbox (University of Michigan)/TWP/tangential-wasserstein-projection/output comparison.png')





##################################################
############ barycenter approach

""" failed attempt
N = 2
d = 2

I1 = plt.imread('Dropbox (University of Michigan)/TWP/tangential-wasserstein-projection/Test Images/apple.jpeg').astype(np.float64)[::4, ::4, 2]
I2 = plt.imread('Dropbox (University of Michigan)/TWP/tangential-wasserstein-projection/Test Images/pear.jpeg').astype(np.float64)[::4, ::4, 2]

sz = I2.shape[0]
XX, YY = np.meshgrid(np.arange(sz), np.arange(sz))

x1 = np.stack((XX[I1 == 0], YY[I1 == 0]), 1) * 1.0
x2 = np.stack((XX[I2 == 0] + 80, -YY[I2 == 0] + 32), 1) * 1.0
x3 = np.stack((XX[I2 == 0], -YY[I2 == 0] + 32), 1) * 1.0

measures_locations = [x1, x2]
measures_weights = [ot.unif(x1.shape[0]), ot.unif(x2.shape[0])]

plt.figure(1, (12, 4))
plt.scatter(x1[:, 0], x1[:, 1], alpha=0.5)
plt.scatter(x2[:, 0], x2[:, 1], alpha=0.5)
plt.title('Distributions')
"""


f1 = 1 - plt.imread('Dropbox (University of Michigan)/TWP/tangential-wasserstein-projection/Test Images/apple.jpeg')[:, :, 2]
f2 = 1 - plt.imread('Dropbox (University of Michigan)/TWP/tangential-wasserstein-projection/Test Images/pear.jpeg')[:, :, 2]
f3 = 1 - plt.imread()[:, :, 2] # two more images needed
f4 = 1 - plt.imread()[:, :, 2] # two more images needed (same comment as above)

f1 = f1 / np.sum(f1)
f2 = f2 / np.sum(f2)
f3 = f3 / np.sum(f3)
f4 = f4 / np.sum(f4)
A = np.array([f1, f2, f3, f4])

nb_images = 5

# those are the four corners coordinates that will be interpolated by bilinear
# interpolation
v1 = np.array((1, 0, 0, 0))
v2 = np.array((0, 1, 0, 0))
v3 = np.array((0, 0, 1, 0))
v4 = np.array((0, 0, 0, 1))


fig, axes = plt.subplots(nb_images, nb_images, figsize=(7, 7))
plt.suptitle('Wasserstein Barycenters in POT')
cm = 'Blues'

# regularization parameter
reg = 0.004
for i in range(nb_images):
    for j in range(nb_images):
        tx = float(i) / (nb_images - 1)
        ty = float(j) / (nb_images - 1)

        # weights are constructed by bilinear interpolation
        tmp1 = (1 - tx) * v1 + tx * v2
        tmp2 = (1 - tx) * v3 + tx * v4
        weights = (1 - ty) * tmp1 + ty * tmp2

        if i == 0 and j == 0:
            axes[i, j].imshow(f1, cmap=cm)
        elif i == 0 and j == (nb_images - 1):
            axes[i, j].imshow(f3, cmap=cm)
        elif i == (nb_images - 1) and j == 0:
            axes[i, j].imshow(f2, cmap=cm)
        elif i == (nb_images - 1) and j == (nb_images - 1):
            axes[i, j].imshow(f4, cmap=cm)
        else:
            # call to barycenter computation
            axes[i, j].imshow(
                ot.bregman.convolutional_barycenter2d(A, reg, weights),
                cmap=cm
            )
        axes[i, j].axis('off')
plt.tight_layout()
plt.show()

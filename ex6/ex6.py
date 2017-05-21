## Machine Learning Online Class
#  Exercise 6 | Support Vector Machines
#
#  Instructions
#  ------------
# 
#  This file contains code that helps you get started on the
#  exercise. You will need to complete the following functions:
#
#     gaussianKernel.m
#     dataset3Params.m
#     processEmail.m
#     emailFeatures.m
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.

import os
import time
import pandas as pd
import timeit
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit as sigmoid
from numpy import *
import scipy.io as sio
from sklearn import svm
from scipy.optimize import fmin_cg

## =============== Part 1: Loading and Visualizing Data ================
#  We start the exercise by first loading and visualizing the dataset. 
#  The following code will load the dataset into your environment and plot
#  the data.
#

print('Loading and Visualizing Data')

# Load from ex6data1: 
# You will have X, y in your environment
mat = sio.loadmat(r'ex6data1.py')
keys='X, y'.split(', ')
X, y = [mat[k] for k in keys]

def plotData(X, y, ax=None):
    pos = (y == 1).flatten()
    neg = (y == 0).flatten()
    if not ax:
        _, ax = plt.subplots(1)
    ax.scatter(X[pos, 0], X[pos, 1], c='r', marker='+', label='pos')
    ax.scatter(X[neg, 0], X[neg, 1], c='b', marker='o', label='neg')
    return ax

# Plot training data
plotData(X, y)

# ## ==================== Part 2: Training Linear SVM ====================
# #  The following code will train a linear SVM on the dataset and plot the
# #  decision boundary learned.

print('Training Linear SVM')
 
# # You should try to change the C value below and see how the decision
# # boundary varies (e.g., try C = 1000)
def svmTrainLin(X,y,C):
    clf = svm.LinearSVC(C=C)
    model = clf.fit(X, y)
    return model

def svmTrain(X, y, C, kernel=None, gamma='auto'):
    clf = svm.SVC(C=C, kernel=kernel, gamma=gamma)
    model = clf.fit(X, y)  
    return model

def visualizeBoundary(X, y, model, ax=None, title=None):
    #VISUALIZEBOUNDARYLINEAR plots a linear decision boundary learned by the
    #SVM
    #   VISUALIZEBOUNDARYLINEAR(X, y, model) plots a linear decision boundary 
    #   learned by the SVM and overlays the data on it
    h = .02  # step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    # Put the result into a color plot
    ax = plotData(X, y, ax=ax)
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.5)  # @UndefinedVariable
    if title:
        ax.set_title(title)
    
# vary C, e.g. 100
C=1
model = svmTrainLin(X, y.ravel(), C)
visualizeBoundary(X, y, model)


# ## =============== Part 3: Implementing Gaussian Kernel ===============
# #  You will now implement the Gaussian kernel to use
# #  with the SVM. You should complete the code in gaussianKernel.m

def gaussianKernelMultidim(X1, X2):
    out = np.zeros((X1.shape[0], X2.shape[0]))
    for nx1, x1 in enumerate(X1):
        for nx2, x2 in enumerate(X2):
            out[nx1, nx2] = gaussianKernel(x1, x2)
    return out
            

def gaussianKernel(x1, x2):
    #RBFKERNEL returns a radial basis function kernel between x1 and x2
    #   sim = gaussianKernel(x1, x2) returns a gaussian kernel between x1 and x2
    #   and returns the value in sim
    sigma=0.1
    
    sim = np.exp(-((x1 - x2)**2).sum() / (2 * sigma**2) )
    # ====================== YOUR CODE HERE ======================
    # Instructions: Fill in this function to return the similarity between x1
    #               and x2 computed using a Gaussian kernel with bandwidth
    #               sigma
    #
    #
    return np.array(sim)[None, None]


# #
print('Evaluating the Gaussian Kernel')
# 
x1 = np.array([1, 2, 1])
x2 = np.array([0, 4, -1])
sigma = 2;
sim = gaussianKernel(x1, x2, sigma);
# 
print(('Gaussian Kernel between x1 = [1; 2; 1], x2 = [0; 4; -1], sigma = {} :{}\n'
         '(for sigma = 2, this value should be about 0.324652)').format(sigma, sim))
# ## =============== Part 4: Visualizing Dataset 2 ================
# #  The following code will load the next dataset into your environment and 
# #  plot the data. 
# #
# 
print('Loading and Visualizing Data')
# load('ex6data2.mat');
mat = sio.loadmat(r'ex6data2.py')
keys='X, y'.split(', ')
X, y = [mat[k] for k in keys]
# # Plot training data
plotData(X, y)
# ## ========== Part 5: Training SVM with RBF Kernel (Dataset 2) ==========
# #  After you have implemented the kernel, we can now use it to train the 
# #  SVM classifier.
# # 
print('Training SVM with RBF Kernel (this may take 1 to 2 minutes)');
# # SVM Parameters
C = 1
# Todo: sklearn need a kernel with exactly 2 arguments. So sigma cannot be passed
sigma = 0.1
# this is awefully slow, due to non efficient implementation of gaussian kernel.
# Compare with sklearn implementation:
tic = timeit.default_timer()
model= svmTrain(X, y.flatten(), C, kernel = gaussianKernelMultidim)
visualizeBoundary(X, y, model)
toc = timeit.default_timer()
print('Custom kernel took {} sec'.format(toc-tic))

tic = timeit.default_timer()
gamma = 1/(2*sigma**2)
model= svmTrain(X, y.flatten(), C, kernel='rbf', gamma=gamma)
visualizeBoundary(X, y, model)
toc = timeit.default_timer()
print('sklearn rbf kernel took {} sec'.format(toc-tic))

# ## =============== Part 6: Visualizing Dataset 3 ================
print('Loading and Visualizing Data')
mat = sio.loadmat(r'ex6data3.py')
keys='X, y, Xval, yval'.split(', ')
X, y, Xval, yval = [mat[k] for k in keys]
# # Plot training data
plotData(X, y)

# ## ========== Part 7: Training SVM with RBF Kernel (Dataset 3) ==========
# 
# #  This is a different dataset that you can use to experiment with. Try
# #  different values of C and sigma here.

Cs = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
sigmas = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
fig, axes = plt.subplots(7, 7, figsize=(12,10), sharex=True, sharey=True)
for _ax, C in zip(axes, Cs):
    for ax, sigma in zip(_ax, sigmas):
        gamma = 1/(2*sigma**2)
        model= svmTrain(X, y.flatten(), C, kernel='rbf', gamma=gamma)
        visualizeBoundary(X, y, model, ax=ax, title='C: {:.2f}, sig: {:.2f}'.format(C,sigma))
plt.tight_layout()
plt.show()

# # Try different SVM Parameters here

def dataset3Params(X, y, Xval, yval):
    Cs = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    sigmas = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    scores = np.zeros((len(Cs), len(sigmas)))
    for nC, C in enumerate(Cs):
        for nsigma, sigma in enumerate(sigmas):
            gamma = 1/(2*sigma**2)
            model= svmTrain(X, y.flatten(), C, kernel='rbf', gamma=gamma)
            # calc the score:
            score = model.score(Xval, yval)
            scores[nC, nsigma] = score
#     fig, ax = plt.subplots(1)
#     ax.imshow(scores)
    indices = np.unravel_index(scores.argmax(), scores.shape)
    bestC = Cs[indices[0]]
    bestsigma = sigmas[indices[1]]
    return bestC, bestsigma

C, sigma = dataset3Params(X, y, Xval, yval)
# 
# # Train the SVM
gamma = 1/(2*sigma**2)
model= svmTrain(X, y.flatten(), C, kernel='rbf', gamma=gamma)
visualizeBoundary(X, y, model)
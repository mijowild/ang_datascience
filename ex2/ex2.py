import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit as sigmoid
from numpy import *
from scipy.optimize import fmin_bfgs

def main_ex2():
    df = pd.read_csv('ex2data1.txt', names='a,b,c'.split(','))
    _X = np.array(df[['a', 'b']]).T
    X = np.r_[ones((1, _X.shape[1])), _X]
    y = np.array(df.c).T.reshape((1,len(df)))
    n,m = _X.shape
    initial_theta = zeros((n+1, 1))
    df.plot.scatter('a','b',c='c', cmap='viridis')
    cost = costFunction(initial_theta, *[X, y])
    grad = costFunctionGradient(initial_theta, *[X, y])
    theta =  fmin_bfgs(costFunction, initial_theta, costFunctionGradient, (X, y), maxiter=400)

    ax = df.plot.scatter('a','b',c='c', cmap='viridis')
    dot(theta.T, X)
    x_dec_bound = np.array([X[1,:].min(), X[1,:].max()])
    dec_bound = -theta[0]/theta[2] - theta[1]/theta[2] * x_dec_bound
    ax.plot(x_dec_bound, dec_bound)
    y_train = predict(theta, X)
    perc_right = (y_train == y).sum()/y.shape[1]*100
    print ('percentage in training dataset classified correctly: '.format(perc_right))

    df = pd.read_csv('ex2data2.txt', names='a,b,c'.split(','))
    _X = np.array(df[['a', 'b']]).T
    X = np.r_[ones((1, _X.shape[1])), _X]
    X = mapFeature(X[1,:], X[2,:])
    y = np.array(df.c).T.reshape((1,len(df)))
    n,m = X.shape
    n = n -1
    initial_theta = zeros((n+1, 1))

    ax = df.plot.scatter('a', 'b', c='c', cmap = 'viridis')

    theta =  fmin_bfgs(costFunction, initial_theta, costFunctionGradient, (X, y), maxiter=400)

    ax = df.plot.scatter('a', 'b', c='c', cmap = 'viridis')
    plot_contour(ax, theta)
    ax = df.plot.scatter('a', 'b', c='c', cmap = 'viridis')

    thetas = [fmin_bfgs(costFunctionReg, initial_theta, costFunctionGradientReg, (X, y, lam))
              for lam in np.logspace(-7, 3, 11)]
    for theta, label in zip(thetas, [str(lam) for lam in np.logspace(-5, 1, 7)]):
        plot_contour(ax, theta, label=label)
    ax.legend()
    plt.show()

def h(theta, X):
    return sigmoid(dot(theta.T, X))

def costFunction(theta, X, y):
    m = y.shape[1]
    J = 1/m *np.sum(-y * log(h(theta, X)) - (1 - y)*log(1-h(theta, X)))    
    return J

def costFunctionGradient(theta, X, y):
    m = y.shape[1]
    grad = 1/m * dot((h(theta, X) - y), X.T)  
    return grad.flatten()

    def costFunctionReg(theta, X, y, lam):
        """ regularized cost function """
    m = y.shape[1]
    J = 1/m *np.sum((-y * log(h(theta, X)) - (1 - y)*log(1-h(theta, X)))) + 1.0*lam/2.0/m * np.sum(theta[1:]**2)   
    return J

def costFunctionGradientReg(theta, X, y, lam):
    """ regularized cost function gradients """

    theta = theta.flatten()
    m = y.shape[1]
    grad0 =  1/m * dot(((h(theta, X)) - y), X[0,:].reshape((1,m)).T)  
    _grad = 1/m * dot((h(theta, X) - y), X[1:,:].T) + 1.0*lam/m*theta[1:]
    return np.c_[grad0,_grad].flatten()

def predict(theta, X):
    return (h(theta, X) >= 0.5).astype('int')

def mapFeature(X1, X2):
    degree = 6
    try:
        out = ones(len(X1))
        for i in range(degree+1):
            for j in range(i+1):
                out = np.c_[out, (X1**(i-j))*(X2**j)]
        return out.T
    except TypeError:
        out = np.array(1).reshape(1, 1)
        for i in range(degree+1):
            for j in range(i+1):
                out = np.c_[out, (X1**(i-j))*(X2**j)]
        return out.T

def plot_contour(ax, theta, label=None):
    u = linspace(-1, 1.5, 50)
    v = linspace(-1, 1.5, 50)

    z = zeros((len(u), len(v)))
    # Evaluate z = theta*x over the grid
    for i in range(len(u)):
        for j in range(len(v)):
            z[i,j] = mapFeature(u[i], v[j]).T.dot(theta)
    z = z.T
    #contour(u, v, z, [0, 0], 'LineWidth', 2)
    ax.contour(u,v,z, 0, label=label)
    
if __name__ == '__main__':
    main_ex2()
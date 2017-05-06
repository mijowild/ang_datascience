import os
import os.path as osp
import numpy as np
import shutil
import matplotlib.pyplot as plt
import pandas as pd
from numpy import *
from matplotlib.pyplot import *
plt.ion()

def main():
    df = pd.read_csv('ex1data1.txt', header=None, names=['popu', 'prof'])
    
    ax = df.plot.scatter('popu', 'prof', marker='x', c='r', s=30)
    
    m = len(df)
    print(m)
        
    _X = np.array(df.popu)
    _X = _X[None, :]
    X = np.r_[ones((1, _X.shape[1])), _X]

    _y = np.array(df.prof)
    y = _y[None, :]

    theta = zeros((2, 1))
    num_iters = 1500
    alpha = 0.01
    
    theta, J_history = gradientDescent(X, y, theta, alpha, num_iters)
    plt_sct(df, theta = theta)
    
    predict1 = dot([1, 3.5] , theta);
    predict2 = dot([1, 7], theta);

def plt_sct(df, theta=None):
    cols=df.columns
    ax = df.plot.scatter(cols[0], cols[1])
    ax.set_xlabel('Population in 10k')
    ax.set_ylabel('Profit in 10k $')
    if theta is not None:
        t = np.arange(df[cols[0]].min(), df[cols[0]].max())
        y = theta[0] + theta[1] * t
        ax.plot(t, y, c='r')

def h(theta, X):
    return dot(theta.T, X)

def computeCost(X, y, theta):
    m = y.shape[1]
    temp = (h(theta, X) - y)
    cost = 1. / (2*m) * dot(temp, temp.T)
    return cost[0][0]

def gradientDescent(X, y, theta, alpha, num_iters):
    m = y.shape[1];
    J_history = zeros(num_iters)
    for iter in range(num_iters):
        delta = 1./m * (dot(X, (dot(theta.T, X) - y).T))
        theta = theta - alpha * delta
        J_history[iter] = computeCost(X, y, theta);
    return theta, J_history

if __name__ == '__main__':
    main()
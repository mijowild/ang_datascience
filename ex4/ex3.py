# coding: utf-8
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit as sigmoid
from numpy import *
import scipy.io as sio
from scipy.optimize import fmin_bfgs
from ex2 import costFunctionReg, costFunctionGradientReg


def main_ex3():
    num_labels = 10
    mat = sio.loadmat('ex3data1.mat')
    y = mat['y'].T
    X = mat['X'].T
    m = X.shape[1]
    # get rid of confusing 10 instead of 0...
    y[y==10] = 0
    visualize_subset_100(X, y)
    print('\nTesting lrCostFunction()')
    theta_t = np.array([-2, -1, 1, 2])[:, None]
    X_t = np.c_[ones((5,1)), arange(1,16).reshape((5,3), order='F')/10].T
    y_t = (array([1,0,1,0,1]) >= 0.5)[None, :]
    lambda_t = 3
    J = costFunctionReg(theta_t, X_t, y_t, lambda_t)
    grad = costFunctionGradientReg(theta_t, X_t, y_t, lambda_t)
    print('\nCost: %f'% J)
    print('Expected cost: 2.534819')
    print('Gradients:')
    print(' {} '.format(grad))
    print('Expected gradients:')
    print(' 0.146561 -0.548558 0.724722 1.398003')
    
    # ============ Part 2b: One-vs-All Training ============
    lambda_all = 0.1
    all_theta = oneVsAll(X, y, num_labels, lambda_all);
    # make an array out of all_theta:
    all_theta_arr = np.array([all_theta[k] for k in all_theta.keys()])

    # ================ Part 3: Predict for One-Vs-All ================
    pred = predictOneVsAll(all_theta_arr, X);
    print('Training Set Accuracy: {}%'.format((pred == y).astype(float).mean()*100))
    print('Expected Accurcy: 94.9%')

    ## some intuition:
    # visualize the wrongly classified:
    indices = np.where(pred != y)[1]
    visualize_subset_100(X[:,indices], y[:, indices], pred[indices])
    # histogram of wrongly classified:
    f, a = plt.subplots(1)
    a.hist(y[:, indices].flatten(), alpha = 0.5, label = 'ground truth')
    a.hist(pred[indices].flatten(), alpha = 0.5, label = 'prediction')
    a.legend(framealpha=0.5)

    # # Neural nets

    # Load the weights into variables Theta1 and Theta2
    mat_weights = sio.loadmat('ex3weights.mat')
    input_layer_size  = 400
    hidden_layer_size = 25
    Theta1, Theta2 = mat_weights['Theta1'], mat_weights['Theta2']
    pred = predict(Theta1, Theta2, X);
    print('Training Set Accuracy: {}%'.format((pred == y).astype(float).mean()*100))
    print('Expected Accurcy: 97.5%')
    plt.show()


def visualize_subset_100(X, y, pred=None):
    _m = X.shape[1]
    fig, axes = plt.subplots(10, 10, sharex=True, sharey=True)
    _indices = np.random.randint(0, _m, 100)
    X_plot = X[:, _indices]
    y_plot = y[:, _indices].flatten()
    if pred is not None:
        pred_plot = pred[_indices]
    for nax, ax in enumerate(axes.flatten()):
        ax.imshow(X_plot[:, nax].reshape((20,20)).T, cmap='viridis', interpolation='none')
        if pred is not None:
            ax.set_title('{}, {}'.format(y_plot[nax], pred_plot[nax]))
        else:
            ax.set_title(str(y_plot[nax]))
        ax.axis('off')

def oneVsAll(X, y, num_labels, lam):
    m = X.shape[1]
    n = X.shape[0]
    initial_theta = zeros((n+1, 1))
    _X = np.r_[ones((1, m)), X]
    all_theta = dict()
    for label in arange(num_labels):
        print ('training one vs all logistic regression classifier for {}'.format(label))
        _y = (y == label).astype(int)
        theta =  fmin_bfgs(costFunctionReg, initial_theta, costFunctionGradientReg, (_X, _y, lam), maxiter=400)
        all_theta[label] = theta
    return all_theta

def h(theta, X):
    return sigmoid(dot(theta.T, X))

def predictOneVsAll(all_theta, X):
    m = X.shape[1]
    _X = np.r_[ones((1, m)), X]
    probs = np.array([h(theta, _X) for theta in all_theta_arr])
    predictions = probs.argmax(axis=0)
    return predictions

def predict(Theta1, Theta2, X):
    """PREDICT Predict the label of an input given a trained neural network
       p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
       trained weights of a neural network (Theta1, Theta2)"""

    m = X.shape[1]
    num_labels = Theta2.shape[0]
    _X = np.r_[ones((1, m)), X]
    hidden_layer = sigmoid(dot(Theta1, _X))
    _hidden_layer = np.r_[ones((1, hidden_layer.shape[1])), hidden_layer]
    output_layer = sigmoid(dot(Theta2, _hidden_layer))
    # fix the index by adding 1 and setting 10 to 0
    pred = output_layer.argmax(axis=0) + 1
    pred[pred == 10] = 0
    return pred

if __name__ == '__main__':
    main_ex3()
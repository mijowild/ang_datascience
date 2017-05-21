import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit as sigmoid
from numpy import *
import scipy.io as sio
from scipy.optimize import fmin_cg



def linearRegCostFunction(theta, X, y, lam):
    m = y.shape[0]; # number of training examples
    h = dot(X, theta)[:,None]
    J = 1/(2*m) * sum((h-y)**2) + lam/(2*m) * sum(theta**2)
    return J

def linearRegCostFunctionGrad(theta, X, y, lam):
    m = y.shape[0]; # number of training examples
    h = dot(X, theta)[:, None]
    grad0 = 1/m * dot((h-y).T, X[:, 0][:,None])
    gradr = 1/m * dot((h-y).T, X[:, 1:]) + lam/m*theta[1:]
    return np.c_[grad0, gradr].flatten()

def trainLinearReg(X, y, lam):
    # initialize Theta
    initial_theta = zeros((X.shape[1], 1))
    # Minimize using fmincg
    theta = fmin_cg(linearRegCostFunction,
                    initial_theta,
                    fprime=linearRegCostFunctionGrad,
                    args=(X, y, lam),
                    maxiter=200)
    return theta
    


def learningCurve(X, y, Xval, yval, lam):

    # Number of training examples
    m = X.shape[0]
    
    
    # ====================== YOUR CODE HERE ======================
    # Instructions: Fill in this function to return training errors in 
    #               error_train and the cross validation errors in error_val. 
    #               i.e., error_train(i) and 
    #               error_val(i) should give you the errors
    #               obtained after training on i examples.
    #
    # Note: You should evaluate the training error on the first i training
    #       examples (i.e., X(1:i, :) and y(1:i)).
    #
    #       For the cross-validation error, you should instead evaluate on
    #       the _entire_ cross validation set (Xval and yval).
    #
    # Note: If you are using your cost function (linearRegCostFunction)
    #       to compute the training and cross validation error, you should 
    #       call the function with the lambda argument set to 0. 
    #       Do note that you will still need to use lambda when running
    #       the training to obtain the theta parameters.
    #
    # Hint: You can loop over the examples with the following:
    #
    error_train, error_val = [], []
    for i in range(m):
        theta = trainLinearReg(X[:i+1,:], y[:i+1,:], lam)
        error_train.append(linearRegCostFunction(theta, X[:i+1,:], y[:i+1,:], 0)) # set lam to 0
        # Compute train/cross validation errors using training examples 
        error_val.append(linearRegCostFunction(theta, Xval, yval, 0)) # set lam to 0    
    return np.array([error_train, error_val])


def polyFeatures(X, p):
    #POLYFEATURES Maps X (1D vector) into the p-th power
    #   [X_poly] = POLYFEATURES(X, p) takes a data matrix X (size m x 1) and
    #   maps each example into its polynomial features where
    #   X_poly(i, :) = [X(i) X(i).^2 X(i).^3 ...  X(i).^p];
    Xi = []
    for i in range(1,p+1):
        Xi.append(X.flatten()**i)
    return np.array(Xi).T


def featureNormalize(X):    
    mu = X.mean(axis=0)
    X_norm = X - mu
    sigma = std(X_norm, axis=0)
    X_norm = X_norm / sigma
    return X_norm, mu, sigma


def plotFit(min_x, max_x, mu, sigma, theta, p, ax):
    #PLOTFIT Plots a learned polynomial regression fit over an existing figure.
    #Also works with linear regression.
    #   PLOTFIT(min_x, max_x, mu, sigma, theta, p) plots the learned polynomial
    #   fit with power p and feature normalization (mu, sigma).
    
    # We plot a range slightly bigger than the min and max values to get
    # an idea of how the fit will vary outside the range of the data points
    x = arange(min_x - 15, max_x + 25, 0.05).T
    
    # Map the X values 
    X_poly = polyFeatures(x, p);
    X_poly = (X_poly - mu) / sigma
    
    # Add ones
    X_poly = np.c_[ones((X_poly.shape[0], 1)), X_poly]
    
    # Plot
    ax.plot(x, dot(X_poly, theta), '--', LineWidth=2)


def validationCurve(X, y, X_val, yval):
    #VALIDATIONCURVE Generate the train and validation errors needed to
    #plot a validation curve that we can use to select lambda
    
    # Selected values of lambda (you should not change this)
    lambda_vec = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
    
    # You need to return these variables correctly.
    # ====================== YOUR CODE HERE ======================
    # Instructions: Fill in this function to return training errors in 
    #               error_train and the validation errors in error_val. The 
    #               vector lambda_vec contains the different lambda parameters 
    #               to use for each calculation of the errors, i.e, 
    #               error_train(i), and error_val(i) should give 
    #               you the errors obtained after training with 
    #               lambda = lambda_vec(i)
    #
    # Note: You can loop over lambda_vec with the following:
    #
    error_train, error_val = [], []
    for i, lam in enumerate(lambda_vec):        
        theta = trainLinearReg(X, y, lam)
        et = linearRegCostFunction(theta, X, y, 0) # set lam to 0
        ev = linearRegCostFunction(theta, X_val, yval, 0) # set lam to 0    
        error_train.append(et)
        error_val.append(ev)
    return [lambda_vec, np.array(error_train), np.array(error_val)]
        

        


def main():
    ## Machine Learning Online Class
    #  Exercise 5 | Regularized Linear Regression and Bias-Variance
    #
    #  Instructions
    #  ------------
    # 
    #  This file contains code that helps you get started on the
    #  exercise. You will need to complete the following functions:
    #
    #     linearRegCostFunction.m
    #     learningCurve.m
    #     validationCurve.m
    #
    #  For this exercise, you will not need to change any code in this file,
    #  or any other files other than those mentioned above.
    #
    
    ## =========== Part 1: Loading and Visualizing Data =============
    #  We start the exercise by first loading and visualizing the dataset. 
    #  The following code will load the dataset into your environment and plot
    #  the data.
    #
    
    # Load Training Data
    print('Loading and Visualizing Data ...')
    
    # Load from ex5data1: 
    # You will have X, y, Xval, yval, Xtest, ytest in your environment
    
    mat = sio.loadmat(r'c:\workspace\ang_machine_learning\ex5\ex5data1.mat')
    keys='X, y, Xtest, ytest, Xval, yval'.split(', ')
    X, y, Xtest, ytest, Xval, yval = [mat[k] for k in keys]
    # m = Number of examples
    m = X.shape[0]
    mval = Xval.shape[0]
    
    # Plot training data
    fig, ax = plt.subplots(1)
    ax.plot(X, y, 'rx', MarkerSize=10, LineWidth=1.5)
    ax.set_xlabel('Change in water level (x)');
    ax.set_ylabel('Water flowing out of the dam (y)')

    ## =========== Part 2: Regularized Linear Regression Cost =============
    theta = np.array([1, 1])
    J = linearRegCostFunction(theta, np.c_[ones((m, 1)), X], y, 1)
    print('Cost at theta = [1 ; 1]: {} (this value should be about 303.993192)'.format(J))
    # ## =========== Part 3: Regularized Linear Regression Gradient =============
    grad = linearRegCostFunctionGrad(theta, np.c_[ones((m, 1)), X], y, 1)
    print(('Gradient at theta = [1 ; 1]:  [{}, {}] (this value should be about'
           '[-15.303016; 598.250744])\n').format(*grad))
    # ## =========== Part 4: Train Linear Regression =============
    # #  Train linear regression with lambda = 0
    lam = 0;
    theta = trainLinearReg(np.c_[ones((m, 1)), X], y, lam)
    # #  Plot fit over the data
    ax.plot(X, dot(np.c_[ones((m, 1)), X], theta), '-', LineWidth=2)

    # ## =========== Part 5: Learning Curve for Linear Regression =============
    #  Next, you should implement the learningCurve function. 
    lam = 0
    error_train, error_val = learningCurve(np.c_[ones((m, 1)), X],
                                            y,
                                            np.c_[ones((mval, 1)), Xval],
                                            yval,
                                            lam)
    fig, ax = plt.subplots(1)
    ax.plot(arange(m), error_train, label='train')
    ax.plot(arange(m), error_val, label='cross validation')
    ax.set_title('Learning curve for linear regression')
    ax.legend()
    ax.set_xlabel('Number of training examples')
    ax.set_ylabel('Error')
    print('# Training Examples\tTrain Error\tCross Validation Error');
    for i in range(m):
        print('  \t{}\t\t{}\t{}'.format(i, error_train[i], error_val[i]))

    # ## =========== Part 6: Feature Mapping for Polynomial Regression =============
    p = 8
    # # Map X onto Polynomial Features and Normalize
    X_poly = polyFeatures(X, p);
    X_poly, mu, sigma = featureNormalize(X_poly);  # Normalize
    X_poly = np.c_[ones((m, 1)), X_poly];                   # Add Ones
    # # Map X_poly_test and normalize (using mu and sigma)
    X_poly_test = polyFeatures(Xtest, p);
    X_poly_test = (X_poly_test - mu) / sigma
    X_poly_test = np.c_[ones((X_poly_test.shape[0], 1)), X_poly_test]
 
    # # Map X_poly_val and normalize (using mu and sigma)
    X_poly_val = polyFeatures(Xval, p);
    X_poly_val = (X_poly_val - mu) / sigma
    X_poly_val = np.c_[ones((X_poly_val.shape[0], 1)), X_poly_val]
    
    print('Normalized Training Example 1:');
    print('  {}  '.format(X_poly[0, :]))

    # ## =========== Part 7: Learning Curve for Polynomial Regression =============
    # #  Now, you will get to experiment with polynomial regression with multiple
    # #  values of lambda. The code below runs polynomial regression with 
    # #  lambda = 0. You should try running the code with different values of
    # #  lambda to see how the fit and learning curve change.
    fig, all_axes = plt.subplots(2, 5)
    for i, lam in enumerate([1,2,3,4,5]):
        axes = all_axes[:,i]
        
        theta = trainLinearReg(X_poly, y, lam);
        # # Plot training data and fit
        ax = axes[0]
        ax.plot(X, y, 'rx', MarkerSize=10, LineWidth=1.5)
        plotFit(min(X), max(X), mu, sigma, theta, p, ax)
        ax.set_xlabel('Change in water level (x)');
        ax.set_ylabel('Water flowing out of the dam (y)');
        ax.set_title ('Poly Reg Fit (lam = {})'.format(lam));
        ax.grid()
        ax.set_ylim([0, 50])
        
        ax = axes[1]
        error_train, error_val = learningCurve(X_poly, y, X_poly_val, yval, lam);
        ax.plot(arange(m), error_train, label='error_train')
        ax.plot(arange(m), error_val, label='error_val')
        ax.set_xlabel('Number of training examples')
        ax.set_ylabel('Error')
        ax.legend()
        ax.grid()
        ax.set_ylim([0, 20])

    # ## =========== Part 8: Validation for Selecting Lambda =============
    lambda_vec, error_train, error_val = validationCurve(X_poly, y, X_poly_val, yval)
    fig, ax = plt.subplots(1)
    ax.plot(lambda_vec, error_train, label='train');
    ax.plot(lambda_vec, error_val, label='val');
    ax.legend();
    ax.set_xlabel('lambda');
    ax.set_ylabel('Error');

if __name__ == '__main__':
    main()
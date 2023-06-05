# set of functions for fitting logistic regression models
import math
import numpy as np
from linear_regression import linear_regression_predict

def sigmoid(z):
    """
    Compute the sigmoid or logistic function
    Args:
        z: values of wx+b

    Returns:
        probability of the event
    """
    return 1 / (1 + np.exp(-z))


def logistic_function(x, w, b):
    """
    Compute the logistic function
    Args:
        x: value of the feature
        w: value of the coefficient of the feature
        b: value of the intercept

    Returns:
        value of the logistic function for a logistic regression model with a single predictor
    """
    z = linear_regression_predict(x, w, b)
    return sigmoid(z)


def loss_function(fx, y):
    """
    Compute the loss for a single sample
    Args:
        fx: value of the logistic function
        y: actual value of y

    Returns:
        the loss for a single sample
    """
    loss = (-y * log(fx)) - ((1-y) * (np.log(1-fx)))
    return loss


def compute_cost_logistic(X, y, w, b):
    """
    Compute the cost function for logistic regression
    Args:
        X: np.array of observations and features
        y: outcome variables (binary 1, 0s)
        w: np.array with values of the coefficients
        b: value of the intercept

    Returns:
        value of the cost function
    """
    # get number of samples
    m = X.shape[0]
    total_loss = 0
    for i in range(m):
        z = np.dot(w, x[i]) + b
        fx = sigmoid(z)
        loss = loss_function(fx, y[i])
        total_loss += loss

    # now compute cost
    cost = total_loss / m
    return cost


def compute_gradient_logistic(X, y, w, b):
    """
    Compute the gradient for a single value of w and b

    Args:
        X: np.array of observations and features
        y: outcome variables (binary 1, 0s)
        w: np.array with values of the coefficients
        b: value of the intercept

    Returns:
        dj_dw, dj_db: values of the derivatives for w and b
    """
    # get the number of samples and number of features
    m, n = X.shape

    # initialise the derivatives
    ## features
    dj_dw = np.zeros(n)
    ## intercept
    dj_db = 0
    # loop through samples
    for i in range(m):
        # compute the loss
        z = np.dot(w, X[i]) + b
        fx = sigmoid(z)

        # calculate the error for that sample - why not the loss function here?
        error_i = fx - y[i]

        # loop through features and compute the derivative terms
        for j in range(n):
            dj_dw[j] = dj_dw[j] + (error_i * X[i,j])
        dj_db = dj_db + error_i

    # now divide by number of samples
    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_dw, dj_db

def gradient_descent_logistic(X, y, starting_w, starting_b: float = 0, niter: int = 1000, alpha: float = 0.1):
    """
    Perform gradient descent for logistic regression
    Args:
        X: np.array of observations and features
        y: outcome variables (binary 1, 0s)
        starting_w: np.array of starting coefficients for the features
        starting_b: starting intercept value defaulting to 0
        niter: number of iterations to perform
        alpha: learning rate, defaults to 0.1

    Returns:
        pd.DataFrame of results from each iteration
        w: values of the coefficients
        b: final intercept
    """
    # initialise key variables
    w_tracker = [starting_w]
    b_tracker = [starting_b]
    cost_tracker = []
    m, n = X.shape
    w = starting_w
    b = starting_b

    # now loop through number of iterations
    for iter in range(niter):
        # compute the gradient with starting w and b
        dj_dw_i, dj_db_i = compute_gradient_logistic(X, y, w, b)

        # update w
        tmp_w = w - alpha * dj_dw_i
        # update b
        tmp_b = b - alpha * dj_db_i

        # now compute cost with these values of w and b
        cost_tracker[iter] = compute_cost_logistic(X, y, w=tmp_w, b=tmp_b)
        w = tmp_w
        b = tmp_b

        # append the values of w and b
        w_tracker += [w]
        b_tracker += [b]

        if iter % math.ceil(niter / 10) == 0:
            print(f"Iteration {iter:4}: Cost {cost_tracker[-1]:0.2e} ",
                  f"w: {w: 0.3e}, b:{b: 0.5e}")

    # create a data frame output
    results = pd.DataFrame({
        'iteration': range(niter),
        'w': w_tracker,
        'b': b_tracker,
        'cost': cost_tracker
    })
    results['learning_rate'] = alpha

    return results, w, b


#print(sigmoid(z=3))

#print(logistic_function(x=5, w=1.5, b=3))
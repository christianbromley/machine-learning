import numpy as np
import pandas as pd
import math
def cost_function(x, y, w, b):
    """

    Args:
        x: values of the predictor
        y: values of the outcome variables
        w: the value of w with which to compute the cost function
        b: the value of b with which to compute the cost function

    Returns:

    """
    # initiate the total cost
    total_cost = 0
    # calculate m - the number of samples
    m = x.shape[0]
    # iterate through the values of x and apply the function
    for i in range(m):
        # compute the predicted y value for these values of w and b
        yhat = (w * x[i]) + b
        # now calculate the cost
        cost = (yhat - y[i]) ** 2
        total_cost = total_cost + cost

    # now we have the sum of the squared errors divide by 2m
    total_cost = total_cost / (2 * m)
    return total_cost

def compute_gradient(x, y, w, b):
    """
    Compute the gradient i.e. the derivatives of w and b
    Args:
        x: values of the predictor
        y: values of the outcome variables
        w: the value of w with which to compute the gradient
        b: the value of b with which to compute the gradient

    Returns:

    """
    # calculate m - the number of samples
    m = x.shape[0]
    # intitate the derivatives that we are computing
    d_dw = 0
    d_db = 0
    # now loop through m
    for i in range(m):
        # compute the function for the values of x and y
        yhat = (w * x[i]) + b
        # calculate the derivative of w for this iteration
        d_dw_i = (yhat - y[i]) * x[i]
        # calculate the derivative of b for this iteration
        d_db_i = (yhat - y[i])
        # add to the total sum of the derivatives
        d_dw += d_dw_i
        d_db += d_db_i
    # divide by the number of samples
    d_dw = d_dw / m
    d_db = d_db / m

    return d_dw, d_db

def perform_gradient_descent(x, y, alpha, niter: int = 1000, starting_w: float = 0, starting_b: float = 0):
    """
    Perform gradient descent for a univariate linear regression model
    Args:
        x: predictor values
        y: outcome variables
        alpha: learning rate
        niter: number of iterations to run
        starting_w: value of the coefficient to start the gradient descent
        starting_b: value of the intercept at which to start the gradient descent

    Returns:

    """
    # initialise key variables
    w_tracker = []
    b_tracker = []
    cost_tracker = []
    w = starting_w
    b = starting_b
    # perform iterations
    for iter in range(niter):
        # get derivatives by computing the gradient
        d_dw, d_db = compute_gradient(x=x, y=y, w=w, b=b)
        # now compute the temporary value of w
        tmp_w = w - alpha * d_dw
        # now compute the temporary value of b
        tmp_b = b - alpha * d_db
        # update w and b
        w = tmp_w
        b = tmp_b
        # keep record of w and b
        w_tracker += [w]
        b_tracker += [b]
        # compute the cost function
        total_cost = cost_function(x=x, y=y, w=w, b=b)
        cost_tracker += [total_cost]

        if iter % math.ceil(niter / 10) == 0:
            print(f"Iteration {iter:4}: Cost {cost_tracker[-1]:0.2e} ",
                  f"d_dw: {d_dw: 0.3e}, d_db: {d_db: 0.3e}  ",
                  f"w: {w: 0.3e}, b:{b: 0.5e}")

    #print(len(range(niter)))
    #print(len(w_tracker))
    #print(len(b_tracker))
    #print(len(cost_tracker))
    # save the results in a pandas data frame
    results = pd.DataFrame({
        'iteration': range(niter),
        'w': w_tracker,
        'b': b_tracker,
        #'learning_rate': [alpha],
        'cost': cost_tracker
    })
    results['learning_rate'] = alpha

    return results, w, b

def linear_regression_predict(x_new, w, b):
    """

    Args:
        x_new: the new value of x you want to predict the outcome for
        w: value of the coefficient for the optimal model
        b: value of the intercept for the optimal model

    Returns:
        predicted value of y
    """
    return (w * x_new) + b
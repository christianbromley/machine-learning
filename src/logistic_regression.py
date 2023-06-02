# set of functions for fitting logistic regression models
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
    z = linear_regression_predict(x, w, b)
    return sigmoid(z)

print(sigmoid(z=3))

print(logistic_function(x=5, w=1.5, b=3))
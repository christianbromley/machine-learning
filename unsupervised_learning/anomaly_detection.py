import math
import numpy as np

def gaussian(X):
    """
    Calculates mean and variance of all features
    in the dataset

    Args:
        X (ndarray): (m, n) Data matrix

    Returns:
        mu (ndarray): (n,) Mean of all features
        var (ndarray): (n,) Variance of all features
    """
    # get shape of input data
    m, n = X.shape
    # get feature means
    mu = np.sum(X, axis=0) / m
    # get feature variances
    var = np.sum((X - mu) ** 2, axis=0) / m
    return mu, var

def gaussian_estimation(x, mu, var):
    a = ((x-mu)**2) / (2 * var)
    b =  np.exp(-a)
    Px = (1 / sqrt(2 * math.pi * var)) * b
    return Px
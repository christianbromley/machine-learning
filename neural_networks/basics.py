import numpy as np
import os
import sys

sys.path.insert(0, f'{"/".join(os.getcwd().split("/")[0:-1])}/src')

#print('/'.join(os.getcwd().split('/')[0:-1]))

from logistic_regression import sigmoid

def dense_layer_logistic(A_in, W, B):
    """
    Dense layer of a neural network

    Args:
        A_in: a np.array with data input matching length of units
        W: a np.array weight matrix (n, j) with n features per unit and j units
        B: a np.array vector of length j

    Returns:

    """
    units = W.shape[1]
    activation_out = np.zeros(units)
    for j in range(units):
        z = np.dot(W[:,j], A_in) + B[j]
        activation_out[j] = sigmoid(z)

    return activation_out

def three_layer_neural_network_logistic(x, W1, b1, W2, b2, W3, b3):
    """
    Construct three layer neural network for classification
    Args:
        x: input data np.array
        W1: weights for first layer
        b1: bias vector for first layer
        W2: weights for second layer
        b2: bias vector for second layer
        W3: weights for third layer
        b3: bias vector for third layer

    Returns:
        Probability
    """
    a1 = dense_layer_logistic(x, W=W1, B=b1)
    a2 = dense_layer_logistic(a1, W=W2, B=b2)
    a3 = dense_layer_logistic(a2, W=W3, B=b3)
    return a3

def predict_from_three_layer_network(X, W1, b1, W2, b2, W3, b3):
    """
    Perform inference with weights from a three layer neural network

    Args:
        X: data to predict
        W1: weights for first layer
        b1: bias vector for first layer
        W2: weights for second layer
        b2: bias vector for second layer
        W3: weights for third layer
        b3: bias vector for third layer

    Returns:
        vector of probabilities
    """
    # get the number of predictions we will need to make (rows of data)
    m = X.shape[0]
    # initiate vector of predictions
    preds = np.zeros(m)
    # loop through samples
    for i in range(m):
        preds[i] = three_layer_neural_network_logistic(X[i], W1, b1, W2, b2, W3, b3)
    return preds
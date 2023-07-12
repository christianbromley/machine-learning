import numpy as np

def relu(z):
    """
    ReLU activation function (rectified linear unit)

    Args:
        z: function estimate wx + b

    Returns:
        event probability
    """
    return np.max(0,z)
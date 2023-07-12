import numpy as np

def softmax_activation(z):
    """
    Compute activation for softmax unit
    Args:
        z: np.array of Z values for multi class classification of length of z

    Returns:
        activation array
    """
    activation_out = []
    for i, element in enumerate(z):
        print(element)
        # get numerator
        numerator = np.exp(element)
        # get denominator
        denominator = np.sum([np.exp(j) for j in z])
        # get activation
        activation_out = activation_out + [numerator / denominator]
    return activation_out


def softmax_loss(y, activations):
    """
    Compute the loss for a specific class from the activations across all classes
    Args:
        y: value (not index) of the class you want to know the loss for
        activations: np.array of activation

    Returns:
        value of the loss for the y class
    """
    return -np.log(activations[y-1])


vals = np.array([0.1,0.3,0.7,0.2])
activation = softmax_activation(z=vals)
loss = softmax_loss(y=3, activations=activation)
print(activation)
print(loss)
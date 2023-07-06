import numpy as np
import os
import sys

sys.path.insert(0, f'{"/".join(os.getcwd().split("/")[0:-1])}/src')

#print('/'.join(os.getcwd().split('/')[0:-1]))

from logistic_regression import sigmoid

def dense_layer_logistic(A_in, W, B):
    return sigmoid(np.matmul(A_in, W) + B)
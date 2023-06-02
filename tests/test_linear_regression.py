# script to test linear regression functions
import numpy as np
import pandas as pd
import os
import sys

sys.path.insert(0, '/'.join(os.getcwd().split('/')[0:-1]))

#print(pd.__version__)

import linear_regression as lr

# test the cost function

## create x and y values to train with
x_train = np.array([1,3,4,2,6,7,9,12])
y_train = np.array([2,5,3,6,6,8,10,11])

mse = lr.cost_function(x=x_train, y=y_train, w=1, b=0)

print(mse)

# now implement gradient descent
descent_results, w_final, b_final = lr.perform_gradient_descent(x=x_train,
                                                             y=y_train,
                                                             alpha=0.01,
                                                             niter = 100,
                                                             starting_w=0,
                                                             starting_b=0)

print(f'Final value of w: {str(w_final)}')
print(f'Final value of b: {str(b_final)}')

# predict on a new value of x
x_new = 30
predicted_y = lr.linear_regression_predict(x_new=x_new, w=w_final, b=b_final)

print(f'the predicted value of Y for x={str(x_new)} is {str(predicted_y)}')
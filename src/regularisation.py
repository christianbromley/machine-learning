import numpy as np

def regularised_cost_function(X, y, w, b, l):
    m, n = X.shape
    cost = 0
    for i in range(m):
        fx = np.dot(w, X[i]) + b
        loss = (fx - y[i]) ** 2
        cost += loss

    regularised_cost = (cost / (2 * m)) + ((l / (2 * m)) * sum(w ** 2))

    return regularised_cost
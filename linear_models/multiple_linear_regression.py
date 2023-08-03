import numpy as np

def compute_cost_multiple(x, y, w, b):
    # compute m and n
    m, n = x.shape
    # intialise cost
    total_cost = 0
    # loop through m
    for i in range(m):
        yhat = np.dot(w, x[i]) + b
        cost = (yhat - y[i]) ** 2
        total_cost += cost
    return total_cost / (2*m)


def compute_gradient_multiple(x, y, w, b):
    # compute m and n
    m, n = x.shape
    # get array of values of the derivative for each feature
    d_dw = np.zeros(n)
    # init value of the derivative for b
    d_db = 0
    # loop through samples
    for i in range(m):
        # loop through features
        for j in range(n):
            d_dw[j] += (((np.dot(w, x[i]) + b) - y[i]) * x[i,j])
            d_db += (np.dot(w, x[i]) + b) - y[i]
    tmp_d_dw = d_dw / m
    tmp_d_db = d_db / m
    return tmp_d_dw, tmp_d_db

def multiple_linear_regression(x, y, alpha, niter, starting_b: float = 0):
    # initialise elements
    m,n = x.shape
    w_tracker = []
    b_tracker = []
    cost_tracker = []
    # start w with an array of zeros of length n
    w = np.zeros(n)
    b = starting_b
    # loop through iterations
    for iter in niter:
        # compute gradient
        d_dw, d_db = compute_gradient_multiple(x, y, w, b)
        # update w vector
        tmp_w = w - alpha * d_dw
        tmp_b = b - alpha * d_db
        # compute cost
        cost = compute_cost_multiple(x, y, w=tmp_w, b=tmp_b)
        # save cost history
        cost_tracker += cost
        w_tracker += tmp_w
        b_tracker += tmp_b
        w = tmp_w
        b = tmp_b

        if iter % math.ceil(niter / 10) == 0:
            print(f"Iteration {iter:4}: Cost {cost_tracker[-1]:0.2e} ")
    return cost_tracker, w_tracker, b_tracker, w, b



import numpy as np
from logistic_regression import sigmoid, compute_cost_logistic

def logistic_regression_regularised_cost_function(X, y, w, b, l):
    # get nfeatures and nsamples
    m, n = X.shape
    # compute the total loss without the regularisation term
    # initialise the loss
    total_loss = 0
    # loop through the samples
    for i in range(m):
        # compute the logistic function with the supplied values of the coefficient and b for the sample
        z = np.dot(w, x[i]) + b
        fx = sigmoid(z)
        # compute the loss using the l
        loss = (-y[i] * log(fx)) - ((1-y[i]) * (np.log(1-fx)))
        total_loss += loss

    # now compute cost
    cost = total_loss / m

    # now loop through the features to compute the regularisation term
    reg_term = 0
    for j in range(n):
        reg = w[j] ** 2
        reg_term += reg
    reg_param = (l / (2 * m)) * reg_term

    full_cost = cost + reg_param

    return full_cost


def linear_regression_regularised_cost_function(X, y, w, b, l):
    # get nfeatures and nsamples
    m, n = X.shape
    # initialise the cost
    cost = 0
    # loop through the samples and compute the cost and then divide by 2m
    for i in range(m):
        fx = np.dot(w, X[i]) + b
        loss = (fx - y[i]) ** 2
        cost += loss

    total_cost = cost / (2 * m)

    # now loop through the features to compute the regularisation term
    reg_term = 0
    for j in range(n):
        reg = w[j] ** 2
        reg_term += reg
    reg_param = (l / (2 * m)) * reg_term

    full_cost = total_cost + reg_param

    return full_cost


def compute_gradient_linear_regularised(X, y, w, b, l):
    m, n = X.shape
    # init the derivative
    dj_dw = np.zeros(m)
    dj_db = 0
    for i in range(m):
        # compute the error
        error_i =(np.dot(w, X[i]) + b) - y[i]
        for j in range(n):
            dj_dw[j] = dj_dw[j] + error_i * X[i,j]
        dj_db = dj_db + error_i
    dj_dw = dj_dw / m
    dj_db = dj_db / m

    # update derivatives with regularisation term
    for j in range(n):
        dj_dw[j] = dj_dw[j] + ((l / m) * w[j])

    return dj_dw, dj_db

def compute_gradient_logistic_regularised(X, y, w, b, l):
    m, n = X.shape
    # init the derivative
    dj_dw = np.zeros(m)
    dj_db = 0
    for i in range(m):
        # compute the error
        error_i = sigmoid(np.dot(w, X[i]) + b) - y[i]
        for j in range(n):
            dj_dw[j] = dj_dw[j] + error_i * X[i, j]
        dj_db = dj_db + error_i
    dj_dw = dj_dw / m
    dj_db = dj_db / m

    # update derivatives with regularisation term
    for j in range(n):
        dj_dw[j] = dj_dw[j] + ((l / m) * w[j])

    return dj_dw, dj_db


def gradient_descent_regularised_linear(X, y, niter, l, alpha, starting_w: float = 0, starting_b: float = 0, linear: bool = True):
    m, n = X.shape
    w_tracker = [starting_w]
    b_tracker = [starting_b]
    cost_tracker = []

    w = starting_w
    b = starting_b

    # loop through iter
    gd_results = pd.DataFrame()
    for iter in range(niter):
        # compute the gradient
        if linear:
            dj_dw, dj_db = compute_gradient_linear_regularised(X, y, w, b, l)
        else:
            dj_dw, dj_db = compute_gradient_logistic_regularised(X, y, w, b, l)

        # now update w and b
        tmp_w = w - dj_dw * alpha
        tmp_b = b - dj_db * alpha

        # compute the cost function
        if linear:
            cost_tracker[iter] = linear_regression_regularised_cost_function(X, y, w=tmp_w, b=tmp_b, l)
        else:
            cost_tracker[iter] = logistic_regression_regularised_cost_function(X, y, w=tmp_w, b=tmp_b, l)

        w = tmp_w
        b = tmp_b

        w_tracker += [w]
        b_tracker += [b]

        if iter % math.ceil(niter / 10) == 0:
            print(f"Iteration {iter:4}: Cost {cost_tracker[-1]:0.2e} ",
                  f"w: {w: 0.3e}, b:{b: 0.5e}")

        # create a data frame output
        results = pd.DataFrame({
            'iteration': range(niter),
            'w': w_tracker,
            'b': b_tracker,
            'cost': cost_tracker
        })
        results['learning_rate'] = alpha

        gd_results = pd.concat([gd_results, results], axis=0)

        return gd_results, w, b
import numpy as np

def colab_learn_cost_func(X, W, b, Y, R, lambda_):
    """
    Returns the cost for the content-based filtering
    Args:
      X (ndarray (num_movies,num_features)): matrix of item features
      W (ndarray (num_users,num_features)) : matrix of user parameters
      b (ndarray (1, num_users)            : vector of user parameters
      Y (ndarray (num_movies,num_users)    : matrix of user ratings of movies
      R (ndarray (num_movies,num_users)    : matrix, where R(i, j) = 1 if the i-th movies was rated by the j-th user
      lambda_ (float): regularization parameter
    Returns:
      J (float) : Cost
    """
    nm, nu = Y.shape
    J = 0

    for j in range(nu):

        for i in range(nm):
            yhat = np.dot(W[j, :], X[i, :]) + b[0, j]
            J += (R[i, j] * (yhat - Y[i, j])) ** 2

    J = (J / 2)

    J += (lambda_ / 2) * (np.sum(np.square(W)) + np.sum(np.square(X)))

    return J


def squared_distance_between vectors(a, b):
    """
    Returns the squared distance between two vectors
    Args:
      a (ndarray (n,)): vector with n features
      b (ndarray (n,)): vector with n features
    Returns:
      d (float) : distance
    """
    d = np.sum((a - b) ** 2)
    return d
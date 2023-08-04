import numpy as np

# define original centroids

def init_centroids(X: K):
    """
        This function initializes K centroids that are to be
        used in K-Means on the dataset X

        Args:
            X (ndarray): Data points
            K (int):     number of centroids/clusters

        Returns:
            centroids (ndarray): Initialized centroids
        """

    # Randomly reorder the indices of examples
    randidx = np.random.permutation(X.shape[0])

    # Take the first K examples as centroids
    centroids = X[randidx[:K]]

    return centroids


## randomly init K cluster centroids - mu1 (location of the first centroid), mu2 (location of the second centroid), muk
## muk = vector of values of length Nfeatures

# compute distance of each point to centroid
## loop through each sample
## set Ci as the index of the cluster centroid closest to xi such that mink ||xi - muk ||**2

## muci = cluster centroid of cluster to which example xi has been assigned
def get_index_closest_centroids(X, centroids):
    """
    Computes the centroid memberships for every example

    Args:
        X (ndarray): (m, n) Input values
        centroids (ndarray): (K, n) centroids

    Returns:
        idx (array_like): (m,) closest centroids

    """

    # Set K
    K = centroids.shape[0]

    # You need to return the following variables correctly
    idx = np.zeros(X.shape[0], dtype=int)

    # loop through samples
    for i in range(X.shape[0]):
        # calculate distance to each centroid
        distance = []
        for j in range(K):
            norm_ij = np.sum((X[i, :] - centroids[j, :]) ** 2)
            distance.append(norm_ij)

        idx[i] = np.argmin(distance)

    return idx

# recompute centroids - move to the mean of all the points in the cluster
## loop through clusters
## muk = average of feature i of points assigned to cluster k
## will be a vector of length Nfeatures
def compute_centroids(X, idx, K):
    """
    Returns the new centroids by computing the means of the
    data points assigned to each centroid.

    Args:
        X (ndarray):   (m, n) Data points
        idx (ndarray): (m,) Array containing index of closest centroid for each
                       example in X. Concretely, idx[i] contains the index of
                       the centroid closest to example i
        K (int):       number of centroids

    Returns:
        centroids (ndarray): (K, n) New centroids computed
    """

    # Useful variables
    m, n = X.shape

    # You need to return the following variables correctly
    centroids = np.zeros((K, n))
    for centroid in range(K):
        # get indices of samples whose closest centroid is centroid
        c_i = [i for i, val in enumerate(idx) if val == centroid]
        centroids[centroid] = np.mean(X[c_i, :], axis=0)

    return centroids

# run algorithm again on a loop so that you keep moving the centroids

# if you reach no further change to the assignment the algorithm must have converged

# cost function is the average squared distance between the sample and the clsuter centroid to which that sample has been assigned
# also called distortion function

# run random initialiseation ~50-1000 times and examine cost function (at point of convergence) of each
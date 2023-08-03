import numpy as np
import sys
import os

print('/'.join(os.getcwd().split('/')[0:7]))

sys.path.insert(0, f"{'/'.join(os.getcwd().split('/')[0:7])}/src")

from sampling import sample_with_replacement

def entropy(vals):
    """
    Compute entropy to assess purity of a leaf node.
    Reduction in entropy = information gain

    Args:
        vals: np.array of 0 and 1s - we assess the purity of 1s

    Returns:
        value of entropy
    """
    p1 = proportion_most_prevalent_class(vals)
    #print(p1)
    if p1 == 1:
        entropy = 0
    else:
        entropy = (-p1 * np.log2(p1)) - ((1-p1)*np.log2(1-p1))
    print(entropy)
    return entropy


def proportion_most_prevalent_class(vals):
    # get the most prevalent class
    count_0 = 0
    count_1 = 0
    for item in vals:
        if item == 0:
            count_0 += 1
        elif item == 1:
            count_1 += 1

    # get most frequent
    most_frequent_class = 1 if count_1 >= count_0 else 0

    #print(most_frequent_class)

    # Calculate the proportion of the most frequent class
    total_count = count_0 + count_1
    proportion_most_frequent_class = count_1 / total_count if most_frequent_class == 1 else count_0 / total_count

    #print(proportion_most_frequent_class)

    return proportion_most_frequent_class


def information_gain(root_vals, left_vals, right_vals):
    # entry at root node - ((left branch P * entropy) + (right branch P * entropy))
    # get the most prevalent class
    #rn_p = proportion_most_prevalent_class(root_vals)
    #print('root entropy')
    rn_entropy = entropy(root_vals)
    # left branch
    lb_p = proportion_most_prevalent_class(left_vals)
    #print('left entropy')
    lb_entropy = entropy(left_vals)
    # righ branch
    rb_p = proportion_most_prevalent_class(right_vals)
    #print('right entropy')
    #print(right_vals)
    rb_entropy = entropy(right_vals)
    # info gain
    return rn_entropy - ((lb_p * lb_entropy) + (rb_p * rb_entropy))

X = np.array([[1,0,0],
             [1,1,0],
             [0,0,1],
             [1,0,1],
             [1,1,1],
             [1,1,0],
             [1,0,1],
             [0,1,0],
             [1,1,1],
             [1,1,0]]
             )

Y = np.array([1,1,0,1,0,1,1,0,0,1])

#print(Y.shape[0])
#print(len(Y))

def select_root_node(X, Y):
    # all examples at root node
    # compute info gain for first three features
    # loop through features
    nfeat = X.shape[1]
    ig_root_node = []
    for i in range(nfeat):
        #print(i)
        #print(X[:,i])
        # now determine which samples go to the left and which to the right
        # first get the indices based on the feature split
        lv_i = [ind for ind, sample in enumerate(X[:,i]) if sample == 1]
        # next subset Y based on these indices
        Y_lb = np.array([Y[j] for j in lv_i])
        # repeat for right side
        rv_i = [ind2 for ind2, sample2 in enumerate(X[:, i]) if sample2 == 0]
        Y_rb = np.array([Y[j] for j in rv_i])

        # get the baseline root values
        ig_root_node += [information_gain(root_vals=Y, left_vals=Y_lb, right_vals=Y_rb)]

    return ig_root_node

print(select_root_node(X, Y))

#print(proportion_most_prevalent_class(vals=np.array([0,0,0,0,0,1])))
#print(proportion_most_prevalent_class(vals=Y))

def variance(vals):
    squares = [(val - np.mean(vals)) ** 2 for val in vals]
    sum_of_squares = np.sum(squares)
    return sum_of_squares / (len(vals)-1)


def create_training_sets(indices, B):
    # loop through number of trees
    training_sets = []
    for b in range(B):
        training_set = sample_with_replacement(indices)
        training_sets += [training_set]
    return training_sets

print(create_training_sets(indices=[1,2,3,4,5,6,7,8,9], B=10))


def compute_entropy(y):
    """
    Computes the entropy for

    Args:
       y (ndarray): Numpy array indicating whether each example at a node is
           edible (`1`) or poisonous (`0`)

    Returns:
        entropy (float): Entropy at that node

    """
    # You need to return the following variables correctly
    entropy = 0.

    if len(y) != 0:
        p_1 = np.sum(y) / len(y)
        if (p_1 != 0) & (p_1 != 1):
            entropy = (-p_1 * np.log2(p_1)) - ((1 - p_1) * np.log2(1 - p_1))

    return entropy

def split_dataset(X, node_indices, feature):
    """
    Splits the data at the given node into
    left and right branches

    Args:
        X (ndarray):             Data matrix of shape(n_samples, n_features)
        node_indices (list):     List containing the active indices. I.e, the samples being considered at this step.
        feature (int):           Index of feature to split on

    Returns:
        left_indices (list):     Indices with feature value == 1
        right_indices (list):    Indices with feature value == 0
    """

    # You need to return the following variables correctly
    left_indices = []
    right_indices = []

    for i in node_indices:
        print(i)
        if X[i, feature] == 1:
            left_indices += [i]
        elif X[i, feature] == 0:
            right_indices += [i]

    return left_indices, right_indices
def compute_information_gain(X, y, node_indices, feature):
    """
    Compute the information of splitting the node on a given feature

    Args:
        X (ndarray):            Data matrix of shape(n_samples, n_features)
        y (array like):         list or ndarray with n_samples containing the target variable
        node_indices (ndarray): List containing the active indices. I.e, the samples being considered in this step.

    Returns:
        cost (float):        Cost computed

    """
    # Split dataset
    left_indices, right_indices = split_dataset(X, node_indices, feature)

    # Some useful variables
    X_node, y_node = X[node_indices], y[node_indices]
    X_left, y_left = X[left_indices], y[left_indices]
    X_right, y_right = X[right_indices], y[right_indices]

    # You need to return the following variables correctly
    information_gain = 0

    #  get entropy of root node
    entropy_root = compute_entropy(y_node)

    # get entropy at left branch
    entropy_left = compute_entropy(y_left)
    w_left = len(y_left) / len(y_node)

    # get entropy at left branch
    entropy_right = compute_entropy(y_right)
    w_right = len(y_right) / len(y_node)

    information_gain = entropy_root - ((w_left * entropy_left) + (w_right * entropy_right))

    return information_gain


def get_best_split(X, y, node_indices):
    """
    Returns the optimal feature and threshold value
    to split the node data

    Args:
        X (ndarray):            Data matrix of shape(n_samples, n_features)
        y (array like):         list or ndarray with n_samples containing the target variable
        node_indices (ndarray): List containing the active indices. I.e, the samples being considered in this step.

    Returns:
        best_feature (int):     The index of the best feature to split
    """

    # Some useful variables
    num_features = X.shape[1]

    # You need to return the following variables correctly
    best_feature = -1
    max_gain = 0

    for feature in range(num_features):
        gain = compute_information_gain(X, y, node_indices, feature)

        if gain > max_gain:
            max_gain = gain
            best_feature = feature
        else:
            next

    return best_feature
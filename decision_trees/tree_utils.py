import numpy as np


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
        #print(lv_i)
        # next subset Y based on these indices
        Y_lb = np.array([Y[j] for j in lv_i])
        #print(Y_lb)
        # repeat for right side
        rv_i = [ind2 for ind2, sample2 in enumerate(X[:, i]) if sample2 == 0]
        #print(rv_i)
        Y_rb = np.array([Y[j] for j in rv_i])
        #print(Y_rb)

        #print(Y)

        # get the baseline root values
        ig_root_node += [information_gain(root_vals=Y, left_vals=Y_lb, right_vals=Y_rb)]

    return ig_root_node

print(select_root_node(X, Y))

#print(proportion_most_prevalent_class(vals=np.array([0,0,0,0,0,1])))
#print(proportion_most_prevalent_class(vals=Y))
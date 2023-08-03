import random
def sample_with_replacement(indices):
    """
    Perform sampling with replacement
    Args:
        indices: list of indices of the training samples

    Returns:
        list of lists with sample sets
    """
    m = len(indices)
    sample_set = []
    for i in range(m):
        sample_set += [random.choice(indices)]
    return sample_set

#example = [1,2,3,4,5,6,7,8,9,10,11,12,13,14]
#print(sample_with_replacement(indices=example))
import numpy as np

def max_division(vals):
    """
    Perform max normalisation
    Args:
        vals: array of values to scale

    Returns:
        array of normalised values
    """
    return vals / np.max(vals)

def min_max_normalisation(vals):
    """
    Perform min max normalisation
    Args:
        vals: array of values to scale

    Returns:
        array of normalised values
    """
    return (vals - np.min(vals)) / (np.max(vals) - np.min(vals))
def mean_normalisation(vals):
    """
    Perform mean normalisation
    Args:
        vals: array of values to scale

    Returns:
        array of normalised values
    """
    return (vals - np.mean(vals)) / (np.max(vals) - np.min(vals))

def z_score_normalisation(vals):
    """
    Perform Z-score normalisation
    Args:
        vals: array of values to scale

    Returns:
        array of normalised values
    """
    return (vals - np.mean(vals)) / np.std(vals)

values = [6,7,4,6,8,2,4,6,12,3,3,5]

print(max_division(vals=values))
print(mean_normalisation(vals=values))
print(z_score_normalisation(vals=values))

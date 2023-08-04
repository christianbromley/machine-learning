import numpy as np

def confusion_matrix(actual, predicted):
    """
    Function to calculate a confusion matrix from actual and predicted values of a binary categorical
    Args:
        actual: np.array of 1 and 0 values: actual binary classification values
        predicted: np.array of 1 and 0 values: predictions from your model

    Returns:
        np.array with values of the confusion matrix
    """
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    for i, pred_val in enumerate(predicted):
        if (pred_val == 1) & (actual[i] == 1):
            tp += 1
        elif (pred_val == 1) & (actual[i] == 0):
            fp += 1
        elif (pred_val == 0) & (actual[i] == 1):
            fn += 1
        elif (pred_val == 0) & (actual[i] == 0):
            tn += 1
    return np.array([[tp,fp],[fn,tn]])


def precision(actual, predicted):
    """
    Compute precision
    Args:
        actual: np.array of 1 and 0 values: actual binary classification values
        predicted: np.array of 1 and 0 values: predictions from your model

    Returns:
        value of precision
    """
    cm = confusion_matrix(actual, predicted)
    return cm[0][0] / np.sum(cm[0])


def recall(actual, predicted):
    """
    Compute recall

    Args:
        actual: np.array of 1 and 0 values: actual binary classification values
        predicted: np.array of 1 and 0 values: predictions from your model

    Returns:
        value of recall
    """
    cm = confusion_matrix(actual, predicted)
    return cm[0][0] / (cm[0][0] + cm[1][0])


def f1_score(actual, predicted):
    """
    Compute F1 score to balance precision and recall

    Args:
        actual: np.array of 1 and 0 values: actual binary classification values
        predicted: np.array of 1 and 0 values: predictions from your model

    Returns:

    """
    prec = precision(actual, predicted)
    rec = recall(actual, predicted)
    return 2 * ((prec*rec) / (prec+rec))


def select_threshold(y_val, p_val):
    """
    Finds the best threshold to use for selecting outliers
    based on the results from a validation set (p_val)
    and the ground truth (y_val)

    Args:
        y_val (ndarray): Ground truth on validation set
        p_val (ndarray): Results on validation set

    Returns:
        epsilon (float): Threshold chosen
        F1 (float):      F1 score by choosing epsilon as threshold
    """

    best_epsilon = 0
    best_F1 = 0
    F1 = 0

    step_size = (max(p_val) - min(p_val)) / 1000

    for epsilon in np.arange(min(p_val), max(p_val), step_size):
        predicted = np.where(p_val < epsilon, 1, 0)
        F1 = f1_score(y_val, predicted)

        if F1 > best_F1:
            best_F1 = F1
            best_epsilon = epsilon

    return best_epsilon, best_F1

n1 = np.array([1,0,1,0,1,0,0,0,0,1,1,1,0])
n2 = np.array([1,0,0,1,1,0,0,1,1,1,0,0,1])

print(f"Precision: {precision(n1,n2)}")
print(f"Recall: {recall(n1,n2)}")
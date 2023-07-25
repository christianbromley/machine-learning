import numpy as np

def confusion_matrix(actual, predicted):
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
    cm = confusion_matrix(actual, predicted)
    return cm[0][0] / np.sum(cm[0])


def recall(actual, predicted):
    cm = confusion_matrix(actual, predicted)
    return cm[0][0] / (cm[0][0] + cm[1][0])

n1 = np.array([1,0,1,0,1,0,0,0,0,1,1,1,0])
n2 = np.array([1,0,0,1,1,0,0,1,1,1,0,0,1])

print(f"Precision: {precision(n1,n2)}")
print(f"Recall: {recall(n1,n2)}")
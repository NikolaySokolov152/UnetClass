import numpy as np


def Dice(y_true, y_pred):
    smooth = 0.0000001
    y_true_bool = np.asarray(y_true, bool)  # Not necessary, if you keep your data
    y_pred_bool = np.asarray(y_pred, bool)  # in a boolean array already!
    intersection = np.double(np.bitwise_and(y_true_bool, y_pred_bool).sum())
    union_and_intersection = y_true_bool.sum() + y_pred_bool.sum()
    return (2. * intersection + smooth) / (union_and_intersection + smooth)

def Jaccard(y_true, y_pred):
    smooth = 0.0000001
    y_true_bool = np.asarray(y_true, bool)  # Not necessary, if you keep your data
    y_pred_bool = np.asarray(y_pred, bool)  # in a boolean array already!

    intersection = np.double(np.bitwise_and(y_true_bool, y_pred_bool).sum())
    union = np.double(np.bitwise_or(y_true_bool, y_pred_bool).sum())
    return (intersection + smooth) / (union + smooth)
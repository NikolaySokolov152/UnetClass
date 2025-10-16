import numpy as np
import math
import sklearn.metrics

def Jaccard(y_true, y_pred):
    smooth = 0.0000001
    y_true_bool = np.asarray(y_true, bool)  # Not necessary, if you keep your data
    y_pred_bool = np.asarray(y_pred, bool)  # in a boolean array already!

    intersection = np.double(np.bitwise_and(y_true_bool, y_pred_bool).sum())
    union = np.double(np.bitwise_or(y_true_bool, y_pred_bool).sum())
    return (intersection + smooth) / (union + smooth)

def Dice(y_true, y_pred):
    smooth = 0.0000001
    y_true_bool = np.asarray(y_true, bool)  # Not necessary, if you keep your data
    y_pred_bool = np.asarray(y_pred, bool)  # in a boolean array already!
    intersection = np.double(np.bitwise_and(y_true_bool, y_pred_bool).sum())
    union_and_intersection = y_true_bool.sum() + y_pred_bool.sum()
    return (2. * intersection + smooth) / (union_and_intersection + smooth)

def SoftDice(y_true, y_pred):
    smooth = 0.0000001
    axes = (0, 1)  # W,H axes of each image  (1, 2) ? @##################################################################################### &?????
    intersection = np.sum(np.abs(y_pred * y_true), axis=axes)
    mask_sum = np.sum(np.abs(y_true), axis=axes) + np.sum(np.abs(y_pred), axis=axes)
    return (2 * intersection + smooth) / (mask_sum + smooth)

def SoftJaccard(y_true, y_pred):
    smooth = 0.0000001
    axes = (0, 1)  # W,H axes of each image   (1, 2) ? @##################################################################################### &?????
    intersection = np.sum(np.abs(y_pred * y_true), axis=axes)
    mask_sum = np.sum(np.abs(y_true), axis=axes) + np.sum(np.abs(y_pred), axis=axes)
    union = mask_sum - intersection
    return (intersection + smooth) / (union + smooth)


def RI(y_true, y_pred):
    try:
        y_true = np.asarray(y_true, bool).astype(np.int32)
        y_pred = np.asarray(y_pred, bool).astype(np.int32)
        TN, FP, FN, TP = sklearn.metrics.confusion_matrix(y_true, y_pred).ravel()
        n = len(y_true)
        a = 0.5 * (TP * (TP - 1) + FP * (FP - 1) + TN * (TN - 1) + FN * (FN - 1))
        b = 0.5 * ((TP + FN) ** 2 + (TN + FP) ** 2 - (TP ** 2 + TN ** 2 + FP ** 2 + FN ** 2))
        c = 0.5 * ((TP + FP) ** 2 + (TN + FN) ** 2 - (TP ** 2 + TN ** 2 + FP ** 2 + FN ** 2))
        d = n * (n - 1) / 2 - (a + b + c)

        RI = (a + b) / (a + b + c + d)
    except:
        print("RI EXEPTION")
        RI = 0

    return RI

def Accuracy(y_true, y_pred):
    try:
        y_true = np.asarray(y_true, bool).astype(np.int32)
        y_pred = np.asarray(y_pred, bool).astype(np.int32)
        TN, FP, FN, TP = sklearn.metrics.confusion_matrix(y_true, y_pred).ravel()
        accuracy = float(TN + TP) / (TN + TP + FN + FP)
    except:
        print("Accuracy EXEPTION")
        accuracy = 0
    return accuracy

def Precition(y_true, y_pred):
    try:
        y_true = np.asarray(y_true, bool).astype(np.int32)
        y_pred = np.asarray(y_pred, bool).astype(np.int32)
        TN, FP, FN, TP = sklearn.metrics.confusion_matrix(y_true, y_pred).ravel()
        precition = float(TP) / (TP + FP)
    except:
        print("Precition EXEPTION")
        precition = 0
    return precition

def Recall(y_true, y_pred):
    try:
        y_true = np.asarray(y_true, bool).astype(np.int32)
        y_pred = np.asarray(y_pred, bool).astype(np.int32)
        TN, FP, FN, TP = sklearn.metrics.confusion_matrix(y_true, y_pred).ravel()
        recall = float(TP) / (TP + FN)
    except:
        print("Recall EXEPTION")
        recall = 0
    return recall

def Fscore(y_true, y_pred):
    try:
        y_true = np.asarray(y_true, bool).astype(np.int32)
        y_pred = np.asarray(y_pred, bool).astype(np.int32)
        precition = Precition(y_true, y_pred)
        recall = Recall(y_true, y_pred)

        fscore = (2 * precition * recall) / (precition + recall)
    except:
        print("Fscore EXEPTION")
        fscore = 0
    return fscore

def CrowdsourcingMetrics(y_true, y_pred):
    y_true = np.asarray(y_true, bool).astype(np.int32).ravel()
    y_pred = np.asarray(y_pred, bool).astype(np.int32).ravel()
    n = len(y_true)
    num_class = 1
    pij_matrix = np.zeros((num_class + 1, num_class + 1), np.float64)

    for i in range(len(y_true)):
        pij_matrix[y_pred[i], y_true[i]] += 1
    pij_matrix = pij_matrix / n  # pij_matrix.sum()

    s_i_arr = np.zeros(pij_matrix.shape[0], np.float64)
    for i in range(0, pij_matrix.shape[0]):
        for j in range(pij_matrix.shape[1]):
            s_i_arr[i] += pij_matrix[i][j]

    t_j_arr = np.zeros(pij_matrix.shape[1], np.float64)
    for j in range(0, pij_matrix.shape[1]):
        for i in range(0, pij_matrix.shape[0]):
            t_j_arr[j] += pij_matrix[i][j]

    sqr_t_sum = (t_j_arr ** 2).sum()
    sqr_s_sum = (s_i_arr ** 2).sum()
    sqr_pij_sum = (pij_matrix ** 2).sum()

    Vrand_split = sqr_pij_sum / sqr_t_sum
    Vrand_merge = sqr_pij_sum / sqr_s_sum

    Rand_Fscore = 2.0 * sqr_pij_sum / (sqr_t_sum + sqr_s_sum)

    p_logp = 0
    for i in range(0, pij_matrix.shape[0]):
        for j in range(0, pij_matrix.shape[1]):
            if pij_matrix[i, j] != 0:
                p_logp += pij_matrix[i, j] * math.log(pij_matrix[i, j])
    s_logs = 0
    for s_i in s_i_arr[:]:
        if s_i != 0:
            s_logs -= s_i * math.log(s_i)
    t_logt = 0
    for t_j in t_j_arr[:]:
        if t_j != 0:
            t_logt -= t_j * math.log(t_j)

    I = p_logp + s_logs + t_logt

    Vinfo_split = I / (s_logs)
    Vinfo_merge = I / (t_logt)

    InformationTheoreticFscore = 2.0 * I / (s_logs + t_logt)

    return [Vrand_split, Vrand_merge, Rand_Fscore, Vinfo_split, Vinfo_merge, InformationTheoreticFscore]

METRIC_NAMES = {
    "Jaccard": ("Jaccard",),
    "Dice": ("Dice",),
    "SoftDice": ("SoftDice",),
    "SoftJaccard": ("SoftJaccard",),
    "RI": ("RI",),
    "Accuracy": ("Accuracy",),
    "Precition": ("Precition",),
    "Recall": ("Recall",),
    "Fscore": ("Fscore",),
    "CrowdsourcingMetrics": ("Vrand_split",
                             "Vrand_merge",
                             "Rand_Fscore",
                             "Vinfo_split",
                             "Vinfo_merge",
                             "InformationTheoreticFscore")
}

METRIC_FUN = {
    "Jaccard": Jaccard,
    "Dice": Dice,
    "SoftDice": SoftDice,
    "SoftJaccard": SoftJaccard,
    "RI": RI,
    "Accuracy": Accuracy,
    "Precition": Precition,
    "Recall": Recall,
    "Fscore": Fscore,
    "CrowdsourcingMetrics": CrowdsourcingMetrics
}

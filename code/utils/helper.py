import random
import datetime
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve, f1_score

import torch

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 提升训练速度，训练结果保持一致
    torch.backends.cudnn.benchmark = True

def search_f1_auc(y_true, y_score):
    """
    as the metrics when save model
    """
    y_true = np.array(y_true)
    y_score = np.array(y_score)

    P = y_true.sum()
    R = y_score.sum()
    TP = ((y_true + y_score) > 1).sum()

    pre = TP / P
    rec = TP / R

    return 2 * (pre * rec) / (pre + rec)

def search_auc(y_true, y_score):
    """
    fp: rarray, shape = [>2]
        Increasing false positive rates such that element i is the false positive rate
        of predictions with score >= thresholds[i].
    tpr: array, shape = [>2]
        Increasing true positive rates such that element i is the true positive rate
        of predictions with score >= thresholds[i].
    thresholds: array, shape = [n_thresholds]
        Decreasing thresholds on the decision function used to compute fpr and tpr.
        thresholds[0] represents no instances being predicted and is arbitrarily set to max(y_score) + 1.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score)

    return auc(fpr, tpr)

def search_f1(y_true, y_score):
    y_true = np.array(y_true)
    y_score = np.array(y_score)

    scores = []
    thresholds = [i / 100 for i in range(100)]
    for threshold in thresholds:
        y_pred = (y_score > threshold).astype(int)
        score = f1_score(y_true, y_pred)
        scores.append(score)

    threshold = thresholds[np.argmax(scores)]

    return threshold

def show_dataframe(df):
    # show all dataframe data
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    print(df)

def format_time(elapsed):
    # takes a time in seconds
    # round to the nearest second
    elapsed_round = int(round(elapsed))
    # format hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_round))

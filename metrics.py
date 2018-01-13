#! /usr/bin/env python
# -*- encoding: utf-8 -*-
import math
from math import log, exp
"""
Python implementation from B Yang - original C++ code
https://www.kaggle.com/c/SemiSupervisedFeatureLearning/forums/t/919/auc-implementation
"""
def auc(y_true, y_preds):
    """
        TestCases:
        >>> labels = [1,0,0,1,0,1,0]
        >>> preds = [0.7, 0.5, 0.2, 0.4, 0.61, 0.8, 0.42]
        >>> auc(labels, preds)
        0.75
    """
    pos_num = sum(y_true)
    if pos_num == len(y_true):
        return 1
    sorted_pairs = sorted(zip(y_true, y_preds), key=lambda x: x[1])

    truePos, tp0 = pos_num, pos_num
    accum, tn = 0, 0
    threshold = sorted_pairs[0][1]
    for label, pred in sorted_pairs:
        if pred != threshold:
            threshold = pred
            accum = accum + tn * (truePos + tp0)
            tp0 = truePos
            tn = 0
        tn = tn + (1 - label)
        truePos = truePos - label

    accum = accum + tn * (truePos + tp0)
    return accum / (2.0 * pos_num * (len(y_true) - pos_num))

def logloss(labels, preds):
    loss = 0.0
    for y, pred in zip(labels, preds):
        loss += -(y * math.log(pred) + (1-y) * math.log(1-pred))

    return loss / len(labels)

def logloss2(y, pred):
    p = max(min(pred, 1.0 - 1e-15), 1e-15)
    return -log(p) if y == 1 else -log(1.0 - p)

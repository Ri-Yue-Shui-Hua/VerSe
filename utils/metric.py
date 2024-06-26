# -*- coding : UTF-8 -*-
# @file   : metric.py
# @Time   : 2024-04-10 9:29
# @Author : wmz

import numpy as np


def Dice(pred: np.ndarray, target: np.ndarray, smooth: float = 1) -> float:
    pred = np.atleast_1d(pred.astype(bool))
    target = np.atleast_1d(target.astype(bool))

    intersction = np.count_nonzero(pred & target)

    dice_coef = (2.*intersction+smooth)/float(
        np.count_nonzero(pred)+np.count_nonzero(target)+smooth
    )
    return dice_coef

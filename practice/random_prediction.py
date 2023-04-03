#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import List

import numpy as np


def prediction_with_randomness_1(x: np.ndarray, threshold: float = 0.5) -> List[str]:
    labels = x > threshold

    cat_is_one = np.random.random() > 0.5

    if cat_is_one:
        preds = ['cat' if l == 1 else 'dog' for l in labels]
    else:
        preds = ['dog' if l == 1 else 'cat' for l in labels]
    return preds


def prediction_with_randomness_2(x: np.ndarray, threshold: float = 0.5) -> List[str]:
    labels = x > threshold

    batch_size = x.shape[0]
    cats_are_ones = np.random.random(size=batch_size) > 0.5

    preds = []
    for l, cat_is_one in zip(labels, cats_are_ones):
        if cat_is_one:
            preds.append('cat' if l == 1 else 'dog')
        else:
            preds.append('dog' if l == 1 else 'cat')
    return preds

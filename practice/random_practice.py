#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from random_prediction import prediction_with_randomness_1, prediction_with_randomness_2


if __name__ == "__main__":
    np.random.seed(42)
    x = np.array([np.random.random() for _ in range(10)])
    print('x =', x)

    # 1. Will the output here be the same as the previous?
    np.random.seed(42)
    y = np.random.random(size=10)
    # uncomment here
    # print('y =', y)

    # 2. Write the code so that per-batch prediction will be the same as per-sample.
    np.random.seed(42)
    preds_batch_1 = prediction_with_randomness_1(x)
    print('\npreds_batch_1  =', preds_batch_1)

    # modify the following section of code:
    preds_sample_1 = []
    for x_i in x:
        x_i_np = np.array([x_i])
        pred = prediction_with_randomness_1(x_i_np)[0]
        preds_sample_1.append(pred)
    # end
    print('preds_sample_1 =', preds_sample_1)

    # 3. Write the code so that per-batch prediction will be the same as per-sample.
    np.random.seed(42)
    preds_batch_2 = prediction_with_randomness_2(x)
    print('\npreds_batch_2  =', preds_batch_2)

    # modify the following section of code:
    preds_sample_2 = []
    for x_i in x:
        x_i_np = np.array([x_i])
        pred = prediction_with_randomness_2(x_i_np)[0]
        preds_sample_2.append(pred)
    # end
    print('preds_sample_2 =', preds_sample_2)

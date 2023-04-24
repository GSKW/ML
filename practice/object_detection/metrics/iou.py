#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch


def intersection_over_union(boxes_preds: torch.Tensor,
                            boxes_labels: torch.Tensor,
                            box_format: str = "midpoint") -> torch.Tensor:
    """
    Calculates intersection over union

    Parameters:
        boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
        boxes_labels (tensor): Correct Labels of Boxes (BATCH_SIZE, 4)
        box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)

    Returns:
        tensor: Intersection over union for all examples
    """

    # Slicing idx:idx+1 in order to keep tensor dimensionality
    # Doing ... in indexing if there would be additional dimensions
    # Like for Yolo algorithm which would have (N, S, S, 4) in shape

    # YOUR CODE HERE

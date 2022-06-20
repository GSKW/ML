#!/usr/bin/env python3
# -*- coding: utf-8 -*-


class Node:
    """
    Implements a node of a decision tree.
    If the node is an internal node of a tree, then
        `feature` is an index of the feature for the split
        `threshold` is the threshold used for spliting based on the feature value
        `left` is the node after splitting by X[:, feature] < threshold
        `right` is the node after splitting by X[:, feature] >= threshold
    if the node is a leaf node:
        `value` holds the prediction class for the node
    """
    def __init__(self,
                 feature: int = None,
                 threshold: float = None,
                 left: Node = None,
                 right: Node = None,
                 value: int = None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self):
        return self.value is not None

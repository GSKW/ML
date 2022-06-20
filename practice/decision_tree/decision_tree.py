#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from node import Node


class DecisionTree:

    split_methods = ('Entropy', 'Gini', 'Ilyusha')

    def __init__(self, max_depth: int = 100, split_method: str = 'Entropy') -> None:

        if split_method not in self.split_methods:
            raise ValueError(f'split_method should be one of {self.split_methods}')

        self.max_depth = max_depth
        self.split_method = split_method
        self.root = None

    def _entropy(self, y: np.ndarray) -> float:
        """
        Calculates entropy / gini / ilyusha measure of chaos
        """
        raise NotImplementedError

    def _information_gain(self, X: np.ndarray, y: np.ndarray, threshold: float) \
            -> float:
        """
        Calculates information gain for a split using threshold
        """
        raise NotImplementedError

    def _best_split(self, X: np.ndarray, y: np.ndarray, features: list)\
            -> Tuple[int, float]:
        """
        Creates the best split from (X, y), chooses the best feature from features.
        Returns tuple of (best feature, best threshold).
        """
        raise NotImplementedError

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> Node:
        """
        Builds the whole decision tree and returns the root node.
        """
        raise NotImplementedError

    def _traverse_tree(self, x: np.ndarray, node: Node) -> float:
        """
        Traverses the tree starting from `node` as a root based on `x`.
        `x` contains a single sample to get predictions for.
        Return class prediction for `x`.
        """
        raise NotImplementedError

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Trains decision tree classifier based on (X, y)
        """
        self.root = self._build_tree(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Gets predictions for X
        """
        predictions = [self._traverse_tree(x, self.root) for x in X]
        return np.array(predictions)

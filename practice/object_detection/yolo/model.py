#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


"""
Implementation of Yolo (v1) architecture
"""

class CNNBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, **kwargs):
        super(CNNBlock, self).__init__()
        # YOUR CODE HERE

    def forward(self, x):
        # YOUR CODE HERE


class Yolov1(nn.Module):
    def __init__(self,
                 in_channels: int = 3,
                 split_size: int = 7,
                 num_boxes: int = 2,
                 num_classes: int = 20):
        super(Yolov1, self).__init__()
        self.in_channels = in_channels
        self.split_size = split_size
        self.num_boxes = num_boxes
        self.num_clases = num_classes

        self.darknet = self._create_conv_layers(in_channels)
        self.fcs = self._create_fcs(split_size, num_boxes, num_classes)

    def forward(self, x):
        x = self.darknet(x)
        return self.fcs(torch.flatten(x, start_dim=1))

    def _create_conv_layers(self, in_channels: int):
        # YOUR CODE HERE

    def _create_fcs(self, split_size: int, num_boxes: int, num_classes: int):
        S, B, C = split_size, num_boxes, num_classes
        # YOUR CODE HERE

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


"""
Implementation of Yolo (v1) architecture
with slight modification with added BatchNorm.

Information about architecture config:
    Tuple is structured by (kernel_size, filters, stride, padding)
    "M" is simply maxpooling with stride 2x2 and kernel 2x2
    List is structured by tuples and lastly int with number of repeats
"""

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNNBlock, self).__init__()
        # YOUR CODE HERE

    def forward(self, x):
        # YOUR CODE HERE


class Yolov1(nn.Module):
    def __init__(self, in_channels=3, **kwargs):
        super(Yolov1, self).__init__()
        self.in_channels = in_channels
        self.darknet = self._create_conv_layers()
        self.fcs = self._create_fcs(**kwargs)

    def forward(self, x):
        x = self.darknet(x)
        return self.fcs(torch.flatten(x, start_dim=1))

    def _create_conv_layers(self):
        # YOUR CODE HERE

    def _create_fcs(self, split_size, num_boxes, num_classes):
        S, B, C = split_size, num_boxes, num_classes
        # YOUR CODE HERE

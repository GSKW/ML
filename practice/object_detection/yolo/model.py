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
        self.conv1 = nn.Conv2d(
            in_channels, 
            out_channels,
            kwargs['kernel_size'],
            kwargs['stride'],
            kwargs['padding'],
            bias = False,
            groups = kwargs['groups']
        )
        self.relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        return x


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
        pool = nn.MaxPool2d(2, 2)
        
        conv1 = CNNBlock(
            in_channels=3,
            out_channels=192,
            kernel_size=(7,7),
            stride=2,
            padding=3,
            groups=3
        )
        conv2 = CNNBlock(
            in_channels=192,
            out_channels=256,
            kernel_size=(3,3),
            stride=1,
            padding=1,
            groups=1
        )
        conv3_1 = CNNBlock(
            in_channels=256,
            out_channels=128,
            kernel_size=(1,1),
            stride=1,
            padding=0,
            groups=1
        )
        conv3_2 = CNNBlock(
            in_channels=128,
            out_channels=256,
            kernel_size=(3,3),
            stride=1,
            padding=1,
            groups=1
        )
        conv3_3 = CNNBlock(
            in_channels=256,
            out_channels=256,
            kernel_size=(1,1),
            stride=1,
            padding=0,
            groups=1
        )
        conv3_4 = CNNBlock(
            in_channels=256,
            out_channels=512,
            kernel_size=(3,3),
            stride=1,
            padding=1,
            groups=1
        )
        conv4_1 = CNNBlock(
            in_channels=512,
            out_channels=256,
            kernel_size=(1,1),
            stride=1,
            padding=0,
            groups=1
        )
        conv4_2 = CNNBlock(
            in_channels=256,
            out_channels=512,
            kernel_size=(3,3),
            stride=1,
            padding=1,
            groups=1
        )
        
        conv4_3 = CNNBlock(
            in_channels=512,
            out_channels=512,
            kernel_size=(1,1),
            stride=1,
            padding=0,
            groups=1
        )
        conv4_4 = CNNBlock(
            in_channels=512,
            out_channels=1024,
            kernel_size=(3,3),
            stride=1,
            padding=1,
            groups=1
        )
        conv5_1 = CNNBlock(
            in_channels=1024,
            out_channels=512,
            kernel_size=(1,1),
            stride=1,
            padding=0,
            groups=1
        )
        conv5_2 = CNNBlock(
            in_channels=512,
            out_channels=1024,
            kernel_size=(3,3),
            stride=1,
            padding=1,
            groups=1
        )
        conv5_3 = CNNBlock(
            in_channels=1024,
            out_channels=1024,
            kernel_size=(3,3),
            stride=1,
            padding=1,
            groups=1
        )
        conv5_4 = CNNBlock(
            in_channels=1024,
            out_channels=1024,
            kernel_size=(3,3),
            stride=2,
            padding=1,
            groups=1
        )
        conv6 = CNNBlock(
            in_channels=1024,
            out_channels=1024,
            kernel_size=(3,3),
            stride=1,
            padding=1,
            groups=1
        )
        
        conv4_1_2 = [
            
        ]
        
        block = nn.Sequential(
            conv1, pool,
            conv2, pool,
            conv3_1, conv3_2, conv3_3, conv3_4, pool,
            conv4_1, conv4_2, conv4_1, conv4_2, conv4_1,
            conv4_2, conv4_1, conv4_2, conv4_3, conv4_4, pool,
            conv5_1, conv5_2, conv5_1, conv5_2, conv5_3, conv5_4,
            conv6, conv6,
            
            
        )
        return block

    def _create_fcs(self, split_size: int, num_boxes: int, num_classes: int):
        S, B, C = split_size, num_boxes, num_classes
        
        block = nn.Sequential(
            fc1, fc2
        )
        
        return block
        # YOUR CODE HERE
        
if __name__ == '__main__':
    x = torch.ones(1, 3, 448, 448)
    yolo = Yolov1()
    yolo(x)
    

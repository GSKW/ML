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
        
        architecture = [
            [in_channels, 192, 7, 2, 3, in_channels], 'M',
            
            [192, 256, 3, 1, 1, 1], 'M',
            
            [256, 128, 1, 1, 0, 1],
            [128, 256, 3, 1, 1, 1],
            [256, 256, 1, 1, 0, 1],
            [256, 512, 3, 1, 1, 1], 'M',
            
            [512, 256, 1, 1, 0, 1],
            [256, 512, 3, 1, 1, 1],
            [512, 256, 1, 1, 0, 1],
            [256, 512, 3, 1, 1, 1],
            [512, 256, 1, 1, 0, 1],
            [256, 512, 3, 1, 1, 1],
            [512, 256, 1, 1, 0, 1],
            [256, 512, 3, 1, 1, 1],
            [512, 512, 1, 1, 0, 1],
            [512, 1024, 3, 1, 1, 1], 'M',
            
            [1024, 512, 1, 1, 0, 1],
            [512, 1024, 1, 1, 0, 1],
            [1024, 512, 1, 1, 0, 1],
            [512, 1024, 1, 1, 0, 1],
            [1024, 1024, 3, 1, 1, 1],
            [1024, 1024, 3, 2, 1, 1],
            
            [1024, 1024, 3, 1, 1, 1]
            
        ]
        
        layers = [CNNBlock(
            in_channels=x[0],
            out_channels=x[1],
            kernel_size=x[2],
            stride=x[3],
            padding=x[4],
            groups=x[5]
        ) if x != 'M' else nn.MaxPool2d(2, 2) for x in architecture]
        
        
        
        block = nn.Sequential(
            *layers
        )
        return block

    def _create_fcs(self, split_size: int, num_boxes: int, num_classes: int):
        S, B, C = split_size, num_boxes, num_classes
        
        block = nn.Sequential(
            nn.Linear(7*7*1024, 4096),
            nn.LeakyReLU(0.1),
            nn.Linear(4096, S*S*(C+5*B))
        )
        
        print(block)
        
        return block
        
if __name__ == '__main__':
    x = torch.ones(1, 3, 448, 448)
    yolo = Yolov1()
    yolo(x)
    

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from typing import Optional, Callable, Tuple

import torch
import pandas as pd
from PIL import Image
from torchvision.datasets import VOCDetection

"""
Creates a Pytorch dataset to load the Pascal VOC dataset
"""


class VOCDataset(torch.utils.data.Dataset):
    def __init__(self,
                 root: str,
                 split: str,
                 S: int = 7,
                 B: int = 2,
                 C: int = 20,
                 download: bool = True,
                 transform: Optional[Callable] = None) -> None:
        assert split in ('train', 'trainval', 'val')
        self.dataset = VOCDetection(root, image_set=split, download=download,
                                    year='2012')
        self.transform = transform
        self.S = S
        self.B = B
        self.C = C
        self.class_names = [
            'person',
            'bird', 'cat', 'cow', 'dog', 'horse', 'sheep',
            'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train',
            'bottle', 'chair', 'dining table', 'potted plant', 'sofa', 'tvmonitor'
        ]

    def __len__(self):
        return 100 # len(self.dataset)

    def _str2float(self, x: str) -> float:
        return float(x) if float(x) != int(float(x)) else int(x)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image, labels = self.dataset[index]

        image_width = self._str2float(labels['annotation']['size']['width'])
        image_height = self._str2float(labels['annotation']['size']['height'])

        boxes = []
        for obj in labels['annotation']['object']:
            class_label = self.class_names.index(obj['name'])
            box = obj['bndbox']
            x, y = self._str2float(box['xmin']), self._str2float(box['ymin'])
            width = self._str2float(box['xmax']) - x
            height = self._str2float(box['ymax']) - y
            x, y = x / image_width, y / image_height
            width, height = width / image_width, height / image_height
            boxes.append([class_label, x, y, width, height])
        boxes = torch.tensor(boxes)

        if self.transform:
            image, boxes = self.transform(image, boxes)

        # Convert To Cells
        label_matrix = torch.zeros((self.S, self.S, self.C + 5 * self.B))
        for box in boxes:
            class_label, x, y, width, height = box.tolist()
            class_label = int(class_label)

            # i,j represents the cell row and cell column
            i, j = int(self.S * y), int(self.S * x)
            x_cell, y_cell = self.S * x - j, self.S * y - i

            """
            Calculating the width and height of cell of bounding box,
            relative to the cell is done by the following, with
            width as the example:

            width_pixels = (width*self.image_width)
            cell_pixels = (self.image_width)

            Then to find the width relative to the cell is simply:
            width_pixels/cell_pixels, simplification leads to the
            formulas below.
            """
            width_cell, height_cell = (
                width * self.S,
                height * self.S,
            )

            # If no object already found for specific cell i,j
            # Note: This means we restrict to ONE object
            # per cell!
            if label_matrix[i, j, 20] == 0:
                # Set that there exists an object
                label_matrix[i, j, 20] = 1

                # Box coordinates
                box_coordinates = torch.tensor(
                    [x_cell, y_cell, width_cell, height_cell]
                )

                label_matrix[i, j, 21:25] = box_coordinates

                # Set one hot encoding for class_label
                label_matrix[i, j, class_label] = 1

        return image, label_matrix


if __name__ == "__main__":
    data = VOCDataset('data/voc', 'val', download=True)

    print(data[0][1])

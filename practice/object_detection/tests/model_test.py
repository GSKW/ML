#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest

import torch

from yolo.model import Yolov1


class TestYolov1Model(unittest.TestCase):
    def setUp(self):
        self.S = 7
        self.B = 2
        self.C = 20

        self.yolo_rgb_input = torch.empty((1, 3, 448, 448))
        self.fc_input = torch.empty((1, 1024, 7, 7))
        self.yolo_gray_input = torch.empty((1, 1, 448, 448))

        self.expected_darknet_shape = torch.Size((1, 1024, 7, 7))
        self.expected_output_shape = torch.Size((1, self.S * self.S * (self.C + self.B * 5)))

        self.new_S = 10
        self.expected_new_S_output_shape = \
            torch.Size((1, self.new_S * self.new_S * (self.C + self.B * 5)))

        self.new_B = 10
        self.expected_new_B_output_shape = \
            torch.Size((1, self.S * self.S * (self.C + self.new_B * 5)))

        self.new_C = 21
        self.expected_new_C_output_shape = \
            torch.Size((1, self.S * self.S * (self.new_C + self.B * 5)))

    def test_darknet_layers(self):
        model = Yolov1(in_channels=3, split_size=self.S, num_boxes=self.B,
                       num_classes=self.C)
        output = model.darknet(self.yolo_rgb_input)
        self.assertEqual(len(output.shape), len(self.expected_darknet_shape))
        self.assertEqual(output.shape, self.expected_darknet_shape)

    def test_fc_layers(self):
        model = Yolov1(in_channels=3, split_size=self.S, num_boxes=self.B,
                       num_classes=self.C)
        output = model.fcs(torch.flatten(self.fc_input, start_dim=1))
        self.assertEqual(len(output.shape), len(self.expected_output_shape))
        self.assertEqual(output.shape, self.expected_output_shape)

    def test_rgb_yolo(self):
        model = Yolov1(in_channels=3, split_size=self.S, num_boxes=self.B,
                       num_classes=self.C)
        output = model(self.yolo_rgb_input)
        self.assertEqual(len(output.shape), len(self.expected_output_shape))
        self.assertEqual(output.shape, self.expected_output_shape)

    def test_gray_yolo(self):
        model = Yolov1(in_channels=1, split_size=self.S, num_boxes=self.B,
                       num_classes=self.C)
        output = model(self.yolo_gray_input)
        self.assertEqual(len(output.shape), len(self.expected_output_shape))
        self.assertEqual(output.shape, self.expected_output_shape)

    def test_with_nondefault_split_size(self):
        model = Yolov1(in_channels=3, split_size=self.new_S, num_boxes=self.B,
                       num_classes=self.C)
        output = model(self.yolo_rgb_input)
        self.assertEqual(len(output.shape), len(self.expected_new_S_output_shape))
        self.assertEqual(output.shape, self.expected_new_S_output_shape)

    def test_with_nondefault_num_boxes(self):
        model = Yolov1(in_channels=3, split_size=self.S, num_boxes=self.new_B,
                       num_classes=self.C)
        output = model(self.yolo_rgb_input)
        self.assertEqual(len(output.shape), len(self.expected_new_B_output_shape))
        self.assertEqual(output.shape, self.expected_new_B_output_shape)

    def test_with_nondefault_num_classes(self):
        model = Yolov1(in_channels=3, split_size=self.S, num_boxes=self.B,
                       num_classes=self.new_C)
        output = model(self.yolo_rgb_input)
        self.assertEqual(len(output.shape), len(self.expected_new_C_output_shape))
        self.assertEqual(output.shape, self.expected_new_C_output_shape)


if __name__ == "__main__":
    print("Running Model Tests:")
    unittest.main()

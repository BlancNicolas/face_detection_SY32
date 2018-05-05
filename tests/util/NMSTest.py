#!/usr/bin/python3
# -*- coding: utf-8 -*-

from NMS import *
import numpy as np


# Test with only one detected face and two overlapping boxes.
def test_with_1_detected_face():
    box1 = np.array([13, 12, 17, 16])
    box2 = np.array([13, 14, 17, 18])
    box3 = np.array([14, 15, 18, 19])
    boxes = np.vstack((box1, box2, box3))
    scores = [0.78, 0.89, 0.77]
    picked = non_max_suppression(boxes, scores)
    assert (picked == boxes[1]).all()


# tests with 2 detected face, two boxes overlapping on the first one and the second box face overlapping
# slightly on the first.
def test_with_2_detected_face():
    box1 = np.array([13, 12, 17, 16])
    box2 = np.array([13, 14, 17, 18])
    box3 = np.array([14, 15, 18, 19])
    box4 = np.array([11, 16, 15, 20])
    boxes = np.vstack((box1, box2, box3, box4))
    scores = [0.65, 0.89, 0.77, 0.76]
    picked = non_max_suppression(boxes, scores)
    assert (picked == boxes[[1, 3]]).all()

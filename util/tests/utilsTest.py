#!/usr/bin/python3
# -*- coding: utf-8 -*-

from util.utils import *
from constants import img_train_path
from skimage import io

def test_compare_area_same_box():
	b1 = [2, 2, 5, 5]
	assert compareAreas(b1, b1) == 1.0


def test_compare_area_not_overlapping():
	b1 = [2, 2, 5, 5]
	b2 = [5, 7, 8, 10]
	assert compareAreas(b1, b2) == 0.0


def test_compare_area_overlapping():
	b1 = [2, 2, 5, 5]
	b2 = [4, 4, 7, 7]
	assert compareAreas(b1, b2) == (1.0 / 7.0)


def test_sliding_window():
	image = io.imread(img_train_path.replace('*', '0001'))
	boxes, windows = slidingWindow(image, 16, (32,32))
	assert boxes.shape[0] == 625



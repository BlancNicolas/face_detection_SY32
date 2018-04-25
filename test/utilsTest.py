#!/usr/bin/python3
# -*- coding: utf-8 -*-

from util.utils import *
import numpy as np


def test_compare_area_same_box():
	b1 = [2, 2, 5, 5]
	assert compare_area(b1, b1) == 1.0


def test_compare_area_not_overlapping():
	b1 = [2, 2, 5, 5]
	b2 = [5, 7, 8, 10]
	assert compare_area(b1, b2) == 0.0


def test_compare_area_overlapping():
	b1 = [2, 2, 5, 5]
	b2 = [4, 4, 7, 7]
	assert compare_area(b1, b2) == (1.0 / 7.0)



#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np


def compare_area(b1, b2):
	# find the overlapping area of the two boxes
	xx1 = np.maximum(b1[0], b2[0])
	yy1 = np.maximum(b1[1], b2[1])
	xx2 = np.minimum(b1[2], b2[2])
	yy2 = np.minimum(b1[3], b2[3])

	# Compute the area of the overlapping region
	common_w = np.maximum(0, xx2 - xx1 + 1)
	common_h = np.maximum(0, yy2 - yy1 + 1)
	common_area = common_w * common_h

	# Compute the area of the union of boxes regions
	b1_area = (b1[2] - b1[0] + 1) * (b1[3] - b1[1] + 1)
	b2_area = (b2[2] - b2[0] + 1) * (b2[3] - b2[1] + 1)
	union_area = b1_area + b2_area - common_area

	return common_area / union_area

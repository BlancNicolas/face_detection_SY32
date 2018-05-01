#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


#-------------------------------------------
# ---------- displayRectOnImg ------------------
# INPUT :
    # image : lthe image on which we want to put rectangle
    # rect_coord : coordonates of the rectangle

# OUTPUT : numpy array of yes and no labels
#-------------------------------------------
def displayRectOnImg(image, rect_coord):
	height = rect_coord[3] - rect_coord[1]
	width = rect_coord[2] - rect_coord[0]
	# Create figure and axes
	fig,ax = plt.subplots(1)
	# Display the image
	ax.imshow(image)
	# Create a Rectangle patch
	rect = patches.Rectangle((rect_coord[0], rect_coord[1]),
	                         width, height,
	                         linewidth=1, edgecolor='r', facecolor='none')
	# Add the patch to the Axes
	ax.add_patch(rect)
	plt.show()


#-------------------------------------------
# ---------- createLabels ------------------
# INPUT :
    # len_yes : length of yes labels we want
    # len_no : length of no labels we want

# OUTPUT : numpy array of yes and no labels
#-------------------------------------------
def createLabels(len_yes, len_no):
    return np.concatenate((np.ones(len_yes), np.zeros(len_no)), axis = 0)


def compareAreas(b1, b2):
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


#-----------------------------------------------
# ---------- slidingWindow ------------------
# INPUT :
#   - image : image we are sliding into
#   - step_size : shift of successive window slides
#   - window_size : size of sliding window
# OUTPUT :
#   - windows : numpy arrays containing every sliding window of the image
#   - boxes : numpy array of boxes corresponding to the successive windows
#-----------------------------------------------
def slidingWindow(image, step_size, window_size):
	boxes = np.empty((0, 4))
	windows = np.empty((0, window_size[1], window_size[0]))
	x_limit = image.shape[1] - window_size[0] + 1
	y_limit = image.shape[0] - window_size[1] + 1
	# slide a window across the image
	for y in range(0, y_limit, step_size):
		for x in range(0, x_limit, step_size):
			boxes = np.concatenate((boxes, [(x, y, x + window_size[0], y + window_size[1])]))
			windows = np.concatenate((windows, [image[y:(y + window_size[1]), x:(x + window_size[0])]]))
	return windows, boxes


def checkBoxShape(boxes_array, img_shape):
	# Retrieve indexes of boxes which coordinates are out of the img
	to_shift_right = boxes_array[:, 0] < 0
	to_shift_left = boxes_array[:, 0] > img_shape[1]
	to_shift_bottom = boxes_array[:, 1] < 0
	to_shift_top = boxes_array[:, 3] > img_shape[0]
	
	if to_shift_right.any():
		boxes_array[to_shift_right, 2] += boxes_array[to_shift_right, 0]
		boxes_array[to_shift_right, 0] = 0
		
	if to_shift_left.any():
		excess = boxes_array[to_shift_left, 2] - img_shape[1]
		boxes_array[to_shift_left, 0] -= excess
		boxes_array[to_shift_left, 2] = img_shape[1]
		
	if to_shift_bottom.any():
		boxes_array[to_shift_bottom, 3] += boxes_array[to_shift_bottom, 1]
		boxes_array[to_shift_bottom, 1] = 0
		
	if to_shift_top.any():
		excess = boxes_array[to_shift_top, 3] - img_shape[0]
		boxes_array[to_shift_top, 1] -= excess
		boxes_array[to_shift_top, 3] = img_shape[0]

	return boxes_array


#-----------------------------------------------
# ---------- rescaleWindow ---------------------
# Goal : after NMS we have a window which is possibly defined in a different scale from original scale.
# INPUT :
#   - boxes_array : numpy array containing the boxes to rescale
#   - original_shape : shape of original image
#   - current_shape : shape of current image which is resized
#
# OUTPUT : Linear SVC classifier
#-----------------------------------------------
def rescaleBoxes(boxes_array, original_shape, current_shape):
	# Comoute the width and height ratios between the original shape and the shape of the resized image
	width_ratio = original_shape[1] / current_shape[1]
	height_ratio = original_shape[0] / current_shape[0]
	# Compute width and height for each box
	current_widths = boxes_array[:, 2] - boxes_array[:, 0] + 1
	current_heights = boxes_array[:, 3] - boxes_array[:, 1] + 1
	# compute new width and height
	new_widths = current_widths * width_ratio
	new_heights = current_heights * height_ratio
	# Compute the offsets
	width_offsets = (0.5 * (new_widths - current_widths)).astype('int')
	height_offsets = (0.5 * (new_heights - current_heights)).astype('int')
	# Resize the boxes
	boxes_array[:, 0] = boxes_array[:, 0] - width_offsets
	boxes_array[:, 1] = boxes_array[:, 1] - height_offsets
	boxes_array[:, 2] = boxes_array[:, 2] + width_offsets
	boxes_array[:, 3] = boxes_array[:, 3] + height_offsets

	boxes_array = checkBoxShape(boxes_array, original_shape)

	return boxes_array

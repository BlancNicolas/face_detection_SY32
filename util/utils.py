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
    # Create figure and axes
    fig,ax = plt.subplots(1)
    # Display the image
    ax.imshow(image)
    # Create a Rectangle patch
    rect = patches.Rectangle((rect_coord[1], rect_coord[2]), rect_coord[3],rect_coord[4],linewidth=1,edgecolor='r',facecolor='none')
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


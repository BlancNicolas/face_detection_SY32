#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np


# Fonction for doing non-maxima suppression of overlapping boxes.
# Multiple boxes can be returned for the detection of a single face,
# so this fonction will be used to eliminate the excess of boxes found.
# It will keep only the box with best score and remove every other overlapping box.
# Input :
#   - boxes = a list of boxes corresponding to face detected by the classifier in one image
#   - scores = a list of scores associated with each box
# Output : the list of boxes to keep
# Source : Adrian Rosebrock, (Faster) Non Maximum Suppression in Python.
def nonMaxSuppression(boxes, scores, overlap_thres=0.5):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return None
    elif len(boxes) == 1:
        return 0

    # if the boxes contain integers, convert them to floats for divisions.
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # retrieve the coordinates of the bounding boxes
    # x1 = x coordinate of top-left corner
    # y1 = y coordinate of top-left corner
    # x2 = x coordinate of bottom-right corner
    # y2 = y coordinate of bottom-right corner

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of every bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)

    # sort the bounding boxes by score
    idxs = np.argsort(scores)

    # keep looping while some indexes still remain in the indexes list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the overlapping area of each box paired with the current box i
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of overlapping area for each pair
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap for each pair
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        # a sufficient overlapping with the current box
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlap_thres)[0])))

    # return picked indexes
    return pick

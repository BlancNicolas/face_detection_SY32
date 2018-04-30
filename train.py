#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 19:12:33 2018

@author: Nico
"""
import glob
from sklearn.utils import shuffle
from sklearn import svm
from skimage.feature import hog
from skimage.color import rgb2gray
from skimage.transform import resize, pyramid_gaussian
from skimage import io
from constants import *
from util.utils import *
from util.NMS import nonMaxSuppression

#Classifier arguments
c_param = 1.0

img_train_dir_content = sorted(glob.glob(img_train_path))
label = np.loadtxt(label_path)


#-------------------------------------------------
# ---------- classifierTraining ------------------
# INPUT :
    # neg_path : path where negative samples are stored
    # pos_path : path where positive samples are stored

# OUTPUT : Linear SVC classifier
#-------------------------------------------------
def classifierTraining(pos_path, neg_path):
    print(" -- Training a classifier SVC Linear --")
    # Initializing classifier
    clf = svm.LinearSVC(C = c_param)
    pos_dir_content = sorted(glob.glob(pos_path))
    neg_dir_content = sorted(glob.glob(neg_path))
    train_dir_content = np.concatenate((pos_dir_content, neg_dir_content), axis=0)
    labels = createLabels(len(pos_dir_content), len(neg_dir_content))
    hog_train = []
    for idx, path in enumerate(train_dir_content):
        img = io.imread(path)
        img = rgb2gray(img)
        img = resize(img, (32, 32))
        hog_train.append(hog(img))
    hog_train, labels = shuffle(hog_train, labels)
    return clf.fit(hog_train, labels)


#-----------------------------------------------
# ---------- rescaleWindow ---------------------
# Goal : after NMS we have a window which is possibly defined in a different scale from original scale.
# INPUT :
    # x : x coord of window we want to rescale
    # y : y coord of window we want to rescale
    # window_height : height of window we want to rescale
    # window_width : width of window we want to rescale
    # original_shape : shape of original image
    # current_img_shape : shape of current image which is resized


# OUTPUT : Linear SVC classifier
#-----------------------------------------------
def rescaleWindow(x, y, window_height, window_width, original_shape, current_img_shape):
    ratio = current_img_shape / original_shape
    rescaled_x = int( x / ratio )
    rescaled_y = int( y / ratio )
    rescaled_height = int ( window_height / ratio )
    rescaled_width = int ( window_width / ratio )
    return [rescaled_x, rescaled_y, rescaled_x + rescaled_width, rescaled_y + rescaled_height]


def rescaleBoxes(boxes_array, original_shape, current_shape):
    # Comoute the width and height ratios between the original shape and the shape of the resized image
    width_ratio = original_shape[1] / current_shape[1]
    height_ratio = original_shape[0] / current_shape[0]
    # Compute width and height for each box
    current_widths = boxes_array[:,2] - boxes_array[:,0] + 1
    current_heights = boxes_array[:,3] - boxes_array[:,1] + 1
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
    return boxes_array



#-----------------------------------------------
# ---------- learningFromData ------------------
# INPUT :
    # path_raw_data : path where raw train data are stored

# OUTPUT : Linear SVC classifier
#-----------------------------------------------
def detectFaces(image, classifier):
    candidate_boxes = np.empty((0,4))
    candidate_scores = np.array([])
    candidate_windows = np.empty((0, WINDOW_SIZE[0], WINDOW_SIZE[1]))
    validated_boxes = []

    # Pyramid on current image
    for (i, resized) in enumerate(pyramid_gaussian(image, downscale = 1.5)):
        if resized.shape[0] < WINDOW_SIZE[0] or resized.shape[1] < WINDOW_SIZE[1]:
            break

        # Create list of successive sliding windows with corresponding boxes.
        windows, boxes = slidingWindow(resized, step_size = 16, window_size = WINDOW_SIZE)

        # Compute hog for each sliding window
        features = np.array([hog(windows[0])])
        for w in windows[1:]:
            features = np.concatenate((features, [hog(w)]), axis = 0)

        # Compute scores based on given classifiers
        scores = classifier.decision_function(features)

        # Keep only boxes ith a detection probability above 50%
        mask = np.zeros(features.shape[0], dtype = bool)
        mask[scores > 0.5] = True
        boxes = boxes[mask]
        scores = scores[mask]
        windows = windows[mask]

        # Rescale boxes if image is resized
        if i > 1:
            rescaled_boxes = rescaleBoxes(boxes, image.shape, resized.shape)
            candidate_boxes = np.concatenate((candidate_boxes, rescaled_boxes))
        else:
            candidate_boxes = np.concatenate((candidate_boxes, boxes))

        candidate_scores = np.concatenate((candidate_scores, scores))
        candidate_windows = np.concatenate((candidate_windows, windows))

    # Delete overlapping boxes using non-maxima suppression
    if len(candidate_scores) > 0:
        validated_boxes = nonMaxSuppression(candidate_boxes, candidate_scores)

    return validated_boxes, windows


classifier = svm.LinearSVC()
classifier = classifierTraining(extracted_pos_faces_path, extracted_neg_faces_path)




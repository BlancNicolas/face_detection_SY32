#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 19:12:33 2018

@author: Nico
"""
from PIL import Image
import os, glob, errno
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.utils import shuffle
from sklearn import cross_validation
from sklearn import svm
from skimage.feature import hog
from skimage.color import rgb2gray
from skimage.transform import resize
from skimage import io
from skimage import util
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
        img = resize(img, (64, 64))
        hog_train.append(hog(img))

    hog_train, labels = shuffle(hog_train, labels)
    return clf.fit(hog_train, labels)

#-----------------------------------------------
# ---------- slidingWindow ------------------
# INPUT :
    # image : image we are sliding into
    # step_size :

# OUTPUT : Linear SVC classifier
#-----------------------------------------------
def slidingWindow(image, step_size, window_size):
	# slide a window across the image
	for y in range(0, image.shape[0], step_size):
		for x in range(0, image.shape[1], step_size):
			# yield the current window
			yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])


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


#-----------------------------------------------
# ---------- learningFromData ------------------
# INPUT :
    # path_raw_data : path where raw train data are stored

# OUTPUT : Linear SVC classifier
#-----------------------------------------------
def learningFromData(path_raw_data, labels, classifier):
    train_dir_content = sorted(glob.glob(img_train_path))
    for idx, path in enumerate(train_dir_content):
        image = io.imread(path)
        image = rgb2gray(image)
        final_boxes = []
        final_scores = []
        scores = []
        #Pyramid on current image
        for (i, resized) in enumerate(transform.pyramid_gaussian(image, downscale = 2)):
            boxes_list = []
            if resized.shape[0] < 32 or resized.shape[1] < 32:
                break
            for (x, y, window) in slidingWindow(resized, step_size = 16, window_size = WINDOW_SIZE):
                if window.shape[0] != WINDOW_SIZE[0] or window.shape[1] != WINDOW_SIZE[1]:
                    continue

                if classifier.predict([hog(window, cells_per_block = (1,1))]):
                    boxes_list.append([x, y, x + WINDOW_SIZE[0], y + WINDOW_SIZE[1]])
                    scores.append(classifier.decision_function(hog(window)))
            nms_box = nonMaxSuppression(boxes_list, scores)
            rescaled_nms_box = rescaleWindow(nms_box[:,0],
                                             nms_box[:,1],
                                             nms_box[:,2] - nms_box[:,0],
                                             nms_box[:,3] - nms_box[:,1],
                                             image.shape[0],
                                             resized.shape[0])
            final_boxes.append(rescaled_nms_box)
            final_scores.append(classifier.decision_function(hog(rescaled_nms_box)))
        final_box = nonMaxSuppression(final_boxes, final_scores)
        displayRectOnImg(image, [final_box[0],
                                 final_box[1],
                                 final_box[2]-final_box[0],
                                 final_box[3]-final_box[1]])

#classifier = svm.LinearSVC()
#classifier = classifierTraining(extracted_pos_faces_path, extracted_neg_faces_path)




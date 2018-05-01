#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 12:19:10 2018

@author: Nico
"""
from learning.train import *
from util.utils import *
from util.NMS import *
import numpy as np
from skimage import io
from skimage import transform
from constants import *
import matplotlib.pyplot as plt
import matplotlib.patches as patches


#image = io.imread(img_train_dir_content[0])
#print("Shape 0 : ", image.shape[0])
#print("Shape 1 : ", image.shape[1])

#Example of the using of pyramid gaussian
#for (i, resized) in enumerate(transform.pyramid_gaussian(image, downscale = 2)):
#        if resized.shape[0] < 32 or resized.shape[1] < 32:
#            break
def displayRectOnImg(image, rect_coord):
    # Create figure and axes
    fig,ax = plt.subplots(1)
    # Display the image
    ax.imshow(image)
    # Create a Rectangle patch
    rect = patches.Rectangle((rect_coord[0], rect_coord[1]), rect_coord[2],rect_coord[3],linewidth=1,edgecolor='r',facecolor='none')
    # Add the patch to the Axes
    ax.add_patch(rect)
    plt.show()

def learningFromImage(path_raw_data, labels, classifier):
        print("-- Learning from image --")
        image = io.imread(path_raw_data)
        image = rgb2gray(image)
        final_boxes = np.empty((0,4))
        final_scores = np.array([])
        label = labels[123]
        #Pyramid on current image
        for (i, resized) in enumerate(transform.pyramid_gaussian(image, downscale = 1.5)):
            if resized.shape[0] < 32 or resized.shape[1] < 32:
                break
            step = 16
            windows, boxes = slidingWindow(resized, step_size = step, window_size = WINDOW_SIZE)
            print("Window : ", windows.shape)
            features = np.empty((len(windows), 324))
            for (idx, i) in enumerate(windows):
                features[idx] = hog(i)
            predictions = classifier.predict(features)
            scores = classifier.decision_function(features)
            mask = np.zeros(features.shape[0], dtype = bool)
            mask[predictions == 1] = True

            #Filtering
            boxes = boxes[mask,:]
            scores = scores[mask]

            rescaled_boxes = rescaleBoxes(boxes, image.shape, resized.shape)

            final_scores = np.concatenate((final_scores, scores))
            final_boxes = np.concatenate((final_boxes, rescaled_boxes))

        if len(final_scores)>0:
            validated_boxes, scores = nonMaxSuppression(final_boxes, final_scores)

        displayRectOnImg(image, label[1:])
        for (idx, i) in enumerate(validated_boxes):
            displayRectOnImg(image, [validated_boxes[idx, 0],
                                     validated_boxes[idx, 1],
                                     validated_boxes[idx, 2]-validated_boxes[idx, 0],
                                     validated_boxes[idx, 3]-validated_boxes[idx, 1]])

learningFromImage("/Users/Nico/DocumentsOnMac/UTC/P18/SY32/face_detection_SY32/projetface/train/0124.jpg",
                  label,
                  classifier)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 12:19:10 2018

@author: Nico
"""
from train import *
from utils import *
from PIL import Image
import os, glob, errno
import numpy as np
from skimage import io
from skimage import transform
from constants import *

#image = io.imread(img_train_dir_content[0])
#print("Shape 0 : ", image.shape[0])
#print("Shape 1 : ", image.shape[1])

#Example of the using of pyramid gaussian
#for (i, resized) in enumerate(transform.pyramid_gaussian(image, downscale = 2)):
#        if resized.shape[0] < 32 or resized.shape[1] < 32:
#            break

def learningFromImage(path_raw_data, labels, classifier):
        print("-- Learning from image --")
        image = io.imread(path_raw_data)
        image = rgb2gray(image)
        final_boxes = []
        final_scores = []
        scores = []
        label = labels[124]
        #Pyramid on current image
        for (i, resized) in enumerate(transform.pyramid_gaussian(image, downscale = 1.2)):
            print("Down sampling image")
            boxes_list = []
            if resized.shape[0] < 32 or resized.shape[1] < 32:
                break
            step = min(resized.shape[0], resized.shape[1])
            for (x, y, window) in slidingWindow(resized, step_size = step, window_size = WINDOW_SIZE):
                print("Sliding in window")
                if window.shape[0] != WINDOW_SIZE[0] or window.shape[1] != WINDOW_SIZE[1]:
                    continue
                features = hog(window)
                if classifier.predict([features]):
                    print("Prediction is okay")
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
        displayRectOnImg(image, [label[0], label[1], label[2], label[3]])
        displayRectOnImg(image, [final_box[0],
                                 final_box[1],
                                 final_box[2]-final_box[0],
                                 final_box[3]-final_box[1]])

learningFromImage("/Users/Nico/DocumentsOnMac/UTC/P18/SY32/face_detection_SY32/projetface/train/0124.jpg",
                  label,
                  classifier)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 18:31:30 2018

@author: Nico
"""

from learning.train import *
from constants import *
from util.utils import *
from util.dataExtraction import importImages
from util.NMS import *


#-----------------------------------------------
# ---------- detectFaces ------------------
# Goal : This functions aims at detecting faces the input image.
# This works in several steps as follow :
#   1. Compute the gaussian pyramid of the image to make the detection on several sizes
#   2. For each pyramid level : compute sliding windows and retrieve candidate boxes the classifier detects
#   3. Rescale candidate boxes to fit on the original image shape
#   4. Eliminate overlapping candidate boxes using non-maxima suppression
#   5. Return remaining boxes as validated boxes
# INPUT :
#   - image : image to detect face on
#   - classifier : trained classifier
#   - threshold : minimum score to for a window to be detected as a face
#
# OUTPUT :
#   - validated_boxes : Boxes corresponding to detected faces on the input image
#-----------------------------------------------
def detectFaces(image, classifier, threshold = 0.5):
    candidate_boxes = np.empty((0,4))
    candidate_scores = np.array([])
    validated_boxes = np.empty((0,4))
    validated_scores = np.empty(0)

    # Pyramid on current image
    for (i, resized) in enumerate(pyramid_gaussian(image, downscale = 1.5)):
        if resized.shape[0] < WINDOW_SIZE[0] or resized.shape[1] < WINDOW_SIZE[1]:
            break

        # Create list of successive sliding windows with corresponding boxes.
        windows, boxes = slidingWindow(resized, step_size = 16, window_size = WINDOW_SIZE)

        # Compute hog for each sliding window
        features = np.array([hog(windows[0])])
        if len(windows) > 1:
            for w in windows[1:]:
                features = np.concatenate((features, [hog(w)]), axis = 0)

        # Compute scores based on given classifiers
        scores = classifier.decision_function(features)

        # Keep only boxes ith a detection probability above 50%
        mask = np.zeros(features.shape[0], dtype = bool)
        mask[scores > threshold] = True
        boxes = boxes[mask]
        scores = scores[mask]

        # Rescale boxes if image is resized
        if i > 1:
            rescaled_boxes = rescaleBoxes(boxes, image.shape, resized.shape)
            candidate_boxes = np.concatenate((candidate_boxes, rescaled_boxes))
        else:
            candidate_boxes = np.concatenate((candidate_boxes, boxes))

        candidate_scores = np.concatenate((candidate_scores, scores))

    # Delete overlapping boxes using non-maxima suppression
    if len(candidate_boxes) == 1:
        return candidate_boxes.astype('int'), candidate_scores.astype('float')
    elif len(candidate_scores) > 1:
        pick = nonMaxSuppression(candidate_boxes, candidate_scores)
        validated_boxes = candidate_boxes[pick].astype('int')
        validated_scores = candidate_scores[pick].astype('float')

    return validated_boxes, validated_scores


#-----------------------------------------------
# ---------- validateTraining ------------------
# Goal : This functions retrieves the false positives in detected faces
# INPUT :
#   - img : image on which the face detection has been done
#   - boxes : boxes containing the detected face (format X1 Y1 X2 Y2)
#   - label : label of training image of the form (format X Y L H)
#
# OUTPUT :
#   - false_pos : Boxes corresponding to false postives among input boxes
#-----------------------------------------------
def validateFaceDetection(images, labels, clf):
    false_pos = []
    err = 0
    for (idx, img) in enumerate(images):
        boxes, scores = detectFaces(img, clf)
        label = labels[idx, 1:]
        label_box = [label[0], label[1], label[0] + label[2], label[1] + label[3]]
        for box in boxes:
            overlap = compareAreas(box, label_box)
            if overlap < 0.5:
                false_pos.append(img[box[1]:box[3], box[0]:box[2]])
        if len(false_pos) == len(boxes):
            err += 1
    err_rate = err * 100 / len(images)
    return err_rate, false_pos


def applyClfOnTestImages(test_images, clf, scoreThresh):
    res = np.empty((0, 6))
    for (idx, img) in enumerate(test_images):
        boxes, scores = detectFaces(img, clf, scoreThresh)
        if len(boxes) > 0:
            # format boxes to X Y W H
            boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
            boxes[:, 3] = boxes[:, 3] - boxes[:, 1]
            # concatenate scores and indexes to box
            print("Length : {}".format(len(boxes)))
            indexes = np.ones((len(boxes), 1), dtype='int') * (idx + 1)
            scores = scores.reshape(-1, 1)
            labels = np.concatenate((indexes, boxes, scores), axis=1)
            res = np.concatenate((res, labels), axis=0)

    np.savetxt(result_path, res, fmt=('%.4d', '%d', '%d', '%d', '%d', '%.2f'))


def trainTestAndStore():
    images = importImages(img_train_path)
    labels = np.loadtxt(label_path).astype('int')
    pos = importImages(extracted_pos_faces_path)
    neg = importImages(extracted_neg_faces_path)

    clf, err_rate = trainAndValidate(images, labels, pos, neg, 5, iter_max=3)

    test_images = importImages(img_train_path)
    applyClfOnTestImages(test_images, clf, 0.6)


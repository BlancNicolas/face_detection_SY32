#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 19:12:33 2018

@author: Nico
"""

from sklearn.utils import shuffle
from sklearn import svm
from skimage.feature import hog
from skimage.transform import resize
from util.dataExtraction import *
from learning.test import validateFaceDetection


#-------------------------------------------------
# ---------- classifierTraining ------------------
# INPUT :
    # neg_path : path where negative samples are stored
    # pos_path : path where positive samples are stored

# OUTPUT : Linear SVC classifier
#-------------------------------------------------
def classifierTraining(pos, neg, c_param = 1.0):
    print(" -- Training a classifier SVC Linear --")
    # Initializing classifier
    clf = svm.LinearSVC(C = c_param)
    images = np.concatenate((pos, neg))
    labels = createLabels(len(pos), len(neg))
    hog_train = []
    for img in images:
        img = resize(img, (32, 32))
        hog_train.append(hog(img))
    hog_train, labels = shuffle(hog_train, labels)
    return clf.fit(hog_train, labels)


#-----------------------------------------------
# ---------- crossValidTraining ------------------
# Goal : This functions performs a cross validation of face detection
# INPUT :
#   - x : list of images
#   - y : list of labels
#   - k : number of partitions wanted for the cross-validation
#
# OUTPUT :
#   - mean_err : mean error of the detection
#-----------------------------------------------
def crossValidTraining(x, y, k):
    x, y = shuffle(x, y)
    mean_err = 0
    for i in range(k):
        mask = np.zeros(x.shape[0], dtype=bool)
        mask[np.arange(i, mask.size, k)] = True
        clf = classifierTraining(x[~mask], y[~mask])
        err_rate, false_pos = validateFaceDetection(x[mask], y[mask], clf)
        mean_err += err_rate
    mean_err /= k
    return mean_err


#-----------------------------------------------
# ---------- trainAndValidate ------------------
# Goal : This functions trains a classifier and validate the model
# using hard negative mining and cross validation.
# This works in several steps as follow :
#   1. Train the classifier a first time using given pos and neg images
#   2. Retrieve error ratio and false positive, add them into the folder "falsePos"
#   3. Do the hard negative mining, which is repeating step 1 and 2 until the convergence is reached
#   4. Convergence is reached when the difference of error rate between two iterations is lower than input threshold.
# INPUT :
#   - images :set of images to test the trained classifier
#   - labels : labels of the set of images
#   - pos : set of positive faces images to train the classifier
#   - neg : set of negatives faces images to train the classifier
#   - convThresh : threshold expressed in % to know when to stop hard negative mining
# OUTPUT :
#   - clf : trained classifier
#-----------------------------------------------
def trainAndValidate(images, labels, pos, neg, convThresh, iter_max):
    # train classifier
    clf = classifierTraining(pos, neg)

    # apply classifier on train_images and retrieve false positives
    err_rate, false_pos = validateFaceDetection(images, labels, clf)
    print("Info : {}% error after first training".format(err_rate))
    print("Info : Number of False postive after first training : {}".format(len(false_pos)))

    # i will serve to know the current iteration and to distinguish false positives names of different iterations
    i = 1

    # Store false positives in the directory falsePos
    storeImages(false_pos, fp_path + str(i))

    # Hard negative mining
    prev_err_rate = err_rate
    converged = False
    niter = 0
    while not converged or niter < iter_max:
        i += 1
        niter += 1
        # train classifier again with new negative faces
        neg += false_pos
        clf = classifierTraining(pos, neg)
        err_rate, false_pos = validateFaceDetection(images, labels, clf)
        print("Info : {}% error after {}e training".format(err_rate, i))
        print("Info : Number of False postive after {}e training : {}".format(i, len(false_pos)))
        # Store false positives in the directory falsePos
        storeImages(false_pos, fp_path + str(i))

        # Check the convergence
        converged = abs(err_rate - prev_err_rate) < convThresh
        prev_err_rate = err_rate

    # TODO : adjust parameters using cross validation

    return clf, err_rate

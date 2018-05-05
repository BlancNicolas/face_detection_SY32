#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 19:12:33 2018

@author: Nico
"""

from sklearn.utils import shuffle
from sklearn import svm
from dataExtraction import *


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
def validateFaceDetection(images, labels, clf, overlapThresh):
    print("Info : Validating detection")
    false_pos = []
    for (idx, img) in enumerate(images):
        boxes, scores = detectFaces(img, clf)
        label = labels[idx, 1:]
        label_box = [label[0], label[1], label[0] + label[2], label[1] + label[3]]
        for box in boxes:
            overlap = compareAreas(box, label_box)
            if overlap < overlapThresh:
                false_pos.append(img[box[1]:box[3], box[0]:box[2]])

    return false_pos


#-----------------------------------------------
# ---------- crossValidTraining ------------------
# Goal : This functions performs a cross validation of face detection
# INPUT :
#   - x : list of images
#   - y : list of labels
#   - k : number of partitions wanted for the cross-validation
#   - C_values : C values to test for chosing best classifier
#
# OUTPUT :
#   - mean_err : mean error of the detection
#   - clf : best classifier fit with all input data
#-----------------------------------------------
def crossValidTraining(x, y, k=5, C_values=[0.01, 0.05, 0.1, 0.5, 1, 3, 5, 10]):
    x, y = shuffle(x, y)
    mean_err = 1
    for C in C_values:
        clf = svm.LinearSVC(C = C)
        errors = np.zeros(k)
        for i in range(k):
            mask = np.zeros(x.shape[0], dtype=bool)
            mask[np.arange(i, mask.size, k)] = True
            clf.fit(x[~mask], y[~mask])
            errors[i] = np.mean(clf.predict(x[mask]) != y[mask])
        e = np.mean(errors)
        if e < mean_err:
            mean_err = e
            best_clf = clf
            best_clf.fit(y, y)

    return mean_err, clf


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
#   - mean_err : mean_err associated with the classifier
#-----------------------------------------------
def trainAndValidate(images, labels, pos, neg):
    import warnings
    warnings.filterwarnings('ignore') # Removing anoying warnings

    # train classifier
    examples = np.concatenate((pos, neg))
    examples_features = computeHogs(examples, resize_images=True)
    examples_labels = createLabels(len(pos), len(neg))

    mean_err, clf = crossValidTraining(examples_features, examples_labels)
    print("Info : Classifier trained. Mean Error during training : {]".format(mean_err))
    # apply classifier on train_images and retrieve false positives
    false_pos = validateFaceDetection(images, labels, clf, overlapThresh=0.1)
    print("Info : Number of False Postive after first training : {}".format(len(false_pos)))

    # Store false positives in the directory falsePos
    storeImages(false_pos, fp_path)
    print("Info : images stored")

    # Hard negative mining
    # train classifier again with new negative faces
    neg += false_pos
    mean_err, clf = crossValidTraining(examples_features, examples_labels)
    print("Info : Classifier trainedwith new false positive. Mean Error during training : {]".format(mean_err))

    return mean_err, clf

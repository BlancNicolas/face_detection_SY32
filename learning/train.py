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
from util.data_extraction import *
from learning.test import validateFaceDetection
label = np.loadtxt(label_path)


#-------------------------------------------------
# ---------- classifierTraining ------------------
# INPUT :
    # neg_path : path where negative samples are stored
    # pos_path : path where positive samples are stored

# OUTPUT : Linear SVC classifier
#-------------------------------------------------
def classifierTraining(images, labels, c_param = 1.0):
    print(" -- Training a classifier SVC Linear --")
    # Initializing classifier
    clf = svm.LinearSVC(C = c_param)
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
# ---------- importImages ------------------
# Goal : This functions aims at importing all images of a directory as a list.
# INPUT :
#   - dir_path : directory of images, of the form "path/to/dir/*.jpg"
#
# OUTPUT :
#   - validated_boxes : Boxes corresponding to detected faces on the input image
#-----------------------------------------------
def train():
    # import images
    train_images = importImages(img_train_path)
    train_labels = np.loadtxt(label_path)
    pos_faces = importImages(extracted_pos_faces_path)
    neg_faces = importImages(extracted_neg_faces_path)

    # train classifier
    train_faces = np.concatenate((pos_faces, neg_faces))
    face_labels = createLabels(len(pos_faces), len(neg_faces))
    clf = classifierTraining(train_faces, face_labels)

    # apply classifier on train_images and retrieve false positives
    err_rate, new_neg_faces = validateFaceDetection(train_images, train_labels, clf)
    print("Info : {}% error after first training".format(err_rate))

    # train classifier with new negative faces
    train_faces = np.concatenate((train_faces, new_neg_faces))
    face_labels += createLabels(0, len(new_neg_faces))
    clf = classifierTraining(train_faces, face_labels)

    # TODO : adjust parameters using cross validation


classifier = svm.LinearSVC()
classifier = classifierTraining(extracted_pos_faces_path, extracted_neg_faces_path)




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

#Classifier arguments
c_param = 1.0

img_train_dir_content = sorted(glob.glob(img_train_path))
label = np.loadtxt(label_path)

#-------------------------------------------
# ---------- classifierTraining ------------------
# INPUT :
    # neg_path : path where negative samples are stored
    # pos_path : path where positive samples are stored

# OUTPUT : Linear SVC classifier
#-------------------------------------------
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

classifier = svm.LinearSVC()
classifier = classifierTraining(extracted_pos_faces_path, extracted_neg_faces_path)




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
from skimage import io
from skimage import util
from constants import *

#Classifier arguments
c_param = 1.0

img_train_dir_content = sorted(glob.glob(img_train_path))
label = np.loadtxt(label_path)
clf = svm.LinearSVC(C = c_param)

def classifierTraining(neg_path, pos_path):

    train_dir_content = np.concatenate((glob.glob(pos_path),glob.glob(neg_path)), axis=0)
    images = io.imread(train_dir_content)
    hog_train = np.empty(len(images))

    for idx, img in enumerate(images):
        hog_train[idx] = hog(img)

    clf.fit(hog_train)





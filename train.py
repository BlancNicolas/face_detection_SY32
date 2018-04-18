#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 19:12:33 2018

@author: Nico
"""
import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from skimage.feature import hog
from skimage import io
from sklearn import cross_validation 
from skimage import util
import glob
import os

#--------------------------------------------
#----Loading training positive images -------
img_face_dir_content = glob.glob("projetface/train/*.jpg")
pos_faces = np.zeros((len(img_face_dir_content),128,64))
j = 0
for x in img_face_dir_content:
    pos_faces[j] = io.imread(x)
    j = j+1
    
#--------------------------------------------
#----Loading label -------
label = np.loadtxt("projetface/label.txt")
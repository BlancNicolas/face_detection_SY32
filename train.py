#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 19:12:33 2018

@author: Nico
"""
import io as io
import os
import glob
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

img_train_dir_content = sorted(glob.glob(img_train_path))
label = np.loadtxt(label_path)

def loadImagesDirectory():
    #--------------------------------------------
    #----Loading training positive images -------
    images[j] = io.imread(x, as_grey=True)

def loadLabels():
    #--------------------------------------------
    #----Loading label -------
    label = np.loadtxt("projetface/label.txt")
    
def displayRectOnImg(image, rect_coord):
    # Create figure and axes
    fig,ax = plt.subplots(1)
    
    # Display the image
    ax.imshow(image)
    # Create a Rectangle patch
    rect = patches.Rectangle((rect_coord[1], rect_coord[2]), rect_coord[3],rect_coord[4],linewidth=1,edgecolor='r',facecolor='none')
    # Add the patch to the Axes
    ax.add_patch(rect)
    
    plt.show()


    

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 19:12:33 2018

@author: Nico
"""

from PIL import Image
import os, glob, errno
import numpy as np
from skimage import io
from skimage import util
from constants import *
from train import *

def extractPosFaces(path, label):
    # Creating directory where we'll put positive faces
    try:
        os.makedirs(root_path + extracted_pos_faces_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    j=0
    for i in img_train_dir_content:
        img = Image.open(i)
        # Defining box to extract defined in label
        coord = label[j]
        box = (coord[1], coord[2], coord[1] + coord[3], coord[2] + coord[4])
        # Crop image
        face = img.crop(box)
        # Save image
        face.save(root_path + extracted_pos_faces_path + '/' + os.path.basename(i), 'JPEG')
        j = j + 1

def generateNegativeSample(path, label):
    # Creating directory where we'll put positive faces
    try:
        os.makedirs(root_path + extracted_neg_faces_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


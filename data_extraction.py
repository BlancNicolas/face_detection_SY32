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
from random import randint
from matplotlib import image

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
    for (idx, content) in enumerate(img_train_dir_content):
        print("idx :", idx)
        img = io.imread(content)
        # Defining box to extract defined in label
        width = randint(MIN_WIDTH, MAX_WIDTH)
        factor = randint(10, 20)/10
        height = int(width * factor)
        x1 = randint(0, img.shape[0])
        y1 = randint(0, img.shape[1])
        x2 = x1 + width
        y2 = y1 + height
        # Crop image
        neg_face = img[x1:x2, y1:y2]
        # Path
        path = root_path + extracted_neg_faces_path + '/' + "0" + str(1+len(img_train_dir_content)+idx) + ".png"
        # Save image
        image.imsave(path, neg_face)

generateNegativeSample(img_train_dir_content, label)


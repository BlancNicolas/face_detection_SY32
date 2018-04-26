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
from util.utils import *
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

    samples_number = 0
    idx = 0
    while samples_number < NEGATIVE_SAMPLES_NUMBER and idx < 1000:
        print("idx :", idx)
        img = image.imread(img_train_dir_content[idx])
        # Defining box to extract defined in label
        factor = randint(10, 20)/10
        x1 = randint(0, img.shape[0] - MIN_WIDTH)
        y1 = randint(0, img.shape[1] - MIN_HEIGHT)
        height = randint(MIN_HEIGHT, img.shape[1] - y1)
        width = int(height / factor)
        if width > img.shape[0]:
            width = img.shape[0]
        x2 = x1 + width
        y2 = y1 + height
        # Check if the box is not similar to the real box in label
        current_label = label[idx]
        area_label = [current_label[1], current_label[2], current_label[1]+current_label[3], current_label[2]+current_label[4]]
        area_new_box = [x1, y1, x2, y2 ]
        if not compareAreas(area_new_box, area_label) > 0.1:
            # Crop image
            print(x1, y1, x2, y2)
            neg_face = img[x1:x2, y1:y2]
            # Path
            path = root_path + extracted_neg_faces_path + '/' + "0" + str(1+len(img_train_dir_content)+idx) + ".jpg"
            # Save image
            print("box shape 0 : ", neg_face.shape[0])
            print("box shape 1 : ", neg_face.shape[1])
            image.imsave(path, neg_face)
            idx = idx + 1


generateNegativeSample(img_train_dir_content, label)


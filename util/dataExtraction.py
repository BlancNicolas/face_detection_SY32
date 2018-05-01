#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 19:12:33 2018

@author: Nico
"""

import glob, errno
from skimage.color import rgb2gray
from skimage import io
from util.utils import *
from constants import *
from random import randint
from PIL import Image

img_train_dir_content = sorted(glob.glob(img_train_path))


#-----------------------------------------------
# ---------- importImages ------------------
# Goal : This functions aims at importing all images of a directory as a list.
# INPUT :
#   - dir_path : directory of images, of the form "path/to/dir/*.jpg"
#
# OUTPUT :
#   - images : list of imported images
#-----------------------------------------------
def importImages(dir_path):
    images = []
    for fimage in glob.glob(dir_path):
        img = io.imread(fimage)
        img = rgb2gray(img)
        images.append(img)
    return images


def storeImages(images, path):
    i = 1
    for img in images:
        name = "000{}.jpg".format(i)
        io.imsave(path + name, img)
        i += 1


#-----------------------------------------------
# ---------- extractPosFaces  ------------------
# GOAL : extract positive faces from train images thanks to labels
# INPUT :
    # path : path where train images are stored
    # label : array of labels

# OUTPUT : none
#-----------------------------------------------
def extractPosFaces(path, label):
    print(" -- Extracting positive faces -- ")
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

#-------------------------------------------------------
# ---------- generateNegativeSamples  ------------------
# GOAL : generates random negative samples from train images
# INPUT :
    # path : path where train images are stored
    # label : array of labels

# OUTPUT : none
#-------------------------------------------------------
def generateNegativeSamples(path, label):
    print(" -- Generating negative samples -- ")
    # Creating directory where we'll put positive faces
    try:
        os.makedirs(root_path + extracted_neg_faces_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    samples_number = 0
    idx = 0
    while samples_number < NEGATIVE_SAMPLES_NUMBER and idx < 1000:
        img = Image.open(img_train_dir_content[idx])
        # Defining box to extract defined in label
        factor = randint(10, 20)/10
        x1 = randint(0, img.size[0] - MIN_WIDTH)
        y1 = randint(0, img.size[1] - MIN_HEIGHT)
        height = randint(MIN_HEIGHT, img.size[1] - y1)
        width = int(height / factor)
        if x1 + width > img.size[0]:
            width = img.size[0] - x1
            height = int(width * factor)

        # Check if the box is not similar to the real box in label
        current_label = label[idx]
        area_label = [current_label[1], current_label[2], current_label[1]+current_label[3], current_label[2]+current_label[4]]
        area_new_box = [x1, y1, x1 + width, y1 + height ]
        if not compareAreas(area_new_box, area_label) > 0.1:
            # Box
            crop_box = [x1, y1, x1 + width, y1 + height]
            # Crop image
            neg_face = img.crop(crop_box)
            # Path
            path = root_path + extracted_neg_faces_path + '/' + "0" + str(1+len(img_train_dir_content)+idx) + ".jpg"
            # Save image
            neg_face.save(path)
            idx = idx + 1

#generateNegativeSamples(img_train_dir_content, label)


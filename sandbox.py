#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 12:19:10 2018

@author: Nico
"""
from train import *
from utils import *
from PIL import Image
import os, glob, errno
import numpy as np
from skimage import io
from constants import *

image = io.imread(img_train_dir_content[0])
print("Shape 0 : ", image.shape[0])
print("Shape 1 : ", image.shape[1])

#Example of the using of pyramid gaussian
for (i, resized) in enumerate(transform.pyramid_gaussian(image, downscale = 2)):
        if resized.shape[0] < 32 or resized.shape[1] < 32:
            break

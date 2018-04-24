#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 12:19:10 2018

@author: Nico
"""

from train import *
from PIL import Image

image = io.imread(img_train_dir_content[0])
print("Shape 0 : ", image.shape[0])
print("Shape 1 : ", image.shape[1])
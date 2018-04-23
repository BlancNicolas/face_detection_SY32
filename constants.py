#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 15:20:48 2018

@author: Nico
"""
import os


root_path = os.path.dirname(os.path.abspath(__file__))
print("Root : ",root_path)
img_train_path = root_path + "/projetface/train/*.jpg"
label_path = root_path + "/projetface/label.txt"
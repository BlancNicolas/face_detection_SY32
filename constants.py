#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 15:20:48 2018

@author: Nico
"""
import os

#Constants for paths
root_path = os.path.dirname(os.path.abspath(__file__))
img_train_path = root_path + "/projetface/train/*.jpg"
label_path = root_path + "/projetface/label.txt"
extracted_faces_path = "/extractedData"
pos_faces = "/pos"
neg_faces = "/neg"
extracted_pos_faces_path = extracted_faces_path + pos_faces
extracted_neg_faces_path = extracted_faces_path + neg_faces

#Constant for the size of sliding window
WINDOW_SIZE = [32, 32]


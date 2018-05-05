#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 18:31:30 2018

@author: Nico
"""

from train import *
from constants import *
from utils import *
from dataExtraction import importImages
from NMS import *


def applyClfOnTestImages(test_images, clf, scoreThresh):
    print("Info : Test on images")
    res = np.empty((0, 6))
    for (idx, img) in enumerate(test_images):
        boxes, scores = detectFaces(img, clf, scoreThresh)
        if len(boxes) > 0:
            # format boxes to X Y W H
            boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
            boxes[:, 3] = boxes[:, 3] - boxes[:, 1]
            # concatenate scores and indexes to box
            print("Length : {}".format(len(boxes)))
            indexes = np.ones((len(boxes), 1), dtype='int') * (idx + 1)
            scores = scores.reshape(-1, 1)
            labels = np.concatenate((indexes, boxes, scores), axis=1)
            print("Labels : ", labels)
            res = np.concatenate((res, labels), axis=0)
    print("Res : ", res)
    np.savetxt(result_path, res, fmt=('%.4d', '%d', '%d', '%d', '%d', '%.2f'))
    print("Info : Test finished, you can observe results in {}".format(result_path))


def trainTestAndStore():
    print("Importing data...")
    images = importImages(img_train_path)
    labels = np.loadtxt(label_path).astype('int')
    pos = importImages(extracted_pos_faces_path)
    neg = importImages(extracted_neg_faces_path)
    test_images = importImages(img_test_path)

    print("Data imported.")

    clf, err_rate = trainAndValidate(images, labels, pos, neg)

    applyClfOnTestImages(test_images, clf, 0.4)


trainTestAndStore()



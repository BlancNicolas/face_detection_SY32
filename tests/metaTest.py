#!/usr/bin/python3
# -*- coding: utf-8 -*-

from test import *
from util.dataExtraction import importImages

sample_size = 200
pos = importImages(extracted_pos_faces_path)[0:sample_size]
neg = importImages(extracted_neg_faces_path)[0:sample_size]

clf = classifierTraining(pos, neg)


def test_detectFaces():
	image = io.imread(img_train_path.replace('*', '0001'))
	image = rgb2gray(image)
	boxes, scores = detectFaces(image, clf)
	print("# Boxes : {}".format(len(boxes)))
	print("# Scores : {}".format(len(scores)))
	# Display the print in test
	assert not True


def test_validation():
	images = importImages(img_train_path)[0:sample_size]
	labels = np.loadtxt(label_path).astype('int')
	err_rate, false_pos = validateFaceDetection(images, labels, clf)
	print("Error rate : {}".format(err_rate))
	print("# False Positive : {}".format(len(false_pos)))
	# Display the print in test
	assert not True


def test_applyClf():
	test_images = importImages(img_test_path)
	applyClfOnTestImages(test_images, clf, 0.5)
	assert True

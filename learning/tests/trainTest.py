#!/usr/bin/python3
# -*- coding: utf-8 -*-

from learning.train import *
from sklearn import svm
from skimage.feature import hog


def test_detectFaces():
	image = io.imread(root_path + "/projetface/train/0001.jpg")
	image = rgb2gray(image)
	pos_example = io.imread(root_path + extracted_faces_path + "/pos/0001.jpg")
	pos_example = resize(rgb2gray(pos_example), (32, 32))
	neg_example = io.imread(root_path + extracted_faces_path + "/neg/01001.jpg")
	neg_example = rgb2gray(neg_example)

	train = np.array((hog(pos_example), hog(neg_example)))
	labels = np.array((1, 0))
	print("Train shape {}".format(train.shape))

	clf = svm.LinearSVC(C = 0.1)
	clf.fit(train, labels)

	boxes, windows = detectFaces(image, clf)

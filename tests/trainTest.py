#!/usr/bin/python3
# -*- coding: utf-8 -*-

from train import *

images = importImages(img_train_path)
labels = np.loadtxt(label_path).astype('int')
pos = importImages(extracted_pos_faces_path)
neg = importImages(extracted_neg_faces_path)


def test_train_sample():
	sample_size = 50
	clf, err_rate = trainAndValidate(images[0:sample_size],
									 labels[0:sample_size],
									 pos[0:sample_size],
									 neg[0:sample_size], 100)
	print("Classifier : {}\nError Ratio : {}".format(clf, err_rate))
	assert not True


def test_train_full():
	clf, err_rate = trainAndValidate(images, labels, pos, neg, 10)
	print("Classifier : {}\nError Ratio : {}".format(clf, err_rate))
	assert not True
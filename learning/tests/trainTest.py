#!/usr/bin/python3
# -*- coding: utf-8 -*-

from learning.train import *
from sklearn import svm
from skimage.feature import hog


def test_train():
	clf = train()
	print(clf)
	assert not True
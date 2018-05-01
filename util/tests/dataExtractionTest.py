#!/usr/bin/python3
# -*- coding: utf-8 -*-

from util.dataExtraction import *


def test_storeImages():
	test_path = "/tmp/"
	images = importImages(img_train_path)[0:5]
	storeImages(images, test_path)
	assert True

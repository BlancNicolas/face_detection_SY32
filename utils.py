#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 12:19:10 2018

@author: Nico
"""

#-------------------------------------------
# ---------- displayRectOnImg ------------------
# INPUT :
    # image : lthe image on which we want to put rectangle
    # rect_coord : coordonates of the rectangle

# OUTPUT : numpy array of yes and no labels
#-------------------------------------------

def displayRectOnImg(image, rect_coord):
    # Create figure and axes
    fig,ax = plt.subplots(1)
    # Display the image
    ax.imshow(image)
    # Create a Rectangle patch
    rect = patches.Rectangle((rect_coord[1], rect_coord[2]), rect_coord[3],rect_coord[4],linewidth=1,edgecolor='r',facecolor='none')
    # Add the patch to the Axes
    ax.add_patch(rect)

    plt.show()

#-------------------------------------------
# ---------- createLabels ------------------
# INPUT :
    # len_yes : length of yes labels we want
    # len_no : length of no labels we want

# OUTPUT : numpy array of yes and no labels
#-------------------------------------------
def createLabels(len_yes, len_no):
    return np.concatenate((np.ones(len_yes), np.zeros(len_no)), axis = 0)

import cv2
import os
import numpy as np


def loadImages(dirName):
    fileNames = list(filter(
        lambda name: name.find(".png") != -1,
        os.listdir(dirName)))

    firstFile = cv2.imread(os.path.join(dirName, fileNames[0]))
    (y, x, c) = firstFile.shape
    images = np.empty((len(fileNames), y, x, c), np.uint8)
    images[0] = firstFile

    for i in range(len(fileNames))[1:]:
        img = cv2.imread(os.path.join(dirName, fileNames[i]))
        images[i] = img

    return images


def getCategories(dirName):
    directories = list(filter(
        lambda directory: os.path.isdir(os.path.join(dirName, directory)),
        os.listdir(dirName)))

    categories = {}
    for directory in directories:
        categories[directory] = loadImages(os.path.join(dirName, directory))

    return categories

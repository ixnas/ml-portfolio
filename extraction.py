import cv2
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial import distance
import matplotlib.pyplot as plt

import processing


def extractSiftDescriptors(img, mask):
    sift = cv2.SIFT_create()
    (kp, des) = sift.detectAndCompute(img, mask)

    return des


def getVisualWords(k, descriptorList):
    kmeans = KMeans(n_clusters=k, n_init=10)
    kmeans.fit(descriptorList)
    visualWords = kmeans.cluster_centers_
    return visualWords


def getSiftFeatures(categories):
    siftVectors = {}
    descriptorList = []

    for category, images in categories.items():
        features = []

        for img in images:
            preProcessed = processing.getProcessedImage(img)
            mask = processing.getMask(preProcessed)
            des = extractSiftDescriptors(preProcessed, mask)
            descriptorList.extend(des)
            features.append(des)

        siftVectors[category] = features

    return (descriptorList, siftVectors)


def find_index(image, center):
    count = 0
    ind = 0
    for i in range(len(center)):
        if(i == 0):
            count = distance.euclidean(image, center[i])
        else:
            dist = distance.euclidean(image, center[i])
            if(dist < count):
                ind = i
                count = dist
    return ind


def getHistograms(siftVectors, visualWords):
    histograms = {}
    for category, siftVector in siftVectors.items():
        categoryHistograms = []

        for img in siftVector:
            histogram = np.zeros(len(visualWords))
            for feature in img:
                i = find_index(feature, visualWords)
                histogram[i] += 1

            categoryHistograms.append(histogram)

        histograms[category] = categoryHistograms

    return histograms


def getTrainingData(trainingCategories, testingCategories, visualWordsCount):
    (trainingDescriptorList, trainingSiftVectors) = getSiftFeatures(trainingCategories)
    (testingDescriptorList, testingSiftVectors) = getSiftFeatures(testingCategories)
    visualWords = getVisualWords(visualWordsCount, trainingDescriptorList)
    trainingData = getHistograms(trainingSiftVectors, visualWords)
    testingData = getHistograms(testingSiftVectors, visualWords)
    return (trainingData, testingData)

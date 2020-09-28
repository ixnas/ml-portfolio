import cv2
import numpy as np
import os
from sklearn.utils import Bunch
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from scipy.spatial import distance
import matplotlib.pyplot as plt
import json

WIN_NAME = "image"
DIR_NAME = os.path.join(".", "data")

cv2.namedWindow(WIN_NAME)


def denoise(img):
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    eroded = cv2.erode(blurred, None, iterations=2)
    return cv2.dilate(eroded, None, iterations=2)


def toGrayScale(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    return cv2.extractChannel(hsv, 2)


def toBinary(img):
    (threshold, binary) = cv2.threshold(img, 80, 255, cv2.THRESH_BINARY)
    return binary


def getMask(img):
    binaryImage = toBinary(img)
    contour = getLargestContour(binaryImage)
    maskMat = np.zeros((len(img), len(img[0]), 1))
    maskMat = maskMat.astype(np.uint8)
    cv2.fillPoly(maskMat, [contour], [255])
    return maskMat


def preProcess(img):
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    eroded = cv2.erode(blurred, None, iterations=2)
    dilated = cv2.dilate(eroded, None, iterations=2)
    hsv = cv2.cvtColor(dilated, cv2.COLOR_RGB2HSV)
    return cv2.extractChannel(hsv, 2)


def getLargestContour(img):
    contours, hier = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL,
                                      cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    return np.squeeze(contour)


def processImage(img):
    preProcessed = preProcess(img)
    mask = getMask(preProcessed)

    sift = cv2.SIFT_create(10)
    (kp, des) = sift.detectAndCompute(preProcessed, mask)

    return (cv2.drawKeypoints(img, kp, None), des)


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
            (imgFinal, des) = processImage(img)
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
            #count = L1_dist(image, center[i])
        else:
            dist = distance.euclidean(image, center[i])
            #dist = L1_dist(image, center[i])
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


def getTrainingData(categories):
    (descriptorList, shiftVectors) = getSiftFeatures(categories)
    visualWords = getVisualWords(150, descriptorList)
    histograms = getHistograms(shiftVectors, visualWords)
    return histograms


def getTrainedModel(trainingData):
    totalImgCount = 0
    for images in trainingData.values():
        totalImgCount += len(images)

    labelEncoder = preprocessing.LabelEncoder()
    labels = np.empty(totalImgCount, np.str)
    features = np.empty((totalImgCount, 150))

    for category, categoryImages in trainingData.items():
        count = 0
        for i in range(len(categoryImages)):
            labels[i] = category
            features[i] = categoryImages[i]
    
    encodedLabels = labelEncoder.fit_transform(labels)

    model = KNeighborsClassifier(n_neighbors = 1)
    model.fit(features, labels)

    return model


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


def showResults():
    categories = getCategories(DIR_NAME)
    trainingData = getTrainingData(categories)

    """
    for category, categoryImages in categories.items():
        for img in categoryImages:
            (processedImg, des) = processImage(img)
            cv2.imshow(WIN_NAME, processedImg)
            if cv2.waitKey(200) == ord("q"):
                break
    """
    return trainingData

categories = showResults()

"""
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

with open('lol.json', 'w') as outfile:
    json_str = json.dump(showResults(), outfile, cls=NumpyEncoder)

"""
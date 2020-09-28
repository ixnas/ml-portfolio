import cv2
import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from scipy.spatial import distance

import imgreader
import extraction

WIN_NAME = "image"
DIR_NAME_TRAIN = os.path.join(".", "data", "train")
DIR_NAME_TEST = os.path.join(".", "data", "test")
VISUAL_WORDS_COUNT = 150


def getTrainedModel(trainingData, nNeighbors):
    totalImgCount = 0
    for images in trainingData.values():
        totalImgCount += len(images)

    labelEncoder = preprocessing.LabelEncoder()
    labels = np.empty(totalImgCount, np.str)
    features = np.empty((totalImgCount, VISUAL_WORDS_COUNT))

    for category, categoryImages in trainingData.items():
        count = 0
        for i in range(len(categoryImages)):
            labels[i] = category
            features[i] = categoryImages[i]

    encodedLabels = labelEncoder.fit_transform(labels)

    model = KNeighborsClassifier(n_neighbors=nNeighbors)
    model.fit(features, labels)

    return model


def knn(images, tests):
    num_test = 0
    correct_predict = 0
    class_based = {}

    for test_key, test_val in tests.items():
        class_based[test_key] = [0, 0]  # [correct, all]
        for tst in test_val:
            predict_start = 0
            minimum = 0
            key = "a"  # predicted
            for train_key, train_val in images.items():
                for train in train_val:
                    if(predict_start == 0):
                        minimum = distance.euclidean(tst, train)
                        key = train_key
                        predict_start += 1
                    else:
                        dist = distance.euclidean(tst, train)
                        if(dist < minimum):
                            minimum = dist
                            key = train_key

            if(test_key == key):
                correct_predict += 1
                class_based[test_key][0] += 1
            num_test += 1
            class_based[test_key][1] += 1
    return [num_test, correct_predict, class_based]


def showResults():
    print("Loading training set...")
    trainingCategories = imgreader.getCategories(DIR_NAME_TRAIN)

    print("Loading test set...")
    testCategories = imgreader.getCategories(DIR_NAME_TEST)

    print("Extracting features...")
    (trainingData, testingData) = extraction.getTrainingData(
        trainingCategories, testCategories, VISUAL_WORDS_COUNT)

    print("Testing model...")
    testingData = knn(trainingData, testingData)

    print("\nTotal:      " + str(testingData[0]))
    print("Correct:    " + str(testingData[1]))
    percentage = (testingData[1] * 100) / (testingData[0] * 100) * 100
    print("Percentage: " + str(percentage))

    for categoryName, categoryResult in testingData[2].items():
        print("\nCategory:    " + categoryName)
        print("Total:       " + str(categoryResult[1]))
        print("Correct:     " + str(categoryResult[0]))
        percentage = (categoryResult[0] * 100) / \
            (categoryResult[1] * 100) * 100
        print("Percentage:  " + str(percentage))

    return trainingData


data = showResults()

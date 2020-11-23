import cv2
import numpy as np
import os
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from scipy.spatial import distance
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from operator import attrgetter
from collections import Counter

import imgreader
import extraction

WIN_NAME = "image"
DIR_NAME_TRAIN = os.path.join(".", "data", "train")
DIR_NAME_TEST = os.path.join(".", "data", "test")
VISUAL_WORDS_COUNT = 20


def knn(images, tests, k):
	num_test = 0
	correct = 0
	correctByClass = {}
	resultInput = []
	resultExpected = []

	for test_key, test_val in tests.items():
		correctByClass[test_key] = [0, 0]  # [correct, total]
		for tst in test_val:
			neighbours = []
			for train_key, train_val in images.items():
				for train in train_val:
					neighbours.append({"label": train_key, "distance": distance.euclidean(tst, train)})

			neighbours = sorted(neighbours, key=lambda x: x["distance"])
			nearestNeighbours = neighbours[slice(k)]
			nearestNeighbours = [x["label"] for x in nearestNeighbours]
			counter = Counter(nearestNeighbours)
			key, occurences = counter.most_common(1)[0]

			resultInput.append(key)
			resultExpected.append(test_key)

			if (test_key == key):
				correct += 1
				correctByClass[test_key][0] += 1
			num_test += 1
			correctByClass[test_key][1] += 1
	return [num_test, correct, correctByClass, resultInput, resultExpected]


def showResults(visualWordsCount, k):
	startTime = time.time()
	print("Loading training set...")
	trainingCategories = imgreader.getCategories(DIR_NAME_TRAIN)

	print("Loading test set...")
	testCategories = imgreader.getCategories(DIR_NAME_TEST)

	print("Extracting features...")
	(trainingData, testingData) = extraction.getTrainingData(
		trainingCategories, testCategories, visualWordsCount)
	endTime = time.time()
	print("Training time: " + str(endTime - startTime))

	startTime = time.time()
	print("Testing model...")
	testingResults = knn(trainingData, testingData, k)
	endTime = time.time()
	print("Testing time: " + str(endTime - startTime))

	print("\nTotal:      " + str(testingResults[0]))
	print("Correct:    " + str(testingResults[1]))
	percentage = (testingResults[1] * 100) / (testingResults[0] * 100) * 100
	print("Percentage: " + str(percentage))

	"""
	for categoryName, categoryResult in testingResults[2].items():
		print("\nCategory:    " + categoryName)
		print("Total:       " + str(categoryResult[1]))
		print("Correct:     " + str(categoryResult[0]))
		categoryPercentage = (float(categoryResult[0]) * 100) / \
		                     	(float(categoryResult[1]) * 100) * 100
		print("Percentage:  " + str(categoryPercentage))
	"""

	return (testingResults[3], testingResults[4])


kRange = 20
resultInput, resultExpected = showResults(80, kRange)

cm = confusion_matrix(resultExpected, resultInput, normalize="true")
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["hang_loose", "paper", "rock", "scissors"])
disp.plot()
plt.title("Confusion matrix")
plt.show()

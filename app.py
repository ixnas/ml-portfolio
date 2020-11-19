import cv2
import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from scipy.spatial import distance
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import imgreader
import extraction

WIN_NAME = "image"
DIR_NAME_TRAIN = os.path.join(".", "data", "train")
DIR_NAME_TEST = os.path.join(".", "data", "test")
VISUAL_WORDS_COUNT = 20


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
					if (predict_start == 0):
						minimum = distance.euclidean(tst, train)
						key = train_key
						predict_start += 1
					else:
						dist = distance.euclidean(tst, train)
						if (dist < minimum):
							minimum = dist
							key = train_key

			if (test_key == key):
				correct_predict += 1
				class_based[test_key][0] += 1
			num_test += 1
			class_based[test_key][1] += 1
	return [num_test, correct_predict, class_based]


def showResults(visualWordsCount):
	print("Loading training set...")
	trainingCategories = imgreader.getCategories(DIR_NAME_TRAIN)

	print("Loading test set...")
	testCategories = imgreader.getCategories(DIR_NAME_TEST)

	print("Extracting features...")
	(trainingData, testingData) = extraction.getTrainingData(
		trainingCategories, testCategories, visualWordsCount)

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
		categoryPercentage = (float(categoryResult[0]) * 100) / \
		                     (float(categoryResult[1]) * 100) * 100
		print("Percentage:  " + str(categoryPercentage))

	return percentage


x = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
y = []
for i in range(len(x)):
	y.append(showResults(x[i]))

fig, ax = plt.subplots(1, 1)
ax.xaxis.set_major_locator(ticker.MultipleLocator(20))
ax.set_title("Accuracy for different numbers of visual words")
ax.set_xlabel("Number of visual words")
ax.set_ylabel("Accuracy percentage")
plt.plot(x, y)
plt.show()

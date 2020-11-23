import cv2
import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
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

"""
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
"""


def knn(images, tests, k):
	num_test = 0
	correct_predict = 0
	class_based = {}

	for test_key, test_val in tests.items():
		class_based[test_key] = [0, 0]  # [correct, all]
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

			if (test_key == key):
				correct_predict += 1
				class_based[test_key][0] += 1
			num_test += 1
			class_based[test_key][1] += 1
	return [num_test, correct_predict, class_based]


def showResults(visualWordsCount, kRange):
	print("Loading training set...")
	trainingCategories = imgreader.getCategories(DIR_NAME_TRAIN)

	print("Loading test set...")
	testCategories = imgreader.getCategories(DIR_NAME_TEST)

	print("Extracting features...")
	(trainingData, testingData) = extraction.getTrainingData(
		trainingCategories, testCategories, visualWordsCount)

	percentages = []
	for i in range(1, kRange + 1):
		print("K = " + str(i) + " Testing model...")
		testingResults = knn(trainingData, testingData, i)

		print("\nTotal:      " + str(testingResults[0]))
		print("Correct:    " + str(testingResults[1]))
		percentage = (testingResults[1] * 100) / (testingResults[0] * 100) * 100
		print("Percentage: " + str(percentage))

		percentages.append(percentage)
		"""
		for categoryName, categoryResult in testingResults[2].items():
			print("\nCategory:    " + categoryName)
			print("Total:       " + str(categoryResult[1]))
			print("Correct:     " + str(categoryResult[0]))
			categoryPercentage = (float(categoryResult[0]) * 100) / \
		                     	(float(categoryResult[1]) * 100) * 100
			print("Percentage:  " + str(categoryPercentage))
		"""
	return percentages


kRange = 20
x = [i for i in range(1, kRange + 1)]
y = showResults(80, kRange)
# x = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
"""
for i in range(len(x)):
	y.append(showResults(x[i]))
	"""

fig, ax = plt.subplots(1, 1)
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
# ax.set_title("Accuracy for different numbers of visual words")
ax.set_title("Accuracy for different numbers of nearest neighbours at 80 visual words")
# ax.set_xlabel("Number of visual words")
ax.set_xlabel("Number of nearest neighbours")
ax.set_ylabel("Accuracy percentage")
plt.plot(x, y)
plt.show()

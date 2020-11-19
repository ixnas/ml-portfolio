import cv2
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial import distance
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.decomposition import PCA

import processing


def extractSiftDescriptors(img, mask):
	sift = cv2.SIFT_create()
	(kp, des) = sift.detectAndCompute(img, mask)

	return des


def getVisualWords(k, descriptorList):
	kmeans = KMeans(n_clusters=k, n_init=50)
	kmeans.fit(descriptorList)
	# y_km = kmeans.fit_predict(descriptorList)
	visualWords = kmeans.cluster_centers_
	# pca = PCA(n_components=2).fit(visualWords)
	# pca2d = pca.transform(visualWords)
	# plt.scatter(pca2d[:,0], pca2d[:,1])
	# plt.grid()
	# plt.show()

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


def find_index(image, centers):
	shortestDistance = 0
	index = 0
	for i in range(len(centers)):
		if (i == 0):
			shortestDistance = distance.euclidean(image, centers[i])
		else:
			dist = distance.euclidean(image, centers[i])
			if (dist < shortestDistance):
				index = i
				shortestDistance = dist
	return index


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


def showHistograms(data, visualWordsCount):
	for category in data.keys():
		plotData = np.array(data[category])
		means = plotData.mean(0)
		mins = plotData.min(0)
		maxes = plotData.max(0)
		std = plotData.std(0)
		plt.figure()
		fig, ax = plt.subplots(1, 1)
		ax.set_title("Class: " + category)
		ax.errorbar(np.arange(visualWordsCount), means, std, fmt='ok', lw=3)
		ax.errorbar(np.arange(visualWordsCount), means, [means - mins, maxes - means], fmt='.k', ecolor='gray', lw=1)
		# ax.xaxis.(-1, 10)
		ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
		ax.set_ylabel("Number of occurences")
		ax.set_xlabel("Visual word number")
		plt.show()
	return


def getTrainingData(trainingCategories, testingCategories, visualWordsCount):
	(trainingDescriptorList, trainingSiftVectors) = getSiftFeatures(trainingCategories)
	(testingDescriptorList, testingSiftVectors) = getSiftFeatures(testingCategories)
	visualWords = getVisualWords(visualWordsCount, trainingDescriptorList)
	trainingData = getHistograms(trainingSiftVectors, visualWords)
	# showHistograms(trainingData, visualWordsCount)
	testingData = getHistograms(testingSiftVectors, visualWords)
	return (trainingData, testingData)

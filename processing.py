import cv2
import numpy as np


def getDenoisedImage(img):
	blurred = cv2.GaussianBlur(img, (5, 5), 0)
	eroded = cv2.erode(blurred, None, iterations=2)
	return cv2.dilate(eroded, None, iterations=2)


def getGrayScaleImage(img):
	hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
	return cv2.extractChannel(hsv, 2)


def getBinaryImage(img):
	(threshold, binary) = cv2.threshold(img, 80, 255, cv2.THRESH_BINARY)
	return binary


def getLargestContour(img):
	contours, hier = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL,
	                                  cv2.CHAIN_APPROX_SIMPLE)
	contour = max(contours, key=cv2.contourArea)
	return np.squeeze(contour)


def getMask(img):
	binaryImage = getBinaryImage(img)
	# cv2.imshow("img", binaryImage)
	# cv2.waitKey(0)
	contour = getLargestContour(binaryImage)
	maskMat = np.zeros((len(img), len(img[0]), 1))
	maskMat = maskMat.astype(np.uint8)
	cv2.fillPoly(maskMat, [contour], [255])
	# cv2.imshow("img", maskMat)
	# cv2.waitKey(0)
	return maskMat


def getProcessedImage(img):
	# cv2.namedWindow("img")
	# cv2.imshow("img", img)
	# cv2.waitKey(0)
	blurred = cv2.GaussianBlur(img, (5, 5), 0)
	# cv2.imshow("img", blurred)
	# cv2.waitKey(0)
	gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
	# cv2.imshow("img", gray)
	# cv2.waitKey(0)
	return gray

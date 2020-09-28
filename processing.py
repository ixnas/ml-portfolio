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
    contour = getLargestContour(binaryImage)
    maskMat = np.zeros((len(img), len(img[0]), 1))
    maskMat = maskMat.astype(np.uint8)
    cv2.fillPoly(maskMat, [contour], [255])
    return maskMat


def getProcessedImage(img):
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    eroded = cv2.erode(blurred, None, iterations=2)
    dilated = cv2.dilate(eroded, None, iterations=2)
    hsv = cv2.cvtColor(dilated, cv2.COLOR_RGB2HSV)
    return cv2.extractChannel(hsv, 2)
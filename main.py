import os
import cv2
import pickle
import numpy as np
import getBOW
import getCentroids
import getDescriptors
import drawHistograms
import matplotlib.pyplot as plt


def tryCompute(myDetector, myDetectorName, myDictionarySize):
    myDetectorName = str.upper(myDetectorName)
    if myDetectorName not in ("AKAZE", "SURF", "SIFT"):
        print("Unknown " + "\'" + myDetectorName + "\' detector")
        return
    d = getDescriptors.main(myDetector, myDetectorName)
    dConverted = getDescriptors.convert(d)
    c = getCentroids.main(myDetectorName, myDictionarySize, dConverted)
    bow = getBOW.main(myDetectorName, d, c)
    drawHistograms.main(bow, myDetectorName,  myDictionarySize)


tryCompute(cv2.xfeatures2d.SURF_create(), "SURF", 555)
tryCompute(cv2.xfeatures2d.SIFT_create(), "SIFT", 555)

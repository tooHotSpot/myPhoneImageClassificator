import os
import cv2
import glob
import pickle
import numpy as np
import time


def main(detector, detectorName):
    print("Currently in descriptors function")
    d = []
    p = "Descriptors/All" + detectorName + ".txt"
    if not os.path.isfile(p):
        print(p, " does not exist")
        p = "Descriptors/" + detectorName + "/"
        if not os.path.isdir(p):
            print("No files in dir ", p)
            p = "EducationalPhotosIPhoneCutNamed2018-02-25/*/"
            for playerFolder in glob.glob(p):
                print(playerFolder)
                t = playerFolder.split('\\')[1]
                folderImages = playerFolder + '*.jpg'
                playerDescriptors = []
                for image in glob.glob(folderImages):
                    img = cv2.imread(image)
                    kp, imageDescriptors = detector.detectAndCompute(img, None)
                    playerDescriptors.append(imageDescriptors[:500])
                    print(len(playerDescriptors))
                directory = 'Descriptors/' + detectorName
                if not os.path.exists(directory):
                    os.makedirs(directory)
                playerDescriptorsFile = "Descriptors/" + detectorName + "/" + t + ".txt"
                f = open(playerDescriptorsFile, "wb")
                pickle.dump(playerDescriptors, f)
                f.close()
        print("Descriptors already exists but not collected")
        p = "Descriptors/" + detectorName + "/*.txt"  # Getting all txt files in the directory
        for playerDescriptorsFile in glob.glob(p):  # Iterating whole txt files array
            print(playerDescriptorsFile)
            f = open(playerDescriptorsFile, "rb")
            playerDescriptors = pickle.load(f)
            # print("There are ", len(allImagesDescriptors), " images with descriptors")
            for imageDescriptors in playerDescriptors:
                d.append(imageDescriptors)
            f.close()
            f = open("Descriptors/All" + detectorName + ".txt", "wb")
            pickle.dump(d, f)
            f.close()
    else:
        print("Importing collected descriptors from: " + p)
        f = open(p, "rb")
        d = pickle.load(f)
        print(len(d))
        f.close()
    return d


def convert(d):
    print("Unpacking all descriptors to one array: start")
    dConverted = []
    # finally for our task
    for image in d:
        for j in image:
            dConverted.append(j)
    print("Unpacked. Returned array of descriptors")
    return dConverted

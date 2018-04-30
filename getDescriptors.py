import os
import cv2
import glob
import pickle
import numpy as np
import time


def pickle_it(data, path):
    """
    Сохранить данные data в файл path
    :param data: данные, класс, массив объектов
    :param path: путь до итогового файла
    :return:
    """
    with open(path, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def unpickle_it(path):
    """
    Достать данные из pickle файла
    :param path: путь до файла с данными
    :return:
    """
    with open(path, 'rb') as f:
        return pickle.load(f)


def getInOneClick(detector, detectorName, minAmountOfDescriptors, maxAmountOfDescriptors, step, dataFolder):
    myDict = {}
    for i in range(minAmountOfDescriptors, maxAmountOfDescriptors + 5, step):
        myDict[i] = 0

    # Checking of data type TODO
    if len(os.listdir(dataFolder)) == 0:
        print("dataFolder is empty")
        return

    print("Function: getDescriptors\n", "\tParameters: ", "detectorName = ", detectorName,
          "minAmountOfDescriptors = ", minAmountOfDescriptors,
          "maxAmountOfDescriptors = ", maxAmountOfDescriptors,
          "step = ", step,
          "dataFolder = ", dataFolder)

    countProcessedPlayers = 0
    countProcessedImages = 0
    print("Processing new descriptors: ")
    for player in os.listdir(dataFolder):
        print("class (player): ", player)
        playerDataFolder = os.path.join(dataFolder, player)
        for image in os.listdir(playerDataFolder):
            print("\t", image)
            imageForComputation = cv2.imread(os.path.join(playerDataFolder, image))
            kp, imageDescriptors = detector.detectAndCompute(imageForComputation, None)
            initialLength = len(imageDescriptors)
            for i in np.arange(minAmountOfDescriptors, maxAmountOfDescriptors + 5, step):
                amountOfDescriptors = i
                imageDescriptors = imageDescriptors[:min(amountOfDescriptors, len(imageDescriptors))]
                if amountOfDescriptors > initialLength:
                    print("Too big amount of descriptors error:, {}  > {}".format(amountOfDescriptors, initialLength))
                    return
                    break
                playerDescriptorsFolder = os.path.join("descriptors/", detectorName, str(amountOfDescriptors), player)
                if not os.path.exists(playerDescriptorsFolder):
                    os.makedirs(playerDescriptorsFolder)
                imageDescriptorsFile = image + ".txt"
                pickle_it(imageDescriptors, os.path.join(playerDescriptorsFolder, imageDescriptorsFile))
                myDict[amountOfDescriptors] += 1
            countProcessedImages += 1
            break  # remove before super-start
        countProcessedPlayers += 1

    print("Successfully processed {} images of {} players".format(countProcessedImages, countProcessedPlayers))
    print("Amount of photos with definite descriptors extracted:")
    for i in range(minAmountOfDescriptors, maxAmountOfDescriptors, step):
        print("{} : {}".format(i, myDict[i]))


def main():
    detectors = [[cv2.xfeatures2d.SIFT_create(), "SIFT"],
                 [cv2.xfeatures2d.SURF_create(), "SURF"],
                 [cv2.AKAZE_create(), "AKAZE"]]
    for pair in detectors:
        getInOneClick(detector=pair[0], detectorName=pair[1],
                      minAmountOfDescriptors=500, maxAmountOfDescriptors=10000, step=50,
                      dataFolder="train")


main()
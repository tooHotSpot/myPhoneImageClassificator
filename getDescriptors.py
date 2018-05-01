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


def countSave(detector, detectorName, minAmountOfDescriptors, maxAmountOfDescriptors, step, dataFolder):
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

    print("Processing new descriptors: ")
    for player in os.listdir(dataFolder):
        print("class (player): ", player)
        playerDataFolder = os.path.join(dataFolder, player)
        for image in os.listdir(playerDataFolder):
            # print("\t", image)
            imageForComputation = cv2.imread(os.path.join(playerDataFolder, image))
            kp, imageDescriptors = detector.detectAndCompute(imageForComputation, None)
            initialLength = len(imageDescriptors)
            for i in np.arange(minAmountOfDescriptors, maxAmountOfDescriptors + 5, step):
                amountOfDescriptors = i
                playerDescriptorsFolder = os.path.join("descriptors/", detectorName, str(amountOfDescriptors), player)
                if os.path.exists(playerDescriptorsFolder) and len(os.listdir(playerDescriptorsFolder)) == len(
                        os.listdir(playerDataFolder)):
                    # Хотя бы попытаться пересчитать еще раз дескрипторы
                    continue
                elif not os.path.exists(playerDescriptorsFolder):
                    os.makedirs(playerDescriptorsFolder)
                imageDescriptorsFile = image + ".txt"
                imageDescriptors = imageDescriptors[:min(amountOfDescriptors, len(imageDescriptors))]
                if amountOfDescriptors > initialLength:
                    print("Too big amount of descriptors error:, {}  > {} for image {}".format(amountOfDescriptors,
                                                                                               initialLength,
                                                                                               image))
                    break
                pickle_it(imageDescriptors, os.path.join(playerDescriptorsFolder, imageDescriptorsFile))
                myDict[amountOfDescriptors] += 1

    # print("Successfully processed {} images of {} players".format(countProcessedImages, countProcessedPlayers))
    print("Amount of photos with definite descriptors extracted:")
    for i in range(minAmountOfDescriptors, maxAmountOfDescriptors + 5, step):
        print("{} : {}".format(i, myDict[i]))


def processDescriptorsRange(a=500, b=500, c=500):
    detectors = [[cv2.xfeatures2d.SIFT_create(), "SIFT"],
                 [cv2.xfeatures2d.SURF_create(), "SURF"],
                 [cv2.AKAZE_create(), "AKAZE"]]
    t0 = time.time()
    for pair in detectors:
        print(pair[1])
        countSave(detector=pair[0], detectorName=pair[1],
                  minAmountOfDescriptors=a, maxAmountOfDescriptors=b, step=c,
                  dataFolder="train")
        print("{} finished, time = {} ".format(pair[1], (time.time() - t0) // 60))
        t0 = time.time()


def getInOneClick():
    pass


def processDescriptorsDefiniteSize(size):
    processDescriptorsRange(size, size - size // 10, size)


processDescriptorsDefiniteSize(750)

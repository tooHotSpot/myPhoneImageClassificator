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


def evaluate(detector, detectorName, amountOfDescriptors, dataFolder):
    # Checking of data type TODO
    print("Function: getDescriptors\n", "\tParameters: ", "detectorName = ", detectorName,
          "amountOfDescriptors = ", amountOfDescriptors,
          "dataFolder = ", dataFolder)
    if len(os.listdir(dataFolder)) == 0:
        print("dataFolder is empty")
        return

    descriptorsFolder = os.path.join("descriptors/", detectorName, str(amountOfDescriptors))
    if not os.path.exists(descriptorsFolder):
        os.makedirs(descriptorsFolder)

    countProcessed = 0
    if len(os.listdir(descriptorsFolder)) < len(os.listdir(dataFolder)):
        print("Processing new descriptors: ")
        for player in os.listdir(dataFolder):
            print("class: ", player)
            playerDescriptorsFolder = os.path.join(descriptorsFolder, player)
            if not os.path.exists(playerDescriptorsFolder):
                os.makedirs(playerDescriptorsFolder)
            playerDataFolder = os.path.join(dataFolder, player)
            for image in os.listdir(playerDataFolder):
                print("\t", image)
                imageForComputation = cv2.imread(os.path.join(playerDataFolder, image))
                kp, imageDescriptors = detector.detectAndCompute(imageForComputation, None)
                imageDescriptors = imageDescriptors[:min(amountOfDescriptors, len(imageDescriptors))]
                if amountOfDescriptors > len(imageDescriptors):
                    print("Too big amountOfDescriptors value error ", amountOfDescriptors, " > ", len(imageDescriptors))
                imageDescriptorsFile = image + ".txt"
                pickle_it(imageDescriptors, os.path.join(playerDescriptorsFolder, imageDescriptorsFile))
                countProcessed += 1

    print("Successfully processed {} images".format(countProcessed))


evaluate(detector=cv2.xfeatures2d.SIFT_create(),
         detectorName="SIFT",
         amountOfDescriptors=500,
         dataFolder="train")



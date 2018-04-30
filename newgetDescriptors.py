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


def savemyfile(data, datapath):
    f = open(datapath, "wb")
    pickle.dump(data, f)
    f.close()


def main(detector, detectorName, amountOfDescriptors):
    print("Currently in descriptors function")
    p = "Descriptors/New/" + detectorName + "/" + str(amountOfDescriptors) + "/"
    if not os.path.exists(p) or (os.path.exists(p) and len(os.listdir(p))):
        if not os.path.exists(p):
            os.makedirs(p)
        print("Processing new descriptors")
        allPlayersAllImages = []
        playerspath = "EducationalPhotosIPhoneCutNamed2018-02-25/*/"
        for playerFolder in glob.glob(playerspath):
            print("in folder ", playerFolder)
            playerName = playerFolder.split('\\')[1]
            everyPlayerImagesDescriptors = []
            for image in glob.glob(playerFolder + '*.jpg'):
                img = cv2.imread(image)
                kp, imageDescriptors = detector.detectAndCompute(img, None)
                imageDescriptors = imageDescriptors[:min(amountOfDescriptors, len(imageDescriptors))]
                if amountOfDescriptors > len(imageDescriptors):
                    print("Too big amountOfDescriptors value error ", amountOfDescriptors, " > ", len(imageDescriptors))
                imageDescriptorsFile = p + "/" + playerName + "/" + image + ".txt"
                savemyfile(imageDescriptors, imageDescriptorsFile)
                everyPlayerImagesDescriptors.append(imageDescriptors)
            everyPlayerImagesDescriptorsFile = p + "/" + playerName + "/TotalCurrent.txt"
            savemyfile(everyPlayerImagesDescriptors, everyPlayerImagesDescriptorsFile)
            allPlayersAllImages.append(everyPlayerImagesDescriptors)
        savemyfile(allPlayersAllImages, p + "/Total.txt")
    else:
        mainFile = p + "/Total.txt"
        print("Importing collected descriptors from: ", mainFile)
        f = open(mainFile, "rb")
        allPlayersAllImages = pickle.load(f)
        print(len(allPlayersAllImages), " (must be 17 like amount of images)")
        f.close()
    return allPlayersAllImages


def reshaper(detector, detectorName, amountOfDescriptors):
    p = "Descriptors/New/" + detectorName + "/" + str(amountOfDescriptors) + "/"
    mainFile = p + "/Total.txt"
    print("Importing collected descriptors from: ", mainFile)
    f = open(mainFile, "rb")

    allPlayersAllImages = pickle.load(f)
    print(len(allPlayersAllImages), " (must be 17 like amount of images)")
    f.close()

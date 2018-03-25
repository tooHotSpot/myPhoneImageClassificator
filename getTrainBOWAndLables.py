import numpy as np
import os
import glob
import getBOW


def main(allbow=[]):
    p = "EducationalPhotosIPhoneCutNamed2018-02-25/*/"
    StartInBOW = 0
    FinInBOWPlusOne = 0
    labels = []
    trainingBoW = []
    for playerFolder in glob.glob(p):
        t = playerFolder.split('\\')[1]
        folderImages = playerFolder + '*.jpg'
        list = os.listdir(playerFolder)
        number_files = len(list)
        FinInBOWPlusOne += number_files
        myset = np.arange(0, number_files, 1)
        np.random.shuffle(myset)
        training = int(np.around(len(myset) * 0.8))
        myset = sorted(myset[:training])
        myset = [myset[i] + StartInBOW for i in range(len(myset))]
        print("Folder <", t, "> has ", number_files, " files")
        print("chosen ", training, " for training from them:", myset)
        for i in range(len(myset)):
            trainingBoW.append(allbow[myset[i]])
            labels.append(t)
        print(len(trainingBoW), " ", len(labels))
        StartInBOW = FinInBOWPlusOne


def mainAnotherChoosing(allbow=[]):
    p = "EducationalPhotosIPhoneCutNamed2018-02-25/*/"
    StartInBOW = 0
    FinInBOWPlusOne = 0
    trainingLabels = []
    validationLabels = []
    testLabels = []
    trainingBoW = []
    validationBoW = []
    testBoW = []
    for playerFolder in glob.glob(p):
        playername = playerFolder.split('\\')[1]
        folderImages = playerFolder + '*.jpg'
        list = os.listdir(playerFolder)
        number_files = len(list)
        FinInBOWPlusOne += number_files
        myset = np.arange(0, number_files, 1)
        np.random.shuffle(myset)
        # region Choosing indices
        training = np.array(myset[:-4])
        validation = np.array(myset[-4:-2])
        test = np.array(myset[-2:])
        # endregion
        # region Adding StartInBOW as offset
        training = [t + StartInBOW for t in training]
        validation = [v + StartInBOW for v in validation]
        test = [t + StartInBOW for t in test]
        # endregion
        # region Statistics
        print("Folder of player <", playername, "> has ", number_files, " files")
        print("chosen ", len(training), " for training from them:", training)
        print("chosen ", len(validation), " for validation from them:", validation)
        print("chosen ", len(test), " for training from them:", test)
        # endregion
        for t in training:
            trainingBoW.append(allbow[t])
            trainingLabels.append(playername)
        return 
        for v in validation:
            validationBoW.append(allbow[v])
            validationLabels.append(playername)
        for t in test:
            testBoW.append(allbow[t])
            testLabels.append(playername)
        StartInBOW = FinInBOWPlusOne


mainAnotherChoosing()


'''
Folder of player < 1 SHANE SMELTZ > has  25  files
chosen  21  for training from them: [19, 14, 17, 15, 24, 22, 3, 7, 0, 21, 9, 20, 10, 5, 23, 11, 4, 13, 2, 12, 1]
chosen  2  for validation from them: [16, 18]
chosen  2  for training from them: [6, 8]
Traceback (most recent call last):
  File "...", line 80, in <module>
    mainAnotherChoosing()
  File "..." line 68, in mainAnotherChoosing
    trainingBoW.append(allbow[t])
IndexError: list index out of range
'''

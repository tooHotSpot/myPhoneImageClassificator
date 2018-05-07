import numpy as np
import os
import glob
import _pickle as pickle
import getBOW


def provideDivision(myDetectorName, allbow):
    myDetectorName = str.upper(myDetectorName)
    mydir = "DividedDataset/" + myDetectorName + "/"
    if os.path.isdir(mydir):
        print("Division is already made and all files are in ", mydir)
        return
    else:
        os.makedirs(mydir)
    p = "EducationalPhotosIPhoneCutNamed2018-02-25/*/"
    StartInBOW = 0
    FinInBOWPlusOne = 0
    # region Initializing empty lists
    trainingLabels = []
    validationLabels = []
    testLabels = []
    trainingBoW = []
    validationBoW = []
    testBoW = []
    # endregion
    for playerFolder in glob.glob(p):
        playername = str(playerFolder.split('\\')[1]).split(' ')[0]
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
        '''
        print("Folder of player <", playername, "> has ", number_files, " files")
        print("chosen ", len(training), " for training from them:", training)
        print("chosen ", len(validation), " for validation from them:", validation)
        print("chosen ", len(test), " for training from them:", test)
        '''
        # endregion
        for t in training:
            trainingBoW.append(allbow[t])
            trainingLabels.append(playername)
        for v in validation:
            validationBoW.append(allbow[v])
            validationLabels.append(playername)
        for t in test:
            testBoW.append(allbow[t])
            testLabels.append(playername)
        StartInBOW = FinInBOWPlusOne

    # region Saving the data
    number_files = len(trainingBoW)
    myset = np.arange(0, number_files, 1)
    np.random.shuffle(myset)
    shuffledtrainingBoW = []
    shuffledtraininglabels = []
    for i in range(number_files):
        shuffledtrainingBoW.append(trainingBoW[myset[i]])
        shuffledtraininglabels.append(trainingLabels[myset[i]])

    mylist = [shuffledtrainingBoW, shuffledtraininglabels]
    f = open(mydir + "trainingBoWLabels.txt", "wb")
    pickle.dump(mylist, f)
    f.close()

    number_files = len(validationBoW)
    myset = np.arange(0, number_files, 1)
    np.random.shuffle(myset)
    shuffledvalidationBoW = []
    shuffledvalidationlabels = []
    for i in range(number_files):
        shuffledvalidationBoW.append(validationBoW[myset[i]])
        shuffledvalidationlabels.append(validationLabels[myset[i]])
    mylist = [shuffledvalidationBoW, shuffledvalidationlabels]
    f = open(mydir + "validationBoWLabels.txt", "wb")
    pickle.dump(mylist, f)
    f.close()

    number_files = len(testBoW)
    myset = np.arange(0, number_files, 1)
    np.random.shuffle(myset)
    shuffledtestBoW = []
    shuffledtestlabels = []
    for i in range(number_files):
        shuffledtestBoW.append(testBoW[myset[i]])
        shuffledtestlabels.append(testLabels[myset[i]])
    mylist = [shuffledtestBoW, shuffledtestlabels]
    f = open(mydir + "testBoWLabels.txt", "wb")
    pickle.dump(mylist, f)
    f.close()
    # endregion


def getSomeBowAndLabels(myDetectorName, setname):
    myDetectorName = str.upper(myDetectorName)
    setname = str.lower(setname)
    myfile = "DividedDataset/" + myDetectorName + "/" + setname + "BoWLabels.txt"
    f = open(myfile, "rb")
    mylist = pickle.load(f)
    setBoW = mylist[0]
    setLabels = mylist[1]
    f.close()
    return setBoW, setLabels

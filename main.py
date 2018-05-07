import os
import cv2
import pickle
import numpy as np
import getBOW
import getCentroids
import getDescriptors
import time
import CompareBOW
import drawHistograms
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import roc_auc_score
from matplotlib.pyplot import Rectangle


def tryCompute(myDetector, myDetectorName, myDictionarySize):
    # region Prepare to learn model
    myDetectorName = str.upper(myDetectorName)
    if myDetectorName not in ("AKAZE", "SURF", "SIFT"):
        print("Unknown " + "\'" + myDetectorName + "\' detector")
        return
    d = getDescriptors.main(myDetector, myDetectorName)
    dConverted = getDescriptors.convert(d)
    c = getCentroids.main(myDetectorName, myDictionarySize, dConverted)
    bow = getBOW.main(myDetectorName, d, c)
    # drawHistograms.main(bow, myDetectorName,  myDictionarySize)
    # CompareBOW.compare(bow, myDetectorName, myDictionarySize)
    # endregion
    provideDivision(myDetectorName, bow)
    # region Dividing dataset
    trainingBoW, trainingLabels = getSomeBowAndLabels(myDetectorName, "training")
    print("training: ", len(trainingBoW), " labeled by the list of len ", len(trainingLabels))
    validationBoW, validationLabels = getSomeBowAndLabels(myDetectorName, "validation")
    print("validation: ", len(validationBoW), " labeled by the list of len ", len(validationLabels))
    testBoW, testLabels = getSomeBowAndLabels(myDetectorName, "test")
    print("test: ", len(testBoW), " labeled by the list of len ", len(testLabels))
    print("End for ", myDetectorName, "\n************************************\n")
    # endregion
    # region learning model
    if not os.path.exists("myModel"):
        os.makedirs("myModel")
    if os.path.isfile("myModel/myModel.txt"):
        print("myModel file does not exist, creating a model")
        t0 = time.time()
        # model = RandomForestClassifier(n_estimators=10, oob_score=True, random_state=1)
        model = RandomForestClassifier(n_estimators=5000, max_features=10)
        model.n_classes_ = 17

        model.fit(trainingBoW, trainingLabels)
        f = open("myModel/myModel.txt", "wb")
        pickle.dump(model, f)
        f.close()
        del model
        print("Time spent: ", (time.time() - t0) // 60)

    f = open("myModel/myModel.txt", "rb")
    model = pickle.load(f)
    f.close()

    predicted = model.predict(validationBoW)

    return validationLabels, predicted
    # print("AUC-ROC (oob) = ", roc_auc_score(trainingLabels, model.oob_prediction_))
    # print("AUC-ROC (test) = ", roc_auc_score(validationLabels, a))
    # endregion


def plottheplot():
    trueClasses, predictedSURF = tryCompute(cv2.xfeatures2d.SURF_create(), "SURF", 555)
    trueClasses, predictedSIFT = tryCompute(cv2.xfeatures2d.SIFT_create(), "SIFT", 555)

    countSURF = 0
    countSIFT = 0
    for i in range(len(trueClasses)):
        print(trueClasses[i], " ", predictedSURF[i], " ", predictedSIFT[i])

    countSURF = np.sum(trueClasses == predictedSURF)
    countSIFT = np.sum(trueClasses == predictedSIFT)

    print("FINALLY:", countSURF, " ", countSIFT)

    fig, axarr = plt.subplots(2, 1)
    myTitle = "Graph predictions "
    fig.suptitle(myTitle)
    fig = plt.gcf()
    plt.tight_layout(pad=1.5)
    fig.set_size_inches(14, 6, forward=True)
    x = np.arange(len(trueClasses))
    axarr[0].plot(x, trueClasses, 'r-', x, predictedSURF, 'b-')
    axarr[1].plot(x, trueClasses, 'r-', x, predictedSIFT, 'g-')
    truev = Rectangle((0, 0), 1, 1, fc="w", color="red")
    surfv = Rectangle((0, 0), 1, 1, fc="w", color="blue")
    siftv = Rectangle((0, 0), 1, 1, fc="w", color="green")
    axarr[0].legend([truev, surfv], ["True", "SURF"])
    axarr.grid(True)
    axarr[1].legend([truev, siftv], ["True", "SIFT"])
    axarr.grid(True)
    plt.savefig(myTitle + ".png")
    #plt.show()
    print("Hello")


plottheplot()


import pickle
import os
import psutil
import numpy as np
import time
import sys


def main(detectorName, descriptors, centroids):
    dictionarySize = len(centroids)
    p = "BOW/" + detectorName + ".txt"
    t0 = time.time()
    if not os.path.isfile(p):
        print(p, " does not exist, wait some time before processing bags")
        allbow = []
        lessthan500 = 0
        for image in descriptors:
            print("Counting bag of words for image #", len(allbow) + 1, end=": ")
            imagebow = np.zeros(dictionarySize)
            if len(image) < 500:
                lessthan500 += 1
            for desc in image:
                mindist = 100000000
                minindex = 0
                for j in range(555):
                    A = centroids[j] - desc
                    dist = np.sqrt(np.dot(A, A.T))
                    if dist < mindist:
                        mindist = dist
                        minindex = j
                imagebow[minindex] += 1  # Descriptor #desci matches best to class #minindex
            allbow.append(imagebow)
            print(" added, time:", (time.time() - t0) // 60, ";")
        print("\nWarning: desc len less than 500: ", lessthan500)
        directory = "BOW/"
        print("Saving to ", directory, " as ", p)
        if not os.path.exists(directory):
            os.makedirs(directory)
        f = open(p, "wb")
        pickle.dump(allbow, f)
        f.close()
    else:
        print("Importing bags of words from: " + p)
        f = open(p, "rb")
        allbow = pickle.load(f)
        f.close()
    return allbow

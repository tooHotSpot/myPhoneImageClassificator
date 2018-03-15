import os
import cv2
import pickle
import numpy as np


def main(detectorName, dictionarySize, descriptors):
    print("Currently in centroid function")
    p = "Centroids/" + detectorName + ".txt"
    if not os.path.isfile(p):
        descriptors = np.float32(descriptors)
        # Clustering
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.01)
        # Apply KMeans with flag cv2.KMEANS_RANDOM_CENTERS (Just to avoid line break in the code)
        flags = cv2.KMEANS_PP_CENTERS
        # desc is a type32 numpy array of vstacked descriptors
        print("Kmeans has started, wait a minute")
        compactness, labels, centroids = cv2.kmeans(descriptors, dictionarySize, None, criteria, 1, flags)

        # Define criteria = ( type, max_iter = 10 , epsilon = 1.0 )
        # My criteria is such that, whenever 10 iterations of algorithm is ran,
        # or an accuracy of epsilon = 1.0 is reached, stop the algorithm and return the answer.
        directory = "Centroids/"
        if not os.path.exists(directory):
            os.makedirs(directory)
        f = open(p, "wb")
        pickle.dump(centroids, f)
        f.close()
    else:
        print("Importing centroids from: " + p)
        f = open(p, "rb")
        centroids = pickle.load(f)
        f.close()
    return centroids

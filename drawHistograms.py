import numpy as np
import os
import matplotlib.pyplot as plt


def main(bow, detectorName, dictionarySize):
    dir = "BOW/Diagrams/" + detectorName
    if not os.path.exists(dir):
        os.makedirs(dir)
    ymax = np.max(bow)
    for u in range(3):  # len(bow)):
        p = "BOW/Diagrams/" + detectorName + "/" + str(u) + ".png"
        if os.path.isfile(p):
            continue
        currentbow = bow[u]
        bin_positions = list(range(len(currentbow)))
        bins_art = plt.bar(bin_positions, currentbow)
        fig = plt.gcf()
        fig.set_size_inches(9, 6, forward=True)
        plt.xlim(-1, dictionarySize + 5)
        plt.ylim(0, ymax)
        plt.ylabel("Встречаемость")
        plt.xlabel("Классы")
        plt.tight_layout()
        plt.title(detectorName + " BOW " + " for image " + str(u))
        plt.savefig(p)
        plt.clf()
        print("Have drawn for ", u)
    print("Ended to draw diagrams")

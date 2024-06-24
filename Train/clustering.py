# Clustering model training. Normalized RGB images.and


import os
import config
import cv2 as cv

from sklearn.cluster import KMeans
import pickle as pkl

import numpy as np


trainingData = list()

f = open("log.txt", "w")

# Loop over empty images BBDD
cc = 0

for root, dirs, files in os.walk("./Data/BBDD_Clustering", topdown=False):
    for name in files:

        # Read and normalize the image
        rutaIMG = os.path.join(root, name)
        img = cv.imread(rutaIMG)
        img = cv.normalize(img, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
        trainingData.append(img)

        cc+=1
        if cc % 1000 == 0:
            f.write(str(cc) + "\n")

        


# Convert each instance to array
trainingData = np.array(trainingData)

f.write("array \n")
trainingData = trainingData.reshape(cc, 256 * 384 * 3)

f.write("reshape \n")

print(len(trainingData[0]))



print("Training K-Means (" + str(config.NUMBER_OF_CLUSTERS) + " clusters)")

kmeans = KMeans(n_clusters=config.NUMBER_OF_CLUSTERS).fit(trainingData)

print("kmeans entrenado \n")

# Guardamos el modelo
pkl.dump(kmeans, open(config.KMEANS_ROUTE, 'wb')) #Saving the model

print("kmeans guardado \n")

print("kmeans trained")

f.close()
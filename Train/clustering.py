'''
Clustering model training. Normalized RGB images.and
'''



import os
import config
import cv2 as cv

from sklearn.cluster import KMeans
import pickle as pkl

import numpy as np


trainingData = list()


# Loop over empty images BBDD
cc = 0

for root, dirs, files in os.walk(config.EMPTY_DATA, topdown=False):
    for name in files:

        # Read and normalize the image
        rutaIMG = os.path.join(root, name)
        img = cv.imread(rutaIMG)
        img = cv.normalize(img, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
        trainingData.append(img)

        cc+=1

        


# Convert each instance to array
trainingData = np.array(trainingData)

trainingData = trainingData.reshape(cc, 256 * 384 * 3)



print("Training K-Means (" + str(config.NUMBER_OF_CLUSTERS) + " clusters)")

kmeans = KMeans(n_clusters=config.NUMBER_OF_CLUSTERS).fit(trainingData)

print("kmeans entrenado \n")

# Guardamos el modelo
pkl.dump(kmeans, open(config.TRAINED_MODELS_ROUTE + "/kmeans.pkl", 'wb')) #Saving the model

print("kmeans guardado \n")

print("kmeans trained")
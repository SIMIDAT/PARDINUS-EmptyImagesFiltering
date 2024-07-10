'''
Apply KMeans model over a set of images
'''

import config
import utils

import os
import pickle
import shutil
import cv2 as cv

# Loading KMeans model (sklearn 1.2.0)
kmeansModel = pickle.load(open(config.KMEANS_ROUTE, "rb"))
print(len(kmeansModel.cluster_centers_))

# Read data and assign a cluster
contador = 0

# Image folder
directorioImagenes = config.POST_CLUSTERING_DIRECTORY_NAME_ANIMALTEST



# Prepare output directory
if os.path.isdir(directorioImagenes):
    # Remove old equalized images
    print("EXISTE")
    shutil.rmtree(directorioImagenes)

os.mkdir(directorioImagenes)

for i in range(config.NUMBER_OF_CLUSTERS):
    os.mkdir(directorioImagenes + os.sep + str(i))
    os.mkdir(directorioImagenes + os.sep + str(i) + os.sep + "images")

print(config.ANIMAL_TEST_DATA)
for root, dirs, files in os.walk(config.ANIMAL_TEST_DATA, topdown=False):

    for name in files:
        
        contador += 1
        rutaIMG = os.path.join(root, name)

        # Read and prepare image
        originalImg = cv.imread(rutaIMG)
        img = cv.normalize(originalImg, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
        img = img.reshape(256 * 384 * 3)
        img = img.reshape(1, -1)
        
        # Predict cluster
        indiceCluster = kmeansModel.predict(img)[0]

        # Get equalized image
        eqImg = utils.equalizeImage(originalImg)

        # Save equalized image
        writeRoute = directorioImagenes + os.sep + str(indiceCluster) + name
        cv.imwrite(directorioImagenes + os.sep + str(indiceCluster) + os.sep + "images" + os.sep + name, eqImg)
        #print(directorioImagenes + os.sep + str(indiceCluster) + os.sep + "images" + os.sep + name)
        
        
        if contador % 100 == 0:
            print(contador)


        
    

print(contador)

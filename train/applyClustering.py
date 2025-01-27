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
kmeansModel = pickle.load(open(config.TRAINED_MODELS_ROUTE + os.sep + "kmeans.pkl", "rb"))
print(len(kmeansModel.cluster_centers_))

# Read data and assign a cluster
contador = 0

# Image folder
clusteredFolder = [config.IMAGE_FOLDER + "BBDD_Clustered_EmptyTrain", config.IMAGE_FOLDER + "BBDD_Clustered_AnimalTrain"]
trainingFolders = [config.EMPTY_DATA, config.ANIMAL_DATA]


for index, folder in enumerate(clusteredFolder):
#folder = config.POST_CLUSTERING_DIRECTORY_NAME_ANIMALTEST



    # Prepare output directory
    if os.path.isdir(folder):
        # Remove old equalized images
        shutil.rmtree(folder)

    os.mkdir(folder)

    for i in range(config.NUMBER_OF_CLUSTERS):
        os.mkdir(folder + os.sep + str(i))
        os.mkdir(folder + os.sep + str(i) + os.sep + "images")


    for root, dirs, files in os.walk(trainingFolders[index], topdown=False):

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
            cv.imwrite(folder + os.sep + str(indiceCluster) + os.sep + "images" + os.sep + name, eqImg)
            
            
            if contador % 100 == 0:
                print(contador)


            
        

    print(contador)

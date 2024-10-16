'''
Apply KMeans model over a set of images
'''

import config
import utils

import os
import pickle
import shutil
import cv2 as cv





def equalizeImage(img):
    img_yuv = cv.cvtColor(img, cv.COLOR_BGR2YUV)

    # equalize the histogram of the Y channel
    img_yuv[:,:,0] = cv.equalizeHist(img_yuv[:,:,0])

    # convert the YUV image back to RGB format
    img = cv.cvtColor(img_yuv, cv.COLOR_YUV2BGR)

    return img


# Loading KMeans model (sklearn 1.2.0)
kmeansModel = pickle.load(open(config.TRAINED_MODELS_ROUTE + "/kmeans.pkl", "rb"))
print(len(kmeansModel.cluster_centers_))

# Read data and assign a cluster
contador = 0

# Prepare output directory
postClusteringDirectory = config.IMAGE_FOLDER + "BBDD_Clustered_Test"

if os.path.isdir(postClusteringDirectory):
    # Remove old equalized images
    print("EXISTE")
    shutil.rmtree(postClusteringDirectory)

os.mkdir(postClusteringDirectory)

for i in range(config.NUMBER_OF_CLUSTERS):
    os.mkdir(postClusteringDirectory + os.sep + str(i))
    os.mkdir(postClusteringDirectory + os.sep + str(i) + os.sep + "images")


for root, dirs, files in os.walk(config.TEST_IMAGES, topdown=True):
    for name in files:
        
        contador += 1
        rutaIMG = os.path.join(root, name)

        # Read and prepare image
        originalImg = cv.imread(rutaIMG)
        img = cv.normalize(originalImg, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
        img = img.reshape(config.IMG_HEIGHT * config.IMG_WIDTH * 3)
        img = img.reshape(1, -1)
        
        # Predict cluster
        indiceCluster = kmeansModel.predict(img)[0]

        # Get equalized image
        eqImg = equalizeImage(originalImg)

        # Save equalized image
        cv.imwrite(postClusteringDirectory + os.sep + str(indiceCluster) + os.sep + "images" + os.sep + name, eqImg)

        
        if contador % 100 == 0:
            print(contador)


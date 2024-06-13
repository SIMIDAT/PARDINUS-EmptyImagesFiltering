import os
import cv2 as cv
import pickle
import shutil
import config


def equalizeImage(img):
    img_yuv = cv.cvtColor(img, cv.COLOR_BGR2YUV)

    # equalize the histogram of the Y channel
    img_yuv[:,:,0] = cv.equalizeHist(img_yuv[:,:,0])

    # convert the YUV image back to RGB format
    img = cv.cvtColor(img_yuv, cv.COLOR_YUV2BGR)

    return img



# Prepare output directory
if os.path.isdir(config.POST_CLUSTERING_DIRECTORY_NAME):
    # Remove old equalized images
    print("EXISTE")
    #shutil.rmtree(config.POST_CLUSTERING_DIRECTORY_NAME)
else:
    os.mkdir(config.POST_CLUSTERING_DIRECTORY_NAME)


    for i in range(config.NUMBER_OF_CLUSTERS):
        os.mkdir(config.POST_CLUSTERING_DIRECTORY_NAME + os.sep + str(i))
        os.mkdir(config.POST_CLUSTERING_DIRECTORY_NAME + os.sep + str(i) + os.sep + "images")


kmeansModel = pickle.load(open(config.KMEANS_ROUTE, "rb"))
print(len(kmeansModel.cluster_centers_))

contador = 0
for root, dirs, files in os.walk(config.EMPTY_DATA, topdown=False):
    for name in files:

        contador += 1
        # Obtenemos la ruta de la imagen
        rutaIMG = os.path.join(root, name)

        # Leemos
        originalImg = cv.imread(rutaIMG)

        # Normalizamos la imagen y preparamos para predict
        img = cv.normalize(originalImg, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
        img = img.reshape(256 * 384 * 3)
        img = img.reshape(1, -1)

        # Hacemos tremendo de predict
        indiceCluster = kmeansModel.predict(img)[0]

        # Hacemos tremenda de ecualización
        eqImg = equalizeImage(originalImg)

        # Save equalized image
        writeRoute = config.POST_CLUSTERING_DIRECTORY_NAME + os.sep + str(indiceCluster) + name
        cv.imwrite(config.POST_CLUSTERING_DIRECTORY_NAME + os.sep + str(indiceCluster) + os.sep + "images" + os.sep + name, eqImg)

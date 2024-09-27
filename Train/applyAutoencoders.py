'''
Apply RAE models over a set of clustered and equalized images to generate reconstruction-error file.
'''


import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from RobustAutoencoder import *
import config
import os

import shutil

# Error metrics
mse = tf.keras.losses.MeanSquaredError()
mae = tf.keras.losses.MeanAbsoluteError()
from skimage.metrics import structural_similarity as ssim

# Functions to calculate reconstruction errors
def calcularMSE(original, reconstruccion):
    return mse(original, reconstruccion).numpy()

def calcularMAE(original, reconstruccion):
    return mae(original, reconstruccion).numpy()

def calcularSSIM(original, reconstruccion):
    return ssim(original, reconstruccion, data_range=1, channel_axis=2)



def errorCalculation(original, reconstruccion, width, height, blockWidth, blockHeight):

    listaErrores = []

    # Step 1: Split the image
    M = int(width / blockWidth)
    N = int(height / blockHeight)

    tilesOriginal = [original[x:x+M,y:y+N] for x in range(0,original.shape[0],M) for y in range(0,original.shape[1],N)]
    tilesReconstruccion = [reconstruccion[x:x+M,y:y+N] for x in range(0,reconstruccion.shape[0],M) for y in range(0,reconstruccion.shape[1],N)]

    # Step 2: For each block, calculate errors
    for i, bloqueOriginal in enumerate(tilesOriginal):
        bloqueReconstruccion = tilesReconstruccion[i]

        valorMse = calcularMSE(bloqueOriginal, bloqueReconstruccion)
        valorMae = calcularMAE(bloqueOriginal, bloqueReconstruccion)
        valorSsim = calcularSSIM(bloqueOriginal, bloqueReconstruccion)

        listaErrores.append(valorMse)
        listaErrores.append(valorMae)
        listaErrores.append(valorSsim)

    # Step 3: Return error list [E1_MSE, E1_MAE, E1_SSIM, E2_MSE, E2_MAE, E2_SSIM...]
    return listaErrores



# Save and store reconstruction error results
def saveResults(results, numberOfCluster, emptyOrNot):

    if emptyOrNot == 0:  # Empty
        fichero = open(config.ERROR_FILES_ROUTE + "Train_Error_" + str(numberOfCluster) + "_Empty.txt", "w")
    else:                # Not empty
        fichero = open(config.ERROR_FILES_ROUTE + "Train_Error_" + str(numberOfCluster) + "_Animal.txt", "w")

    for resultado in results:

        for i in range(len(resultado)):
            fichero.write(str(resultado[i]) + ",")

        fichero.write("\n")

    fichero.close()



# Test RAE models and saves the error file
def main():
    numberOfClusters = config.NUMBER_OF_CLUSTERS

    # Size of the block
    blockWidth = config.BLOCK_WIDTH
    blockHeight = config.BLOCK_HEIGHT

    # Image features
    IMG_WIDTH = config.IMG_WIDTH
    IMG_HEIGHT = config.IMG_HEIGHT
    input_shape = config.INPUT_SHAPE


    # Depending if we want to create a error file of training images or testing images, we use different workflows

    # Train workflow
    clusteredFolder = [config.IMAGE_FOLDER + "BBDD_Clustered_EmptyTrain", config.IMAGE_FOLDER + "BBDD_Clustered_AnimalTrain"]
    
    for emptyOrNot, folder in enumerate(clusteredFolder):


        results = list()

        # For each cluster...
        for clusterIndex in range(numberOfClusters):

            # Load RAE model
            AEname = config.TRAINED_MODELS_ROUTE + "RAE_" + str(numberOfClusters) + "CL_" + str(clusterIndex) + ".h5"
            input_shape = config.INPUT_SHAPE
            model = getRobustAE(input_shape)
            model.load_weights(AEname)

            print('\n\n\n SCANNING CLUSTER ' + str(clusterIndex))
            print('Model: ' + AEname)


            # Process the empty training dataset
            train_dataset = ImageDataGenerator(rescale=1./255, data_format='channels_last')

            train_generator = train_dataset.flow_from_directory(
                folder + os.sep + str(clusterIndex),
                target_size = (IMG_HEIGHT, IMG_WIDTH),
                batch_size=1,
                class_mode=None,
                shuffle=False,
                seed=1491
            )

            contador = 0

            # While there are images in the generator...
            while contador < train_generator.n:

                original = train_generator.next()
                prediccion = model.predict(original)

                # Calcutate errors.

                errores = errorCalculation(original[0], prediccion[0], IMG_WIDTH, IMG_HEIGHT, blockWidth, blockHeight)

                errores.append(clusterIndex)
                errores.append(emptyOrNot)  # Label empty (0) or animal (1)

                # Save results
                results.append(errores)

                contador += 1

            
            print("Cluster ", clusterIndex, " (empty images): ", contador)



        # Save results to file [msi-mae-ssim-cluster-tag]
        saveResults(results, numberOfClusters, emptyOrNot)



    


if __name__ == '__main__':
    main()






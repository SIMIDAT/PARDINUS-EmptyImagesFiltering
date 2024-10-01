'''
Reconstruction of equalized images and creation of errors file
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

    # Step 3: Return error list [E11, E12, E13, E21, E22, E23...]
    return listaErrores


# Save reconstruction error results on disk
def saveResults(results, filepath):

    # # Prepare output directory
    # if os.path.isdir(config.ERRORS_DIRECTORY):
    #     # Remove old error files
    #     shutil.rmtree(config.ERRORS_DIRECTORY)

    # os.mkdir(config.ERRORS_DIRECTORY)
    # TODO
    fichero = open(config.ERRORS_DIRECTORY + os.sep + "Test_Name_Errors_7_RAE_VACIO.txt", "w")

    for indiceImagen, resultado in enumerate(results):

        for i in range(len(resultado)):
            fichero.write(str(resultado[i]) + ",")

        fichero.write(filepath[indiceImagen])

        fichero.write("\n")

    fichero.close()


### MAIN ###
# TODO
# testFolder = config.IMAGE_FOLDER + "BBDD_Clustered_Test" + os.sep
testFolder = "./Data/BBDD_Clustered_EmptyTest/"
results = list()
filepath = list()

numberOfClusters = config.NUMBER_OF_CLUSTERS

# For each cluster of images
for clusterIndex in range(numberOfClusters):

    # TODO
    AEName = config.TRAINED_MODELS_ROUTE + "RAE_Reentreno_" + str(numberOfClusters) + "CL_" + str(clusterIndex) + ".h5"

    print('\n\n\n SCANNING CLUSTER ' + str(clusterIndex))
    print('Model: ' + AEName)

    input_shape = config.INPUT_SHAPE
    model = getRobustAE(input_shape)
    model.load_weights(AEName)

    # Process the test dataset
    test_dataset = ImageDataGenerator(rescale=1./255, data_format='channels_last')

    test_generator = test_dataset.flow_from_directory(
        testFolder + str(clusterIndex),
        target_size = (config.IMG_HEIGHT, config.IMG_WIDTH),
        batch_size=1,
        class_mode=None,
        shuffle=False,
        seed=config.SEED
    )

    contador = 0


    # Nombres de archivos
    filepath_carpeta = test_generator.filenames
    #print(filepath_carpeta)
    

    for file in filepath_carpeta:
        file = file.split(os.sep)
        file = file[1]
        filepath.append(file)




    # While there are images in the generator...
    while contador < test_generator.n:

        original = test_generator.next()
        prediccion = model.predict(original)

        

        # Calcutate errors.

        errores = errorCalculation(original[0], prediccion[0], config.IMG_WIDTH, config.IMG_HEIGHT, config.BLOCK_WIDTH, config.BLOCK_HEIGHT)

        errores.append(clusterIndex)

        # Save results
        results.append(errores)

        contador += 1

    
    print("Cluster ", clusterIndex, " (test images): ", contador)

    print(len(results), len(filepath))

    

# Save results to file [mse-mae-ssim-cluster-tag]
saveResults(results, filepath)

print("FIN")



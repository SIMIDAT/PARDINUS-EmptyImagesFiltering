from RobustAutoencoder import *
import config
import utils


import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Error metrics
mse = tf.keras.losses.MeanSquaredError()
mae = tf.keras.losses.MeanAbsoluteError()
from skimage.metrics import structural_similarity as ssim


def calcularMSE(original, reconstruccion):
    return mse(original, reconstruccion).numpy()

def calcularMAE(original, reconstruccion):
    return mae(original, reconstruccion).numpy()

def calcularSSIM(original, reconstruccion):
    return ssim(original, reconstruccion, data_range=1, channel_axis=2)


# Calculate the difference between original and reconstructed images
def errorCalculation(original, reconstruccion, ancho, alto, blockWidth, blockHeight):

    listaErrores = []

    # Step 1: Split the image
    M = int(ancho / blockWidth)
    N = int(alto / blockHeight)

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
def saveResults(results, trainTest, numberOfCluster):
    fichero = open("./ErrorFiles/" + trainTest + "_Errors_" + str(numberOfCluster) + "_RAE.txt", "w")

    for resultado in results:

        for i in range(len(resultado)):
            fichero.write(str(resultado[i]) + ",")

        fichero.write("\n")

    fichero.close()


# Test RAE models and saves the error file
def main():
    numberOfClusters = config.NUMBER_OF_CLUSTERS

    # Mode
    trainTest = config.TRAINTEST

    # Size of the block
    blockWidth = config.BLOCK_WIDTH
    blockHeight = config.BLOCK_HEIGHT

    # Image features
    IMG_WIDTH = config.IMG_WIDTH
    IMG_HEIGHT = config.IMG_HEIGHT
    input_shape = config.INPUT_SHAPE


    # Depending if we are training or testing, there are different workflows

    # Train workflow
    if trainTest == "Train":
        animalFolder = config.ANIMAL_EQUALIZED_TRAINING_IMAGES_PATH
        emptyFolder = config.EMPTY_EQUALIZED_TRAINING_IMAGES_PATH


        results = list()

        # For each cluster...
        for clusterIndex in range(numberOfClusters):
            AEname = utils.getAEName(numberOfClusters, clusterIndex)

            print('\n\n\n SCANNING CLUSTER ' + str(clusterIndex))
            print('Model: ' + AEname)

            # Load RAE model
            model = utils.getRAEModel(input_shape)
            model.load_weights(AEname + ".h5")

            # Process the empty training dataset
            train_empty_dataset = ImageDataGenerator(rescale=1./255, data_format='channels_last')

            train_empty_generator = train_empty_dataset.flow_from_directory(
                emptyFolder + str(clusterIndex),
                target_size = (IMG_HEIGHT, IMG_WIDTH),
                batch_size=1,
                class_mode=None,
                shuffle=False,
                seed=config.SEED
            )

            contador = 0

            # While there are images in the generator...
            while contador < train_empty_generator.n:

                original = train_empty_generator.next()
                prediccion = model.predict(original)

                # Calcutate errors.

                errores = errorCalculation(original[0], prediccion[0], IMG_WIDTH, IMG_HEIGHT, blockWidth, blockHeight)

                tag_real = 0

                errores.append(clusterIndex)
                errores.append(tag_real)

                # Save results
                results.append(errores)

                contador += 1

            
            print("Cluster ", clusterIndex, " (empty images): ", contador)





            # Process the animal training dataset
            train_animal_dataset = ImageDataGenerator(rescale=1./255, data_format='channels_last')

            train_animal_generator = train_animal_dataset.flow_from_directory(
                animalFolder + str(clusterIndex),
                target_size = (IMG_HEIGHT, IMG_WIDTH),
                batch_size=1,
                class_mode=None,
                shuffle=False,
                seed=config.SEED
            )

            contador = 0

            # While there are images in the generator...
            while contador < train_animal_generator.n:

                original = train_animal_generator.next()
                prediccion = model.predict(original)

                # Calcutate errors.

                errores = errorCalculation(original[0], prediccion[0], IMG_WIDTH, IMG_HEIGHT, blockWidth, blockHeight)

                tag_real = 1

                errores.append(clusterIndex)
                errores.append(tag_real)

                # Save results
                results.append(errores)

                contador += 1

            
            print("Cluster ", clusterIndex, " (animal images): ", contador)

        # Save results to file [msi-mae-ssim-cluster-tag]
        saveResults(results, trainTest, numberOfClusters)



    # Test workflow
    else:
        testFolder = config.EQUALIZED_TEST_IMAGES_PATH


        results = list()

        # For each cluster...
        for clusterIndex in range(numberOfClusters):
            AEname = utils.getRAEName(numberOfClusters, clusterIndex)

            print('\n\n\n SCANNING CLUSTER ' + str(clusterIndex))
            print('Model: ' + AEname)

            # Load pretrained model
            model = utils.getRAEModel(input_shape)
            model.load_weights(AEname + ".h5")

            # Process the test dataset
            test_dataset = ImageDataGenerator(rescale=1./255, data_format='channels_last')

            test_generator = test_dataset.flow_from_directory(
                testFolder + str(clusterIndex),
                target_size = (IMG_HEIGHT, IMG_WIDTH),
                batch_size=1,
                class_mode=None,
                shuffle=False,
                seed=config.SEED
            )

            contador = 0

            # While there are images in the generator...
            while contador < test_generator.n:

                original = test_generator.next()
                prediccion = model.predict(original)

                # Calcutate errors.

                errores = errorCalculation(original[0], prediccion[0], IMG_WIDTH, IMG_HEIGHT, blockWidth, blockHeight)

                errores.append(clusterIndex)

                # Save results
                results.append(errores)

                contador += 1

            
            print("Cluster ", clusterIndex, " (test images): ", contador)
            

        # Save results to file [msi-mae-ssim-cluster-tag]
        saveResults(results, trainTest, numberOfClusters)


if __name__ == '__main__':
    main()



'''
Train RAEs using empty clustered training data. One RAE for each cluster of images
'''

import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow import keras

import config
from RobustAutoencoder import *

import utils

import matplotlib.pyplot as plt




trainFolder = config.IMAGE_FOLDER + "BBDD_Clustered_EmptyTrain"
input_shape = config.INPUT_SHAPE



# Training parameters
batch_size = config.BATCH_SIZE
epoch = config.EPOCHS
verbose = config.VERBOSE
seed = config.SEED

numberOfClusters = config.NUMBER_OF_CLUSTERS

# Check GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
print("-----------------------------------------\n")
print("Number of available GPUs: ", len(gpus))
print("-----------------------------------------\n")

print("PARAMETERS")
print(batch_size, epoch)
print(trainFolder)




# For each cluster, train a RAE
for clusterIndex in range(numberOfClusters):

    print("\n\n\nChecking...")
    print(trainFolder + os.sep + str(clusterIndex))

    print("Generating training and validation datasets...")

    # Read datasets (20% validation set)
    trainDataset = ImageDataGenerator(rescale=1./255, data_format='channels_last', validation_split=0.2)

    trainGenerator = trainDataset.flow_from_directory(
        trainFolder + os.sep + str(clusterIndex),
        target_size = (config.IMG_HEIGHT, config.IMG_WIDTH),
        batch_size=batch_size,
        class_mode='input',
        shuffle=True,
        seed=seed,
        subset='training'
    )

    validationGenerator = trainDataset.flow_from_directory(
        trainFolder + os.sep + str(clusterIndex),
        target_size = (config.IMG_HEIGHT, config.IMG_WIDTH),
        batch_size=batch_size,
        class_mode='input',
        shuffle=True,
        seed=seed,
        subset='validation'
    )

    # Creation and compilation of the model
    RAEName = "RAE_" + str(numberOfClusters) + "CL_" + str(clusterIndex)

    print("Processing RAE: " + RAEName)

    model = getRobustAE(input_shape)
    model.compile(
        optimizer='adam',
        loss=utils.correntropy,
        metrics=['mse']
    )

    # Fit
    history = model.fit(
        trainGenerator,
        validation_data = validationGenerator,
        epochs = 75, # TODO
        verbose = verbose,
        steps_per_epoch=trainGenerator.samples // batch_size,  ##########
        validation_steps = validationGenerator.samples // batch_size  ##########
    )

    model.save_weights(config.TRAINED_MODELS_ROUTE + RAEName + ".h5")

    # Just in case you want to save loss graphics during training (MSE and Correntropy), uncomment this
    '''
    print(history.history.keys())


    plt.plot(history.history['mse'])
    plt.plot(history.history['val_mse'])
    plt.title('MSE loss')
    plt.ylabel('Mse')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    plt.savefig("./mse_" + str(clusterIndex) + "_REDUCEMEAN.pdf")
    plt.savefig("./mse_" + str(clusterIndex) + "_REDUCEMEAN.png")
    plt.clf()


    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Correntropy loss')
    plt.ylabel('Correntropy loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    plt.savefig("./loss_" + str(clusterIndex) + "_REDUCEMEAN.pdf")
    plt.savefig("./loss_" + str(clusterIndex) + "_REDUCEMEAN.png")

    '''



    







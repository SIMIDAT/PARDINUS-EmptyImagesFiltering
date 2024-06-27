import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow import keras

import config
from RobustAutoencoder import *

import utils

import matplotlib.pyplot as plt




trainFolder = config.POST_CLUSTERING_DIRECTORY_NAME
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


# For each cluster
#for clusterIndex in range(numberOfClusters):



# TODO: Modificar para cada ejecución
clusterIndex = 3

print("\n\n\nChecking...")
print(trainFolder + str(clusterIndex))

print("Generating training and validation datasets...")

# Read datasets
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
RAEName = "RAE_Reentreno_" + str(numberOfClusters) + "CL_" + str(clusterIndex)

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
    epochs = 5, # TODO
    verbose = verbose
)

print(history.history.keys())


plt.plot(history.history['mse'])
plt.plot(history.history['val_mse'])
plt.title('MSE loss')
plt.ylabel('Mse')
plt.xlabel('Epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
plt.savefig("./Graficas/mse_" + str(clusterIndex) + ".pdf")
plt.savefig("./Graficas/mse_" + str(clusterIndex) + ".png")
plt.clf()


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Correntropy loss')
plt.ylabel('Correntropy loss')
plt.xlabel('Epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
plt.savefig("./Graficas/loss_" + str(clusterIndex) + ".pdf")
plt.savefig("./Graficas/loss_" + str(clusterIndex) + ".png")



model.save_weights("./TrainedModels/" + RAEName + ".h5")







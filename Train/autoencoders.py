import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras

import config
from RobustAutoencoder import *




trainFolder = config.POST_CLUSTERING_DIRECTORY_NAME
input_shape = config.INPUT_SHAPE

model = getRobustAE(input_shape)

# Training parameters
batch_size = config.BATCH_SIZE
epoch = config.EPOCHS
verbose = config.VERBOSE
seed = config.SEED

numberOfClusters = config.NUMBER_OF_CLUSTERS







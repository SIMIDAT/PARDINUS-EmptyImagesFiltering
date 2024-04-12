import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K

from tensorflow.keras import layers

from RobustAutoencoder import *


# Correntropy loss function used for Robust Autoencoder
tf_2pi = tf.constant(tf.sqrt(2*np.pi), dtype=tf.float32)

def robust_kernel(alpha, sigma = 0.2):
    return 1 / (tf_2pi * sigma) * K.exp(-1 * K.square(alpha) / (2 * sigma * sigma))

def correntropy(y_true, y_pred):
    return -1 * K.sum(robust_kernel(y_pred - y_true))



# Check number of available GPUs
def checkGPU():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print("-----------------------------------------\n")
    print("Number of available GPUs: ", len(gpus))
    print("-----------------------------------------\n")


# Get PARDINUS RAE model
def getRAEModel(input_shape):
    model = getRobustAE(input_shape)
    return model

# Get PARDINUS RAE name (folders names)
def getRAEName(numberOfClusters, clusterIndex):
    nameAE = './TrainedModels/RAE/' + str(numberOfClusters) + 'GR_RAE_cluster' + str(clusterIndex)
    return nameAE
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow import keras

import config
from RobustAutoencoder import *

import utils

import matplotlib.pyplot as plt

input_shape = config.INPUT_SHAPE

RAEName = "RAE_Reentreno_7CL_0"

print("Processing RAE: " + RAEName)

model = getRobustAE(input_shape)
model.compile(
    optimizer='adam',
    loss=utils.correntropy,
    metrics=['mse']
)

model.summary()
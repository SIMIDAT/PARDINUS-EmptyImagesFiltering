

from RobustAutoencoder import *
import config

input_shape = config.INPUT_SHAPE
model = getRobustAE(input_shape)

model.summary()
'''
Reconstruction of equalized images and creation of errors file
'''

from RobustAutoencoder import *
import config
import os


input_shape = config.INPUT_SHAPE
modelo = getRobustAE(input_shape)

print("FIN")



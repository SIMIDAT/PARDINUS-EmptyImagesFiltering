import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K

import cv2 as cv


print("\n\n\n\n\n\n____________REDUCE_MEAN___________\n\n\n\n\n\n")

# Correntropy loss function used for Robust Autoencoder
tf_2pi = tf.constant(tf.sqrt(2*np.pi), dtype=tf.float32)

def robust_kernel(alpha, sigma = 0.2):
    return 1 / (tf_2pi * sigma) * K.exp(-1 * K.square(alpha) / (2 * sigma * sigma))

def correntropy(y_true, y_pred):
    # return -1 * K.sum(robust_kernel(y_pred - y_true))  ## Clasico
    return -1 * tf.reduce_mean(robust_kernel(y_pred - y_true))  # tf.reduce_mean
    




def equalizeImage(img):
    img_yuv = cv.cvtColor(img, cv.COLOR_BGR2YUV)

    # equalize the histogram of the Y channel
    img_yuv[:,:,0] = cv.equalizeHist(img_yuv[:,:,0])

    # convert the YUV image back to RGB format
    img = cv.cvtColor(img_yuv, cv.COLOR_YUV2BGR)

    return img
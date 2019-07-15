# This file shows how to save a keras model to tensorflow pb file 
# and how to use tensorflow to reload the model and inferece by the model 

from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
import numpy as np
import keras
from keras import backend as K
import matplotlib.pylab as plt
from skimage.transform import rescale, resize
from skimage.io import imread

# parameter ==========================
wkdir = '.'
pb_filename = 'tf.pb'
keras_model = 'weights.39-0.56.hdf5'

# build a keras model ================
model = keras.models.load_model(keras_model)
K.set_learning_phase(0) # 0 testing, 1 training mode


input_dim = (384, 512, 3)

img = resize(imread('0000008.jpg'), input_dim,
                           mode='reflect', anti_aliasing=True)

lr = rescale(img, 1 / 4, mode='reflect',
                                        multichannel=True, anti_aliasing=True)

y = model.predict([img[np.newaxis,...], lr[np.newaxis,...]])

plt.imshow(y[0,...,-1])
plt.show()
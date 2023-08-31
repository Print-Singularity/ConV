# -*- coding: utf-8 -*-
"""
Created on Sun Aug 27 11:13:56 2023

@author: rungs
"""

import numpy as np
import keras
import tensorflow as tf
from keras.utils.data_utils import get_file
from tensorflow.python.util.tf_export import keras_export
import matplotlib.pyplot as plt
import warnings
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import Sequential
import tensorflow.keras.layers
from keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.python.keras.layers import Input, Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.utils import plot_model
import time
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.image import rgb_to_grayscale
from PIL import Image, ImageOps



tf.keras.datasets.mnist.load_data(path="mnist.npz")
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data() #spil"t da"ta



# Reshape the input data to (num_samples, height, width, num_channels)
x_train = x_train.reshape((60000, 28, 28, 1))
x_test = x_test.reshape((10000, 28, 28, 1))

#x_train, x_test = np.expand_dims(x_train, -1), np.expand_dims(x_test, -1)   #np.expand_dim Create 3D array

#Check class 
num_class = len(np.unique(y_train))
num_class

#Nomalizeation 0-1
x_train = x_train.astype('float32') /255
x_test = x_test.astype('float32') /255


x_train = np.pad(x_train,((0,0),(2,2),(2,2),(0,0)), 'constant')
x_test = np.pad(x_test,((0,0),(2,2),(2,2),(0,0)), 'constant')

in_shape = x_train.shape[1:]


model = Sequential()

#ConV2D (filter = 6) (kernel_size = 5) (pooling =1)
model.add(Conv2D(6,(5,5), activation='relu', input_shape=in_shape))
model.add(AveragePooling2D(2,2))

#ConV2D (filter = 16) (kernel_size = 5) (pooling =1)
model.add(Conv2D(16,(5,5), activation='relu'))
model.add(AveragePooling2D(2,2))

model.add(Conv2D(120,(5,5), activation='relu'))

model.add(Flatten())
model.add(Dense(84, activation='relu'))

model.add(Dense(num_class, activation='softmax'))       #Dense=10


model.summary()


plot_model(model, 'digit.png', show_shapes=True, show_layer_names=True)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
import time
start = time.time()


#Train Model
history = model.fit(x_train, y_train, epochs=20, batch_size=128 ,verbose=1,validation_data=(x_test,y_test))


y_pred = model.predict(x_test) #predict x_test
y_pred_cls = np.argmax(y_pred, axis=1) #find max each row

idx_miss = np.where(y_pred_cls != y_test) #check which index y_pred_cls != y_test
print(len(idx_miss[0])) #show sum y_pred_cls != y_test
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 13:23:11 2023

@author: rungs
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Aug 27 11:46:37 2023

@author: rungs
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D,Dense, Dropout, Flatten
from tensorflow.python.keras.layers import Input, Dense
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


train_dir = 'D:/AI/Convolution/train'
test_dir = 'D:/AI/Convolution/test'


target_image_shape = (64,64)


train_datagen = ImageDataGenerator(rescale = 1./255,    
                                   rotation_range = 20,   #มุมหมุน+ - 20
                                   height_shift_range=0.15,     #ระยะเลื่อนเเนวดิ่ง
                                   width_shift_range=0.15,     #ระยะเลื่อนเเนวนอน
                                   shear_range=0.9,      #การเฉือน
                                   zoom_range=0.2,      #ค่าซูม
                                   horizontal_flip=True,    #พลิกเเนวนอน
                                   fill_mode='nearest')         # โหมดเติม pixel

#target_size = reshape Pic to 64*64,   class_mode= defind class     batch_size = defind input data 
train_set =  train_datagen.flow_from_directory(train_dir, target_size=target_image_shape,
                                               batch_size=32, class_mode='binary')



val_datagen = ImageDataGenerator(rescale=1./255)
val_set = val_datagen.flow_from_directory(test_dir, target_size=target_image_shape,
                                          batch_size=32, class_mode='binary')

ids, counts = np.unique(train_set.classes, return_counts=True)
print(ids, counts)


train_set[0][0][0].shape

plt.imshow(train_set[0][0][0])
plt.show()

in_shape = (target_image_shape[0],target_image_shape[1],3)


model=Sequential()
model.add(Conv2D(32,(3,3),activation='relu',input_shape=in_shape))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(128,(3,3),activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Flatten())

model.add(Dense(128,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

model.summary()

start = time.time()
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy', 'val_loss', 'val_accuracy'])

history=model.fit(train_set,steps_per_epoch=len(train_set),validation_data=val_set
                            ,epochs=20 , verbose=1)

#TIME               
end = time.time()
times = end - start
s = int(times%60)
m = int(times/60) 
print("Time Taken {} minutes {} second" .format(m,s ))




plt.Figure(figsize=(10,5))
plt.subplot(1, 2,1)
plt.title("Loss")
plt.plot(history.history['loss'], label='Train_Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()


plt.Figure(figsize=(10,3.5))
plt.subplot(1, 2,1)
plt.title("Accuracy")
plt.plot(history.history['accuracy'],'r',lw=3.2, label='Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend()

history.history.keys()
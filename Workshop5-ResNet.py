# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 17:08:54 2023

@author: rungs
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.preprocessing.image import load_img, img_to_array

model = ResNet50(weights='imagenet')

model.summary()


image = load_img('D:/AI/Convolution/train/dog/ดาวน์โหลด.jpg', target_size=(224,224))
plt.imshow(image)


image1 = img_to_array(image)
print("Max is",np.max(image1), "Min is",np.min(image1))

image2 = np.expand_dims(image1 , axis=0)

image3= preprocess_input(image2)
plt.imshow(image3[0])
print("Max is",np.max(image3), "Min is",np.min(image3))

image3.shape

pred = model.predict(image3)
#pred = np.argmax(pred)
pred_decode = decode_predictions(pred,top=3)[0]

print("Rank 1 is",pred_decode[0][1], "Percent accuracy is ",int(pred_decode[0][2]*100))
print("Rank 1 is",pred_decode[1][1], "Percent accuracy is ",int(pred_decode[1][2]*100))
print("Rank 1 is",pred_decode[2][1], "Percent accuracy is ",pred_decode[2][2]*100)

plt.imshow(image3[0])
plt.xlabel(pred_decode[0][1])

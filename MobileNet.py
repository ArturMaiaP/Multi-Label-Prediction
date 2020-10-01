"""
Created on Mon Oct  7 16:24:24 2019

@author: artur
"""
from tensorflow.keras import metrics

from tensorflow.keras.layers import MaxPooling2D, GlobalAveragePooling2D, Conv2D, Flatten, Dense, Input, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers, regularizers

from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from tensorflow.keras.applications.inception_v3 import preprocess_input
[from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input]


import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

INP = 224
BATCHSIZE = 128

# --------------  MobileNetV2 ------------
model_mobileNetV2 = MobileNetV2(include_top=False,
                        weights='imagenet',
                        input_tensor=None,
                        input_shape = (INP,INP,3),
                        )


model_mobileNetV2.trainable = False
    
keras_input = Input(shape=(INP, INP, 3), name = 'image_input')
    
#Add the fully-connected layers 
output_mobileNetV2 = model_mobileNetV2(keras_input)
x = Flatten()(output_mobileNetV2)
x = Dense(4096, activation='relu', name='fc1')(x)
prediction = Dense(num_classes, activation='sigmoid', name='predictions')(x)
    
pretrained_model_mobile = Model(inputs=keras_input, outputs=prediction)   

print(pretrained_model_mobile.summary())

sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)

pretrained_model_mobile.compile(optimizer = 'sgd', loss = 'mean_squared_error', metrics = ['accuracy', metrics.binary_accuracy])
es = EarlyStopping(monitor = 'val_accuracy', mode = 'max', verbose = 1, restore_best_weights = True, patience = 4)
mc = ModelCheckpoint('best_MobileNet_model.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)

for l in pretrained_model_mobile.layers:
    print(l.name, l.trainable)

history = pretrained_model_mobile.fit(X_train, y_train,
                                   epochs = 50,
                                   validation_data=(X_test, y_test),
                                   workers=4,
                                   batch_size= BATCHSIZE,
                                   callbacks = [es, mc]
                         )

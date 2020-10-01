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


import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import train_test_split

INP = 224
BATCHSIZE = 128

#Read Images
def load_skirt_images(df, inputPath):
  # initialize our images array (i.e., the house images themselves)
  images = []
  clothesPaths = df["images__path"].tolist()

  for clothesPath in clothesPaths:
    fullPath = inputPath +"/" + clothesPath
    image = cv2.imread(fullPath)
    image = cv2.resize(image, (INP, INP))
    images.append(image)

  return np.array(images)

images = load_skirt_images(labels, 'clothesdatabase')
images = images/255.0


X_train, X_test, y_train, y_test = train_test_split(images, Y,test_size=0.33, shuffle=False)
print(len(X_train))
print(len(y_test))

# InceptionResNetV2

net = InceptionResNetV2(include_top=False,
                        weights='imagenet',
                        input_tensor=None,
                        input_shape = (INP,INP,3)
                        )
net.trainable = False
net_input = Input(shape=(INP, INP, 3), name = 'image_input')

x =  net(net_input)

x = Flatten()(x)
x = Dense(4096, activation='relu', name='fc1')(x)
output_layer = Dense(num_classes, activation='sigmoid', name='sigmoid')(x)
net_final = Model(inputs=net_input, outputs=output_layer)

for l in net_final.layers:
    print(l.name, l.trainable)


sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)


net_final.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy', metrics.binary_accuracy])
es = EarlyStopping(monitor = 'val_accuracy', mode = 'max', verbose = 1, restore_best_weights = True, patience = 5)
mc = ModelCheckpoint('best_InceptionResNetV2.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)

history = net_final.fit(X_train, y_train,
                                   epochs = 50,
                                   validation_data=(X_test, y_test),
                                   workers=4,
                                   batch_size= BATCHSIZE,
                                   callbacks = [es, mc]
                         )



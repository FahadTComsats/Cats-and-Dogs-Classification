#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 15:11:13 2019

@author: fahadtariq
"""

#Part-1 building the CNN Model.

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense


classifier = Sequential()

#Step 1 Convolution
classifier.add(Conv2D(filters=32,kernel_size=(3,3), input_shape=(64,64,3),activation = 'relu'))

#Step 2 Pooling

classifier.add(MaxPooling2D(pool_size=(2,2)))

# Adding second convolutional layer

classifier.add(Conv2D(filters=32,kernel_size=(3,3),activation = 'relu'))

classifier.add(MaxPooling2D(pool_size=(2,2)))

#Step 3 Flattening

classifier.add(Flatten())

#step 4 Full Connected

classifier.add(Dense(units = 128 , activation = 'relu'))
classifier.add(Dense(units = 1 , activation = 'sigmoid'))

#Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy',metrics = ['accuracy'])

#Fitting the CNN to images
from keras.preprocessing.image import ImageDataGenerator
import scipy.ndimage

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
                                                    'dataset/training_set',
                                                    target_size=(64, 64),
                                                    batch_size=32,
                                                    class_mode='binary')

test_set = test_datagen.flow_from_directory(
                                                    'dataset/test_set',
                                                    target_size=(64, 64),
                                                    batch_size=32,
                                                    class_mode='binary')

classifier.fit_generator(
                                                    training_set,
                                                    steps_per_epoch=250,
                                                    epochs=25,
                                                    validation_data=test_set,
                                                    validation_steps=62.5)



print("Saving model")
classifier.save('cat_dog_classifier.h5')
print('Done..')

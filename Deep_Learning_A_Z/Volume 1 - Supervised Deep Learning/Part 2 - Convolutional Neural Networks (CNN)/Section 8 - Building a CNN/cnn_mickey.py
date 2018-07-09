#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 18:49:41 2018

@author: mickey.liu
"""

#Part 1 - Building the CNN

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

img_width, img_height = 150, 150

"""
    p: dropout rate
    input_shape: shape of input
"""
def build_classifier(p, input_shape=(32, 32, 3)):
    print(input_shape)
    classifier = Sequential()
    classifier.add(Conv2D(32, 3, padding='same', input_shape=input_shape, activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    classifier.add(Conv2D(32, 3, padding='same', activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    classifier.add(Conv2D(64, 3, padding='same', activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    classifier.add(Conv2D(64, 3, padding='same', activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    
    classifier.add(Flatten())
    classifier.add(Dense(64, activation = 'relu' ))
    classifier.add(Dropout(p))
    classifier.add(Dense(64, activation = 'relu' ))
    classifier.add(Dense(64, activation = 'relu' ))
    classifier.add(Dropout(p/2))
    classifier.add(Dense(1, activation = 'sigmoid'))
    
    optimizer = Adam(lr=0.001)
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier



#Part 2 - Fitting CNN to images
def run_training(batch_size, epochs): 
    train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    training_set = train_datagen.flow_from_directory(
            'dataset/training_set',
            target_size=(img_width, img_height),
            batch_size=batch_size,
            class_mode='binary')
    
    test_set = test_datagen.flow_from_directory(
            'dataset/test_set',
            target_size=(img_width, img_height),
            batch_size=batch_size,
            class_mode='binary')
    
    classifier = build_classifier(0.55, (img_width, img_height, 3))
    classifier.fit_generator(
            training_set,
            steps_per_epoch=(8000/batch_size),
            epochs=epochs,
            validation_data=test_set,
            validation_steps=(2000/batch_size))

def main():
    run_training(32, 100)

if __name__ == "__main__":
    main()
    
#Part 3 - Making new predictions
import numpy as np
from keras.preprocessing import image 
test_image = image.load_img(
        'dataset/single_prediction/cat_or_dog_2.jpg',
        grayscale = False,
        target_size = (64, 64))
#At this point, the test_image is 2D, however we need it to be 3D
test_image = image.img_to_array(test_image)
#Now image is 3D (64, 64, 3)

#We need to add another dimension to represent the batch. Because functions of NN only accepts inputs in a batch. In this case, we'll have 1 batch of 1 input
test_image = np.expand_dims(test_image, axis = 0) #axis = position to add the new dimension
#(1, 64, 64, 3)
result = classifier.predict(test_image)
#result = array([[ 1.]])
training_set.class_indices #class_indices is an attribute of ImageDataGenerator.flow_from_directory
#returns {'cats':0, 'dogs':1}

if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'
    
    
    
#Evaluation was already made during the training with the validation set, therefore k-Fold Cross Validation is not needed.
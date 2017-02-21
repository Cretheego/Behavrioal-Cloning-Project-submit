# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 07:27:38 2017

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 23:51:07 2017

@author: Administrator
"""
import os
import csv
import cv2
import numpy as np
import sklearn
from sklearn.utils import shuffle
import matplotlib.image as mpimg
import model_func as mf
from sklearn.model_selection import train_test_split

rows = 32
cols = 64
# data pre-process
#=====================================
def generator(samples,labels, batch_size=64):
    num_samples = len(samples)
    while 1: # Used as a reference pointer so code always loops back around
        samples,labels = shuffle(samples,labels)
        for offset in range(0, num_samples, batch_size):
            X_train = samples[offset:offset+batch_size][:]
            y_train  = labels[offset:offset+batch_size][:]

            yield shuffle(X_train, y_train)

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Lambda
from keras.layers.core import Dropout,Reshape
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D

def network_model():
    model = Sequential()
    if 1:
        model.add(Reshape((rows,cols,1), input_shape=(rows,cols)))
        model.add(Lambda(lambda x: x/255.0 - 0.5,input_shape=(rows,cols,1)))
        model.add(Convolution2D(8,3,3,border_mode='valid',activation='elu'))
        model.add(Convolution2D(12,3,3,border_mode='valid',activation='elu'))
        model.add(MaxPooling2D((4,4),(4,4),'valid')) 
        model.add(Dropout(0.5))
        model.add(Convolution2D(16,3,3,border_mode='valid',activation='elu'))
        model.add(Convolution2D(16,3,3,border_mode='valid',activation='elu'))
        model.add(MaxPooling2D()) 
        model.add(Dropout(0.5))
   
        model.add(Flatten())
        model.add(Dense(128)) 
        model.add(Activation('elu'))
        model.add(Dropout(0.5))
        model.add(Dense(64)) 
        model.add(Activation('elu'))
        
        model.add(Dense(1)) 
    return model
    
def main():
    if 1:
        train_samples,train_labels = mf.get_data(rows,cols)
    if 0:
        train_samples = np.load('./a.npy')
        train_labels = np.load('./b.npy')
    print(np.shape(train_samples))
    print("------")
    train_samples,validation_samples,  train_labels, valid_labels =\
                 train_test_split(train_samples ,train_labels, test_size=0.2)
    print(np.shape(train_samples))
    
    model = network_model()
    model.compile(loss='mse', optimizer='adam')
    history = model.fit_generator(generator(train_samples,train_labels, batch_size=256),
                    samples_per_epoch=len(train_labels),\
                    validation_data=generator(validation_samples,valid_labels, batch_size=256),\
                    nb_val_samples=len(validation_samples), nb_epoch=10)

    mf.save_model(model, 'model')
    model.summary()
    
if __name__ == '__main__':
    main()    
    
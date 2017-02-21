# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 21:10:20 2017

@author: Administrator
"""

import numpy as np
import csv
import cv2
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
from sklearn.utils import shuffle
from skimage import exposure
from PIL import Image
from PIL import ImageEnhance
import os

def discard_zero(samples):
    sample = []
    for i in range(0, len(samples)):
        if (random.random() > 0.01) and (np.abs(float(samples[i][3])) < 0.005):
            continue
        else:
            sample.append(samples[i][:])
    return sample

# add left,right camora 
def add_left_rigt(name_left,name_right, center_angle):
    # create adjusted steering measurements for the side camera images
    correction = 0.3 # this is a parameter to tune
    steering_left = center_angle + correction
    steering_right = center_angle - correction

    img_left = plt.imread(name_left)
    img_left = image_rize(img_left,rows=32,cols=64)
    
    img_right = plt.imread(name_right)
    img_right = image_rize(img_right,rows=32,cols=64)
    return steering_left,steering_right,img_left,img_right

def save_model(model, name):
    model_file = name + '.json'
    model_weights_file = name + '.h5'

    if os.path.exists(model_file):
        os.remove(model_file)
    if os.path.exists(model_weights_file):
        os.remove(model_weights_file)

    model_json = model.to_json()

    with open(model_file, 'w') as file:
        file.write(model_json)
    
    model.save_weights(model_weights_file)
    
def image_rize(img, rows,cols):
    img = (cv2.cvtColor(img, cv2.COLOR_RGB2HSV))
    #print(img.shape)
    img = cv2.resize(img[:,:,1],(cols,rows))
    return img
    
   
def train_data(samples,path,sp, rows=32,cols=64):
    train_samples = np.zeros([0,rows,cols],dtype='uint8')
    train_labels  = np.zeros(0,dtype='float32')
    i = 0
    num = len(samples)
    for sample_i in samples:
        name = path + sample_i[0].split(sp)[-1]
        #print(np.shape(samples))
        name_left = path + sample_i[1].split(sp)[-1]
        name_right = path + sample_i[2].split(sp)[-1]
    
        center_image0 = plt.imread(name)
        #print(center_image0.shape)
        center_image = image_rize(center_image0,rows=32,cols=64)
        angles_center = float(sample_i[3])
   
        image_center_flipped = np.fliplr(center_image)
        angles_center_flipped = -1*angles_center
        if 1:
            angles_left,angles_right,left_image,right_image = \
                    add_left_rigt(name_left,name_right, angles_center)
            image_left_flipped = np.fliplr(left_image)
            image_right_flipped = np.fliplr(right_image)
                    
            angles_left_flipped = -1*angles_left
            angles_right_flipped = -1*angles_right

        train_samples = np.concatenate([train_samples,[center_image]])
        train_samples = np.concatenate([train_samples,[image_center_flipped]])
        if 1:
            train_samples = np.concatenate([train_samples,[left_image]])
            train_samples = np.concatenate([train_samples,[image_left_flipped]])
    
            train_samples = np.concatenate([train_samples,[right_image]])
            train_samples = np.concatenate([train_samples,[image_right_flipped]])
               
        train_labels = np.concatenate([train_labels,[angles_center]])
        train_labels = np.concatenate([train_labels,[angles_center_flipped]])
        if 1:
            train_labels = np.concatenate([train_labels,[angles_left]])
            train_labels = np.concatenate([train_labels,[angles_left_flipped]])
     
            train_labels = np.concatenate([train_labels,[angles_right]])
            train_labels = np.concatenate([train_labels,[angles_right_flipped]])
        i =i +1
        print(i,num)
    
    return shuffle(train_samples, train_labels)

def get_data(rows,cols):
    # data collected from simulator 
    samples = []
    with open('./data/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)
    
    samples = discard_zero(samples)
    path = './data/IMG/'
    sp = '\\'
    train_samples, train_labels = train_data(samples,path,sp,rows,cols)
    
    np.save("a.npy", train_samples)
    np.save("b.npy", train_labels)
    
    return train_samples,train_labels

import os
import librosa
import pickle
import h5py
import numpy as np


## GET DATA TO WORK ON
print("Start loading data")

fd = open("../S2L2.0_storage/data_x_freesfxx_mfcc.pkl", 'r')
fd2 = open("../S2L2.0_storage/data_y_freesfxx_mfcc.pkl", 'r')
features = pickle.load(fd)
labels = pickle.load(fd2)


print(labels[10])
print(labels[20])
#print(features[10])
#print(features[20])


data=h5py.File('../S2L2.0_storage/data_freesfxx_mfcc.h5','r')
features_=data.get('features')
labels_=data.get('labels')


print(labels_[10])
print(labels_[20])
#print(features_[10])
#print(features_[20])


print("Data loaded")
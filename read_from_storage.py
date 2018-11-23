
import os
import librosa
import pickle
import h5py
import numpy as np


## GET DATA TO WORK ON
print("Start loading data")

fd = open("../S2L2.0_storage/data_x_librosa_test_1.pkl", 'r')
fd2 = open("../S2L2.0_storage/data_y_librosa_test_1.pkl", 'r')
features = pickle.load(fd)
labels = pickle.load(fd2)


print(labels[100])
print(labels[200])
print(features[100])
print(features[200])


data=h5py.File('../S2L2.0_storage/audio_data_test_1.h5','r')
features_=data.get('features')
labels_=data.get('labels')


print(labels_[100])
print(labels_[200])
print(features_[100])
print(features_[200])


print("Data loaded")
## For sound feature analysis
## Written by: leopauly ! cnlp[at]leeds.ac.uk


## Imports
import os
import librosa
import pickle
import scipy
#import h5py
import numpy as np
import matplotlib.pyplot as plt
import librosa.display

## Values
n_bands = 60 #No: of bands used while taking DCT transformations
n_mfcc = 60 # No of MFCCs to return


def blur_signal(signal):
	return np.array(blur_signal)

def display_signal(signal,signal_name):
	librosa.display.waveplot(signal)
	plt.title(signal_name)
	plt.savefig(signal_name)
	plt.show()


def cut_16_slices(signal):
	print('Is file present:',os.path.isfile(full_filename))
	print('File loaded from:', full_filename)
	sound_clip, sr = librosa.load(full_filename)
	sound_slice=extract_slice(sound_clip, sr,fs)
	mfcc_spec = librosa.feature.mfcc(sound_slice, n_mfcc=n_mfcc)
	print('mfcc_spec.shape)',mfcc_spec.shape)
	librosa.display.specshow(mfcc_spec,x_axis='time')
	return np.array(slices)


full_filename='./A0.wav'
sound_clip, sr = librosa.load(full_filename)
display_signal(sound_clip[0:sr*1],'Sound clip')
print('\nSound clip shape:',sound_clip.shape)
print('Sampling rate:',sr)
print('Lenght of sound clip:',len(sound_clip/sr))









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
work_dir = "../../Dataset_online"


#-------------------------------------------------------------------------------------#

## Extracting slice
def extract_slice(sound_clip, sr,fs):
    print('sound clip shape:',sound_clip.shape)
    print('Sampling rate of full clip:',sr)
    print('Time duration of full clip:',librosa.core.get_duration(sound_clip,sr=sr))


    librosa.display.waveplot(sound_clip,sr=sr)
    #plt.title('Sound clip'+fs)
    #plt.savefig('Wave_clip'+fs+'.png')
    #plt.show()


    #-------------------------------------------------------------------------------------#
    start=0
    end=sr*2 # setting durating of a slice
    sound_slice=sound_clip[start:end]
    print('sound slice shape:',sound_slice.shape)
    print('Sampling rate of sound slice:',sr)
    print('Time duration of slice:',librosa.core.get_duration(sound_slice,sr=sr))


    librosa.display.waveplot(sound_slice)
    #plt.title('A single slice of the clip'+fs)
    #plt.savefig('Wave_slice'+fs+'.png')
    #plt.show()

    return sound_slice


#-------------------------------------------------------------------------------------#

## Extracting MFCC features
def extract_mfcc_features(class_dir):
    mfcc_spec_class=[]
    class_dir_files=sorted(os.listdir(work_dir + "/" +class_dir))
    print(class_dir_files)

    for fs in class_dir_files:
        full_filename=work_dir + "/" + class_dir + "/" + fs
        print('Is file present:',os.path.isfile(full_filename))
        print('File loaded from:', full_filename)
        sound_clip, sr = librosa.load(full_filename)
        sound_slice=extract_slice(sound_clip, sr,fs)
        mfcc_spec = librosa.feature.mfcc(sound_slice, n_mfcc=n_mfcc)
        print('mfcc_spec.shape)',mfcc_spec.shape)
        librosa.display.specshow(mfcc_spec,x_axis='time')
        mfcc_spec_class.append(mfcc_spec)

    return np.array(mfcc_spec_class)

                    
#-------------------------------------------------------------------------------------#

## Extracting features from alarm folder
class_dir= 'Alarm'
mfcc_spec_A=extract_mfcc_features(class_dir)


#-------------------------------------------------------------------------------------#

## Extracting features from Drilling folder
class_dir= 'Drilling'
mfcc_spec_D=extract_mfcc_features(class_dir)

#-------------------------------------------------------------------------------------#


def avg_distance(X,Y):
    distance=[]
    for x in X:
        for y in Y:
            distance.append(np.linalg.norm(x-y))
    distance_nonzero=[x for x in distance if x!=0]
    avg_distance_value=sum(distance_nonzero)/len(distance_nonzero)
    return np.array(avg_distance_value)


print('A with A:',avg_distance(mfcc_spec_A,mfcc_spec_A))
print('D with D:',avg_distance(mfcc_spec_D,mfcc_spec_D))
print('A with D:',avg_distance(mfcc_spec_A,mfcc_spec_D))
print('D with A:',avg_distance(mfcc_spec_D,mfcc_spec_A))


print('END')





















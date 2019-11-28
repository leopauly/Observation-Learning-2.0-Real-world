## AUDIO DATA PROCESSING
import os
import librosa
import pickle
import scipy
#import h5py
import numpy as np
import matplotlib.pyplot as plt
import librosa.display

'''
dim_ordering: 'th' or 'tf'. In 'th' mode, the channels dimension (the depth) is at index 1,
in 'tf' mode is it at index 3. It defaults to the  image_dim_ordering value found in your
Keras config file at ~/.keras/keras.json. If you never set it, then it will be "tf".

print K.image_dim_ordering()
'''

## This for mel spectogram resolution
n_bands = 60 #No: of bands used while taking DCT transformations
n_mfcc = 60 # No of MFCCs to return


work_dir = "./Dataset_realtime"


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
def extract_mfcc_features(sound_slice, n_mfcc,fs):
    mfcc_spec = librosa.feature.mfcc(sound_slice, n_mfcc=n_mfcc)
    print('mfcc_spec.shape)',mfcc_spec.shape)
    librosa.display.specshow(mfcc_spec,x_axis='time')
    #plt.title('MFCC'+fs+'.png')
    #plt.savefig('MFCC'+fs+'.png')
    #plt.colorbar()
    #plt.show()
    return mfcc_spec

                    
#-------------------------------------------------------------------------------------#

## Extracting features from alarm folder
sub_dir= 'Hammer_box'

fs= 'box0.wav'
full_filename=work_dir + "/" + sub_dir + "/" + fs
print('File loaded from:', full_filename)
sound_clip, sr = librosa.load(full_filename)
sound_slice=extract_slice(sound_clip, sr,fs)
mfcc_spec_A0=extract_mfcc_features(sound_slice, n_mfcc,fs)


fs= 'box1.wav'
full_filename=work_dir + "/" + sub_dir + "/" + fs
print('File loaded from:', full_filename)
sound_clip, sr = librosa.load(full_filename)
sound_slice=extract_slice(sound_clip, sr,fs)
mfcc_spec_A1=extract_mfcc_features(sound_slice, n_mfcc,fs)


fs= 'box2.wav'
full_filename=work_dir + "/" + sub_dir + "/" + fs
print('File loaded from:', full_filename)
sound_clip, sr = librosa.load(full_filename)
sound_slice=extract_slice(sound_clip, sr,fs)
mfcc_spec_A2=extract_mfcc_features(sound_slice, n_mfcc,fs)


#-------------------------------------------------------------------------------------#

## Extracting features from Drilling folder
sub_dir= 'Hammer_floor'

fs= 'floor0.wav'
full_filename=work_dir + "/" + sub_dir + "/" + fs
print('File loaded from:', full_filename)
sound_clip, sr = librosa.load(full_filename)
sound_slice=extract_slice(sound_clip, sr,fs)
mfcc_spec_D0=extract_mfcc_features(sound_slice, n_mfcc,fs)


fs= 'floor1.wav'
full_filename=work_dir + "/" + sub_dir + "/" + fs
print('File loaded from:', full_filename)
sound_clip, sr = librosa.load(full_filename)
sound_slice=extract_slice(sound_clip, sr,fs)
mfcc_spec_D1=extract_mfcc_features(sound_slice, n_mfcc,fs)


fs= 'floor2.wav'
full_filename=work_dir + "/" + sub_dir + "/" + fs
print('File loaded from:', full_filename)
sound_clip, sr = librosa.load(full_filename)
sound_slice=extract_slice(sound_clip, sr,fs)
mfcc_spec_D2=extract_mfcc_features(sound_slice, n_mfcc,fs)



#-------------------------------------------------------------------------------------#

print(np.linalg.norm(mfcc_spec_A0-mfcc_spec_A0))
print(np.linalg.norm(mfcc_spec_A0-mfcc_spec_A1))
print(np.linalg.norm(mfcc_spec_A0-mfcc_spec_A2))
print(np.linalg.norm(mfcc_spec_A0-mfcc_spec_D0))
print(np.linalg.norm(mfcc_spec_A0-mfcc_spec_D1))
print(np.linalg.norm(mfcc_spec_A0-mfcc_spec_D2))

print(np.linalg.norm(mfcc_spec_A1-mfcc_spec_A0))
print(np.linalg.norm(mfcc_spec_A1-mfcc_spec_A1))
print(np.linalg.norm(mfcc_spec_A1-mfcc_spec_A2))
print(np.linalg.norm(mfcc_spec_A1-mfcc_spec_D0))
print(np.linalg.norm(mfcc_spec_A1-mfcc_spec_D1))
print(np.linalg.norm(mfcc_spec_A1-mfcc_spec_D2))

print(np.linalg.norm(mfcc_spec_A2-mfcc_spec_A0))
print(np.linalg.norm(mfcc_spec_A2-mfcc_spec_A1))
print(np.linalg.norm(mfcc_spec_A2-mfcc_spec_A2))
print(np.linalg.norm(mfcc_spec_A2-mfcc_spec_D0))
print(np.linalg.norm(mfcc_spec_A2-mfcc_spec_D1))
print(np.linalg.norm(mfcc_spec_A2-mfcc_spec_D2))
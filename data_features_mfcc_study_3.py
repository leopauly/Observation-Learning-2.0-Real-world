## AUDIO DATA PROCESSING
import os
import librosa
import pickle
#import h5py
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import scipy

'''
dim_ordering: 'th' or 'tf'. In 'th' mode, the channels dimension (the depth) is at index 1,
in 'tf' mode is it at index 3. It defaults to the  image_dim_ordering value found in your
Keras config file at ~/.keras/keras.json. If you never set it, then it will be "tf".

print K.image_dim_ordering()
'''

window_size = 512
classes=['air condition','car horn','children playing','dog bark','drilling','engine idling','gun_shot','jack hammer','siren','street music']

## This for mel spectogram resolution
n_bands = 60
n_mfcc = 40
n_fft=2048
hop_length=512
power=2
dct_type=2
norm='ortho'

start=0
end=19968


work_dir = "/media/ironman2/2230FAE530FABF3B/UrbanSound8K/audio"
sub_dir= 'fold2'
fs= '14387-9-0-12.wav'

#-------------------------------------------------------------------------------------#


sound_clip, sr = librosa.load(work_dir + "/" + sub_dir + "/" + fs)
label = fs.split('-')[1]
print(label)

print('sound clip shape:',sound_clip.shape)
print('Sampling rate:',sr)

librosa.display.waveplot(sound_clip)
plt.title('Sound clip')
plt.savefig('Wave_clipe.png')
plt.show()


#-------------------------------------------------------------------------------------#

sound_slice=sound_clip[start:end]
print('sound slice shape:',sound_slice.shape)
print('Sampling rate:',sr)

librosa.display.waveplot(sound_slice)
plt.title('A single slice of the clip')
plt.savefig('Wave_slice.png')
plt.show()


#-------------------------------------------------------------------------------------#

spectrogram = librosa.core.stft(sound_slice,n_fft=n_fft,hop_length=hop_length)
print('Spectrogram shape:',spectrogram.shape)
print('Spectrogram',spectrogram)
librosa.display.specshow(spectrogram,sr=sr,y_axis='log',x_axis='time')
#plt.imshow(spectrogram)
plt.title('Amplitude spectrogram- Plot')
plt.colorbar()
plt.savefig('Spectrogram'+'.png')
plt.show()

#-------------------------------------------------------------------------------------#


dct=scipy.fftpack.dct(spectrogram,axis=0,type=dct_type,norm=norm)[:n_mfcc]
print('dct.shape)',dct.shape)
print(dct)
librosa.display.specshow(dct,x_axis='time')
plt.title('DCT')
plt.savefig('DCT'+'.png')
plt.colorbar()
plt.show()

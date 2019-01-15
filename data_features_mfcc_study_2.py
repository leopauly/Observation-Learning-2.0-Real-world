## AUDIO DATA PROCESSING
import os
import librosa
import pickle
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

window_size = 512
work_dir = "/media/ironman2/2230FAE530FABF3B/UrbanSound8K/audio"
classes=['air condition','car horn','children playing','dog bark','drilling','engine idling','gun_shot','jack hammer','siren','street music']

## This for mel spectogram resolution
n_bands = 60
n_mfcc = 40
n_fft=16
n_mels=200


sound_clip= np.array([1,2,3,4,5,6,7,8,9,10],dtype=float)
print('sound clip shape:',sound_clip.shape)
spectrogram = librosa.core.stft(sound_clip,n_fft=n_fft)
print('spectrogram.shape:',spectrogram.shape)
print('spectrogram:',spectrogram)

##-------------------------------------------------------------------------

spectrogram_abs=np.abs(spectrogram)
print('spectrogram_abs:',spectrogram_abs)
plt.imshow(spectrogram_abs)
plt.title('spectrogram_abs')
plt.colorbar()
plt.show()

##-------------------------------------------------------------------------

spectrogram_abs_db = librosa.amplitude_to_db(spectrogram_abs)
print('DB value of spectrogram_abs shape: ',spectrogram_abs_db.shape)
print('Spectrogram_abs_db',spectrogram_abs_db)
plt.imshow(spectrogram_abs_db)
plt.title('spectrogram_abs_db')
plt.colorbar()
plt.show()

#librosa.display.specshow(spectrogram_abs_db)
#plt.colorbar()
#plt.show()


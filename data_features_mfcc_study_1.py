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

spectrogram_abs = np.abs(spectrogram)**power
print('Absolute value of spectrgram shape: ',spectrogram_abs.shape)
print('Spectrogram_abs',spectrogram_abs)
librosa.display.specshow(spectrogram_abs,sr=sr,y_axis='log',x_axis='time')
#plt.imshow(spectrogram_abs)
plt.title('Power spectrogram')
plt.colorbar()
plt.savefig('Spectrogram_abs'+'.png')
plt.show()

#-------------------------------------------------------------------------------------#

spectrogram_abs_db = librosa.power_to_db(spectrogram_abs,ref=np.max)
print('DB value of spectrogram_abs shape: ',spectrogram_abs_db.shape)
print('Spectrogram_abs_db',spectrogram_abs_db)
librosa.display.specshow(spectrogram_abs_db,sr=sr,y_axis='log',x_axis='time')
#plt.imshow(spectrogram_abs_db)
plt.title('Log power spectrogram ')
plt.colorbar(format='%+2.0f dB')
plt.savefig('Spectrogram_abs_db'+'.png')
plt.tight_layout()
plt.show()
#-------------------------------------------------------------------------------------#



mel_fb=librosa.filters.mel(sr,n_fft,n_mels=n_bands)
print(mel_fb)
print('mel_fb.shape',mel_fb.shape)


##-------------------------------------------------------------------------

mel_spectrogram_user=np.dot(mel_fb,spectrogram_abs)
print(mel_spectrogram_user.shape)
librosa.display.specshow(mel_spectrogram_user,sr=sr,y_axis='mel',x_axis='time')
plt.colorbar()
plt.title('Mel_spectrogram user')
plt.savefig('Mel spectrogram user'+'.png')
plt.show()

mel_spectrogram_db_user = librosa.power_to_db(mel_spectrogram_user)
print('mel_spec_db user shape',mel_spectrogram_db_user.shape)
librosa.display.specshow(mel_spectrogram_db_user,sr=sr,y_axis='mel',x_axis='time')
plt.title('Log Mel_spectrogram -user')
plt.colorbar(format='%+2.0f dB')
plt.savefig('Mel_spectrogram _db_user'+'.png')
plt.show()



mel_spectrogram = librosa.feature.melspectrogram(sound_slice,n_mels=n_bands)
print(mel_spectrogram.shape)
librosa.display.specshow(mel_spectrogram,sr=sr,y_axis='mel',x_axis='time')
plt.title('Mel_spectrogram')
plt.colorbar()
plt.savefig('Mel spectrogram'+'.png')
plt.show()

print('sum',np.sum(mel_spectrogram-mel_spectrogram_user))
plt.imshow(mel_spectrogram-mel_spectrogram_user)
plt.colorbar()
plt.show()

#-------------------------------------------------------------------------------------#


mfcc_user=scipy.fftpack.dct(mel_spectrogram_db_user,axis=0,type=dct_type,norm=norm)[:n_mfcc]
print('mfcc_user.shape)',mfcc_user.shape)
librosa.display.specshow(mfcc_user,x_axis='time')
#plt.imshow(mfcc_spec)
plt.title('MFCC user')
plt.savefig('MFCC_user'+'.png')
plt.colorbar()
plt.show()


mfcc_spec = librosa.feature.mfcc(sound_slice, n_mfcc=n_mfcc, n_mels=n_bands)
print('mfcc_spec.shape)',mfcc_spec.shape)
librosa.display.specshow(mfcc_spec,x_axis='time')
#plt.imshow(mfcc_spec)
plt.title('MFCC')
plt.savefig('MFCC'+'.png')
plt.colorbar()
plt.show()

                    
#-------------------------------------------------------------------------------------#

print('sum',np.sum(mfcc_spec-mfcc_user))
plt.imshow(mfcc_spec-mfcc_user)
plt.colorbar()
plt.show()
## AUDIO DATA PROCESSING
import os
import librosa
import pickle
#import h5py
import numpy as np
import matplotlib.pyplot as plt

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
n_mfcc = 40
n_frames = 40


def windows(data, n_frames):
    ws = window_size * (n_frames - 1)
    print('window size',ws)
    start = 0
    while start < len(data):
        print(yield start, start + ws, ws)
        yield start, start + ws, ws
       
        start += (ws / 2)
        ## OVERLAP OF 50%
## END windows


def extract_features():
    raw_features = []
    _labels = []
    folder=0
    cnt = 0
    for sub_dir in os.listdir(work_dir):
        print("Working on dir: ", sub_dir)
        if sub_dir=='.DS_Store': continue
        if folder==1 : 
            print('breaking outer loop')
            break
        print('folder:',folder)
        folder=folder+1
        

        for fs in os.listdir(work_dir + "/" + sub_dir):
            if ".wav" not in fs: continue
            print("Loading file: ", fs)
            sound_clip, sr = librosa.load(work_dir + "/" + sub_dir + "/" + fs)
            print('Sound clip.shape and sampling rate:',sound_clip,sr)
            label = fs.split('-')[1]
            print(cnt, "Try Loading file: ", fs, " class: ", label)
            cnt += 1

            if cnt>5 :
                print('breaking inner loop')
                break 
            
            ## Work of file bacthes
            for (start, end, ws) in windows(sound_clip, n_frames):
                ## Get the sound part
                print('start to end',start,end)
                signal = sound_clip[start:end]
                print('Sound clip shape',sound_clip.shape)
                print('Signal shape',signal.shape)
                if len(signal) == ws:
                    print('start')
                    mfcc_spec = librosa.feature.mfcc(signal, n_mfcc=n_mfcc,n_mels=2000)
                    print(mfcc_spec.shape)
                    plt.imshow(mfcc_spec)#,cmap=plt.get_cmap('viridis',10))
                    plt.savefig('Slice spectrogram'+str(cnt)+'.png')
                    plt.show()
                    print('T',mfcc_spec.T.flatten().shape)
                    mfcc_spec = mfcc_spec.T.flatten()[:, np.newaxis].T
                    print(mfcc_spec.shape)
                    raw_features.append(mfcc_spec)
                    print('over')
                    _labels.append(label)
                    
                    
    print("Loaded ", cnt, " files")
    
    ## Add a new dimension
    raw_features_=np.array(raw_features)
    print('raw features shape before', raw_features_.shape)
    raw_features = np.asarray(raw_features).reshape(len(raw_features), n_mfcc, n_frames, 1)
    print('raw features shape after', raw_features.shape)

    ## Concate 2 elements on axis=3

    _features = np.concatenate((raw_features, np.zeros(np.shape(raw_features))), axis=3)
    print(' features shape before', _features.shape)
    _features = np.concatenate((_features, np.zeros(np.shape(raw_features))), axis=3)
    print('features shape before', _features.shape)

    for i in range(len(_features)):
        _features[i, :, :, 1] = librosa.feature.delta(order=1, data=_features[i, :, :, 0])
        _features[i, :, :, 2] = librosa.feature.delta(order=2, data=_features[i, :, :, 0])

    print('features shape final', _features.shape)

    plt.imshow(_features[10, :, :, 0])
    plt.show()
    plt.imshow(_features[10, :, :, 1])
    plt.show()
    plt.imshow(_features[10, :, :, 2])
    plt.show()

    return np.array(_features), np.array(_labels, dtype=np.int)
## END extract_features


features, labels = extract_features()



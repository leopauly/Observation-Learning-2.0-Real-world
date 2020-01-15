#### Recording audio visual data for object manipulation
#### Written by: leopauly (cnlp[at]leeds.ac.uk)

import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import librosa.display
import matplotlib.pyplot as plt

fs=44100 #Sample rate
duration = 2  # seconds
myrecording = sd.rec(duration * fs, samplerate=fs, channels=2,dtype='float64')
print("Recording Audio")
print(myrecording)
sd.wait()
print("Audio recording complete , Play Audio")
librosa.display.waveplot(myrecording,sr=fs)
plt.show()
sd.play(myrecording, fs)
sd.wait()
print("Play Audio Complete")

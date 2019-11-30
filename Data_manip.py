import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from python_speech_features import mfcc, logfbank
import librosa
import Plot
import soundfile as sf

#calculate fast fourier transform that changes the amplitude, time ----> frequency, magnitude
def calc_fft(signal, rate):
    n = len(signal)
    #real fft frequency values
    frequency = np.fft.rfftfreq(n,d=1/rate)
    #take magnitute
    Y = abs(np.fft.rfft(signal)/n)
    return (Y, frequency)

def envelope(signal, rate, threshold):
    mask =[]
    #convert a list to a series then apply as to it because alot of the time the signal is -
    signal = pd.Series(signal).apply(np.abs)
    #rolling window over series - min period is the minimum number of values before doing a calculation
    signal_mean = signal.rolling(window=int(rate/10), min_periods=1, center=True).mean()
    for mean in signal_mean:
        if(mean>threshold):
            mask.append(True)
        else:
            mask.append(False)
    return mask

def filter_data(classes):
    #dictionaries for saving the data we will plot
    signals ={}
    fft ={}
    fbank = {}  # bank energy signal
    mfccs = {}
    ## we try now to balance the data and clean it
    ##we will sample a sound from every class
    for c in classes:
        wav_file = df[df.label==c].iloc[0,0]
        signal, rate = librosa.load("Audio\\FSDKaggle2018.audio_train\\"+wav_file, sr=44100)
        #run an envelope on data to get rid of silence
        mask = envelope(signal, rate, 0.0005)
        signal = signal[mask]

        signals[c]=signal
        fft[c] = calc_fft(signal, rate)
        #filter bank
        #I think signal[:rate] is one second
        bank = logfbank(signal[:rate], rate, nfilt=26, nfft=1103).T
        fbank[c]= bank
        #Mel filtering / representation ----> Compression
        mel = mfcc(signal[:rate],rate, numcep=13, nfilt=26, nfft=1103).T
        mfccs[c]= mel
    return signals, fft, fbank, mfccs


def length_distrib(df):
    #we will be looking at the file names
    df.set_index('fname', inplace=True)
    for f in df.index:
        #for each file retrive sampling rate and signal
        rate, signal = wavfile.read("Audio\\FSDKaggle2018.audio_train\\"+f)
        # at that index create a new value called length and save in it --> length of signal in seconds
        df.at[f, 'length'] =signal.shape[0]/rate

#Read in Training Data
df = pd.read_csv('Audio\\FSDKaggle2018.meta\\train_post_competition.csv')
#calculate the length of each sample
#length_distrib(df)
#classes = list(np.unique(df.label))
#groups data by label, retrieve length, calculate total mean ----> mean distribution for every class
#class_distrib = df.groupby(['label'])['length'].mean()
#Plot.plot_class_distrib(class_distrib)

#return filename into its own place because we are done with it
#df.reset_index(inplace=True)


#signals, fft, fbank, mfccs = filter_data(classes)

#Plot.plot_signals(signals)
#Plot.plot_fft(fft)
#Plot.plot_fbank(fbank)
#Plot.plot_mfccs(mfccs)
#plt.show()

#clean the data and cp it into clean dir
#if len(os.listdir('Audio\\clean')) ==0:
#    for f in tqdm(df.fname):
#        #we down sample the data to 16000Hz instead of 44100Hz
#        signal, rate = librosa.load('Audio\\FSDKaggle2018.audio_train\\'+f, sr= 16000)
#        mask = envelope(signal, rate, 0.0005)
#        signal = signal[mask]
#        wavfile.write(filename='Audio\\clean\\'+f, rate = rate, data = signal)

#for f in tqdm(os.listdir('Audio\\FSDKaggle2018.audio_test\\')):
#    #signal, rate = librosa.load('output\\'+f, sr= 16000)
#    signal, rate = sf.read('Audio\\FSDKaggle2018.audio_test\\'+f, dtype='float32')
#    mask = envelope(signal, rate, 0.0005)
#    signal = signal[mask]
#    wavfile.write(filename='Audio\\clean_test\\'+f, rate = 16000, data = signal)
    #wavfile.write(filename='Audio\\clean\\'+f, rate = rate, data = signal)


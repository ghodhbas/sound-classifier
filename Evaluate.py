import os
from scipy.io import wavfile
import pandas as pd
import numpy as np
from tqdm import tqdm
from python_speech_features import mfcc
import pickle
from tensorflow.keras.models import load_model
from cfg import Config
from sklearn.metrics import accuracy_score
import sounddevice as sd
import Data_manip
from scipy.io import wavfile
import soundfile as sf

def build_pred(audio_dir):
    y_true=[]
    y_pred = []
    result =0
    print('extracting feature from audio file')
    rate, wav = wavfile.read('Audio\\clean_test\\1c01994c.wav')
    label = 'Computer_keyboard'
    c = classes.index(label)
    y_prob =[]
    #swipe the audio file
    for i in range(0, wav.shape[0]-config.step, config.step):
        sample = wav[i:i+config.step]
        x = mfcc(sample ,rate, numcep= config.nfeat, nfilt=config.nfilt, nfft=config.nfft)
        #scale data
        x = (x - config.min)/(config.max - config.min)
        if config.mode == 'alexnet' or config.mode == 'conv':
            x= x.reshape(1, x.shape[0], x.shape[1],1)
        #model prediction
        y_hat = model.predict(x)
        y_prob.append(y_hat)
        #add prediciton ons ample
        y_pred.append(np.argmax(y_hat))
        y_true.append(c)
    #take avg of all predictions
    total_probabilities = np.mean(y_prob, axis=0).flatten()
    return y_true, y_pred, total_probabilities


def predict(audio, rate):
    y_prob = []
    y_pred = []
    #swipe the audio file
    for i in range(0, audio.shape[0]-config.step, config.step):
        sample = audio[i:i+config.step]
        x = mfcc(sample ,rate, numcep= config.nfeat, nfilt=config.nfilt, nfft=config.nfft)
        #scale data
        x = (x - config.min)/(config.max - config.min)
        if config.mode == 'alexnet' or config.mode == 'conv':
            x= x.reshape(1, x.shape[0], x.shape[1],1)
        #model prediction
        y_hat = model.predict(x)
        y_prob.append(y_hat)
        y_pred.append(np.argmax(y_hat))

    total_probabilities = np.mean(y_prob, axis=0).flatten()
    return y_prob, y_pred, total_probabilities

#retreive classes
#df = pd.read_csv('Audio\\FSDKaggle2018.meta\\test_post_competition_scoring_clips.csv')
#classes = list(np.unique(df.label))

#classes = ['Acoustic_guitar', 'Applause', 'Bark', 'Bass_drum', 'Burping_or_eructation', 'Bus', 'Cello', 'Chime', 'Clarinet', 'Computer_keyboard', 'Cough', 'Cowbell', 'Double_bass', 'Drawer_open_or_close', 'Electric_piano', 'Fart', 'Finger_snapping', 'Fireworks', 'Flute', 'Glockenspiel', 'Gong', 'Gunshot_or_gunfire', 'Harmonica', 'Hi-hat', 'Keys_jangling', 'Knock', 'Laughter', 'Meow', 'Microwave_oven', 'Oboe', 'Saxophone', 'Scissors', 'Shatter', 'Snare_drum', 'Squeak', 'Tambourine', 'Tearing', 'Telephone', 'Trumpet', 'Violin_or_fiddle', 'Writing']
classes = [ 'Applause', 'Bark', 'Burping_or_eructation', 'Computer_keyboard', 'Cough', 'Drawer_open_or_close', 'Fart', 'Finger_snapping',  'Keys_jangling', 'Knock', 'Laughter', 'Meow', 'Microwave_oven', 'Scissors', 'Shatter', 'Squeak', 'Tearing', 'Telephone', 'Writing']
#classes = ["something", "Finger Snapping"]
#rebuild model
p_path = os.path.join('pickles','alexnet.p')
with open(p_path, 'rb') as file:
    config = pickle.load(file)
model = load_model(config.model_path)



#
#start predictions:
duration = 1
rate = 16000
#true_index = classes.index('Finger_snapping')
#clear terminal
clear = lambda: os.system('cls')
clear()
while True:
    print("Recording")
    data = sd.rec(int(duration * rate), samplerate=rate, channels=1)
    sd.wait()
    data = np.array(data).flatten()
    #clean
    mask = Data_manip.envelope(data, rate, 0.0005)
    data = data[mask]
    y_prob, y_pred, total_probabilities = predict(data, rate)
    print("Individual Predictions: "+str([classes[index] for index in y_pred]))
    #print("Probs:"+str(y_prob))
    #print("Total Probs:"+str(total_probabilities))
    ## predicition from avg
    result = total_probabilities[np.argmax(total_probabilities)]
    result_class = classes[np.argmax(total_probabilities)]
    print("Average result: "+ str(result_class) + "   ------  with prob: "+ str(result))
    print()

    #if true_index in y_pred:
    #    print("Snaps!")




#predict
#y_true, y_pred, result = build_pred('output')
#acc_score= accuracy_score(y_true=y_true, y_pred=y_pred)
#
#proba ={}
#for c, p in zip(classes, result):
#    proba[c] = p;
##clear terminal
#clear = lambda: os.system('cls')
#clear()
#
#print()
#print()
#print("expected label: "+ classes[y_true[0]])
#print()
#print("Predicted lavel: " + classes[np.argmax(result)] +"-------"+ str(result[np.argmax(result)]))
#print()
#print()
#print("probabilities "+ str(proba))
#
##proba = ''
##for c, p in zip(classes, y_prob):
##    proba += str(c)+": "+str(p)+"\n"
#
#y_pred = [classes[np.argmax(y)]  for y in y_probs]
#
#print(proba)
#print("Final Pred:"+ str(y_pred))
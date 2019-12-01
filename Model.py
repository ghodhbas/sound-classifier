import os
from scipy.io import wavfile
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D,  MaxPool3D, Flatten, LSTM, MaxPooling2D, BatchNormalization
from tensorflow.keras.layers import Dropout, Dense, TimeDistributed
from keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
from python_speech_features import mfcc
import Plot
import pickle
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from cfg import Config

#Check if we already built data
def check_data():
    if os.path.isfile(config.p_path):
        print("loading existing data for {} model".format(config.mode))
        with open(config.p_path, 'rb') as handle:
            tmp = pickle.load(handle)
            return tmp
    else:
        return None

def build_rand_feat():
    tmp = check_data()
    if tmp:
        #return x, y already stored
        return tmp.data[0], tmp.data[1]

    x = []
    y = []
    #we need to know min and max because they will be used in normalizing the input
    _min, _max = float('inf'), -float('inf')
    for _ in tqdm(range(n_samples)):
        # choose a random class
        choice = np.random.choice(class_distrib.index, p=prob_dist)
        # choose a random file in that class and load sound
        file = np.random.choice(df[df.label==choice].index)
        rate, signal = wavfile.read('Audio\\clean\\'+file)
      

        #retreive label
        #label = df.at[choice, 'label']
        #ingore files that are less than 0.25sec long
        if signal.shape[0]-config.step <=0 :
            continue
        #random start point
        rand_index = np.random.randint(0, signal.shape[0]-config.step)  #shape[0] mean length of signal -- might ommit  "-step" if file is too short
        #get sample
        sample = signal[rand_index: rand_index+ config.step]
        #filter sample into mfcc format -- compress mel spectogram data
        x_sample = mfcc(sample ,rate, numcep= config.nfeat, nfilt=config.nfilt, nfft=config.nfft)
        #track min and max
        _min = min(np.amin(x_sample), _min)
        _max = max(np.amax(x_sample), _max)
        #append data depending on the type of neural network
        x.append(x_sample)
        #append output as index of class
        y.append(classes.index(choice))
        #if(classes.index(choice) == 7):
        #    y.append(1)
        #else:
        #    y.append(0)

    config.min = _min
    config.max = _max
    #convert lists to np.array
    x , y = np.array(x), np.array(y)
    #normalize
    x = (x - _min)/(_max - _min)
    ##reshape the input to match the mode of neural network
    if config.mode == 'conv' or config.mode == 'vgg' or config.mode == 'alexnet':
        #nsample , time , depth
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2], 1)
    elif config.mode == 'recurr':
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2])
    
    #encode output into the class number (from index position --> class number) ---------- add 1 to index
    y = to_categorical(y, num_classes=19)
    print("Y = "+ str(y))
    config.data = (x,y)
    #store data in pickle
    with open(config.p_path,'wb') as handle:
        pickle.dump(config, handle, protocol=2)
    return x, y


def get_alexnet_model():
    print("MODEL PATH: "+ config.model_path)
    
    if os.path.isfile(os.path.join('.\\',config.model_path)):
        print("Loading Existing Alexnet Model")
        model = load_model(config.model_path)
        return model

    model = tf.keras.models.Sequential()
    model.add(Conv2D(48, 11,  input_shape=input_shape, strides=(2,3), activation='relu', padding='same'))
    model.add(MaxPooling2D(2, strides=(1,2)))
    model.add(BatchNormalization())
    model.add(Conv2D(128, 5, strides=(2,3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(192, 3, strides=(1, 2), activation='relu', padding='same'))
    model.add(Conv2D(192, 3, strides=(1, 1), activation='relu', padding='same'))
    model.add(Conv2D(128, 3, strides=(1, 1), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(19, activation='softmax'))
    #opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer ='adam', loss='categorical_crossentropy', metrics=['acc'])
    return model


def calculate_length_distrib(df):
    #we will be looking at the file names
    df.set_index('fname', inplace=True)
    for f in df.index:
        #for each file retrive sampling rate and signal
        rate, signal = wavfile.read("Audio\\clean\\"+f)
        # at that index create a new value called length and save in it --> length of signal in seconds
        df.at[f, 'length'] =signal.shape[0]/rate

df = pd.read_csv('Audio\\FSDKaggle2018.meta\\train_post_competition.csv')
#calculate_length_distrib(df) # and change index to file name
#classes = list(np.unique(df.label))
#class_distrib = df.groupby(['label'])['length'].mean()
##Plot.plot_class_distrib(class_distrib)


#this is to try to work on smaller set of classes
classes = [ 'Applause', 'Bark', 'Burping_or_eructation', 'Computer_keyboard', 'Cough', 'Drawer_open_or_close', 'Fart', 'Finger_snapping',  'Keys_jangling', 'Knock', 'Laughter', 'Meow', 'Microwave_oven', 'Scissors', 'Shatter', 'Squeak', 'Tearing', 'Telephone', 'Writing']
#new filtered  data ste
df_in_class = [True if label in classes  else False for label in df['label'] ]
df = df[df_in_class]
calculate_length_distrib(df) # and change index to file name
class_distrib = df.groupby(['label'])['length'].sum()
#print(class_distrib)
#Plot.plot_class_distrib(class_distrib)
#plt.show()

#Our samples represents the total number of audio chanks that we form from our wav files:
# We chunk the wave files into smaller clips of (10ms) / x2 for more samples = more data
# we basiclaly multiply the total length of our audios by 10 and then by 2  ( was 0.1)
n_samples = 6*  int(df['length'].sum())
#probability distribution of classes based on their total length of audio clips
prob_dist = class_distrib / class_distrib.sum()

config = Config(mode='alexnet')
#buidl feature set from random sampling(choice)
x, y = build_rand_feat()
#conver the 2d y array into the original 0 indexed 1d array
y_flat = np.argmax(y, axis=1)
if config.mode == 'conv':
    input_shape = (x.shape[1], x.shape[2],1)
    model = get_conv_model()

elif config.mode == 'recurr':
    input_shape = (x.shape[1], x.shape[2])
    model = get_recurr_model()

elif config.mode == 'vgg':
    input_shape = (x.shape[1], x.shape[2],1)
    model = get_vgg_model()

elif config.mode == 'alexnet':
    input_shape = (x.shape[1], x.shape[2],1)
    model = get_alexnet_model()

#this will help us balance the classes using prob distrib of the classes 
#it componsate the lack of data for some classes within the model parameters
#it helps accuracy by a little bit
# return the weight for classes to be used so that the training is balanced
class_weight = compute_class_weight('balanced', np.unique(y_flat), y_flat)

#create a check point
#monitors validationa ccuracy and saves only when there is improvement 
#checks every epoch (period)
checkpoint = ModelCheckpoint(config.model_path, monitor='val_acc', verbose = 1, mode='max', save_best_only=True, save_weights_only=False, period=1)

#clear terminal
clear = lambda: os.system('cls')
clear()


model.fit(x,y, epochs=30, batch_size=32, shuffle=True, class_weight=class_weight, validation_split=0.15, callbacks = [checkpoint] )
model.save(config.model_path)


#changed rate of data and nftt to 1103 from 512
# im using the data the way it is without cleaning (without envelope and withotu downgrade to 16hz rate)
import sounddevice as sd
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
import msvcrt
import Data_manip
import numpy as np

sd.default.channels = 1
duration = 1
i=2

##start recording
print("Press Enter To terminate")
#while True:
#    name = 'ouput\output'+str(i)+'.wav'
#    print('Recording Iterations: '+str(i))
#    data = sd.rec(int(duration * 16000), channels=1)
#    sd.wait()
#    write(name, 16000,data)
#    ##start prediction
#
#    ###########
#    #start recording
#    data = sd.rec(int(duration * 16000), channels=2)
#    if msvcrt.kbhit():
#        break
#    i +=1

while True:
    name = "output\\"+ str(i) + ".wav"
    data = sd.rec(int(duration * 44100), channels=1)
    sd.wait()
    data = np.array(data).flatten()
    mask = Data_manip.envelope(data, 44100, 0.0005)
    signal = data[mask]
    write(name, 00, signal) 
    i +=1

    
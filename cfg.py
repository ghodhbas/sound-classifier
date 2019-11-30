import os

class Config:
    #nfilt: number of filter used to compress the audio
    #nfeat mumber of mfcc coefficient
    #nfft is down to 512 because we down sampled rate from 44100 to 16000
    def __init__(self, mode='conv', nfilt=26, nfeat=13, nfft=512, rate = 16000):
        self.mode = mode
        self.nfilt = nfilt
        self.nfeat = nfeat
        self.nfft = nfft
        self.rate = rate
        self.step = int(rate/10) # 
        self.model_path = os.path.join('models', mode+'.model')
        self.p_path = os.path.join('pickles', mode+'.p')

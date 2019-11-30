import matplotlib.pyplot as plt

def plot_signals(signals):
    fig, axes = plt.subplots(nrows=8, ncols=5, sharex=False,
                             sharey=True, figsize=(15,10))
    fig.suptitle('Time Series', size=20)
    i = 0
    for x in range(8):
        for y in range(5):
            axes[x,y].set_title(list(signals.keys())[i])
            axes[x,y].plot(list(signals.values())[i])
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i += 1
    plt.draw()

def plot_fft(fft):
    fig, axes = plt.subplots(nrows=8, ncols=5, sharex=False,
                             sharey=True, figsize=(15,10))
    fig.suptitle('Fourier Transforms', size=16)
    i = 0
    for x in range(8):
        for y in range(5):
            data = list(fft.values())[i]
            Y, freq = data[0], data[1]
            axes[x,y].set_title(list(fft.keys())[i])
            axes[x,y].plot(freq, Y)
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i += 1
    plt.draw()

def plot_fbank(fbank):
    fig, axes = plt.subplots(nrows=8, ncols=5, sharex=False,
                             sharey=True, figsize=(15,10))
    fig.suptitle('Filter Bank Coefficients', size=16)
    i = 0
    for x in range(8):
        for y in range(5):
            axes[x,y].set_title(list(fbank.keys())[i])
            axes[x,y].imshow(list(fbank.values())[i],
                    cmap='hot', interpolation='nearest')
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i += 1
    plt.draw()

def plot_mfccs(mfccs):
    fig, axes = plt.subplots(nrows=8, ncols=5, sharex=False,
                             sharey=True, figsize=(15,10))
    fig.suptitle('Mel Frequency Cepstrum Coefficients', size=16)
    i = 0
    for x in range(8):
        for y in range(5):
            axes[x,y].set_title(list(mfccs.keys())[i])
            axes[x,y].imshow(list(mfccs.values())[i],
                    cmap='hot', interpolation='nearest')
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i += 1
    plt.draw()

def plot_class_distrib(class_distrib):
    fig, ax = plt.subplots(figsize=(15,10))
    ax.set_title("Class Distribution", y=1.08)
    ax.pie(class_distrib,labels=class_distrib.index , autopct='%1.1f%%', shadow=False, startangle=90)
    plt.draw()
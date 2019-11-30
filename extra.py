
def get_conv_model():

    if os.path.isdir( os.path.join(dir,config.model_path)  ):
        print("Loading Existing Convolutional Model")
        model = load_model(config.model_path)
        return model
    #for now our input space is really smalll (13,9,1) so our model is small
    #we cann add batch_normalization
    print("Creating a new convolutional model")
    model = tf.keras.models.Sequential()
    model.add(Conv2D(16, (3,3), activation='relu', strides=(1,1), padding='same', input_shape= input_shape  ))
    model.add(Conv2D(32, (3,3), activation='relu', strides=(1,1), padding='same' ))
    model.add(Conv2D(64, (3,3), activation='relu', strides=(1,1), padding='same' ))
    model.add(Conv2D(128, (3,3), activation='relu', strides=(1,1), padding='same' ))
    model.add(Conv2D(256, (3,3), activation='relu', strides=(1,1), padding='same' ))
    model.add(MaxPool2D((2,2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(128 , activation = 'relu'))
    model.add(Dense(64 , activation = 'relu'))
    model.add(Dense(41 , activation = 'softmax'))
    model.summary()
    opt = tf.keras.optimizers.Adam(learning_rate=0.005)
    model.compile(optimizer = 'adam', loss='categorical_crossentropy', metrics=['acc'])
    return model

def get_recurr_model():
    #shape of data for rnn is (n,time, feature)
    model = tf.keras.models.Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape = input_shape))
    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(64, activation='relu'))  )
    model.add(TimeDistributed(Dense(32, activation='relu'))  )
    model.add(TimeDistributed(Dense(16, activation='relu'))  )
    model.add(TimeDistributed(Dense(8, activation='relu'))  )
    model.add(Flatten() )
    model.add(Dense(41, activation = 'softmax'))
    model.summary()
    #opt = tf.keras.optimizers.Adam(learning_rate=0.005, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(optimizer ='adam', loss='categorical_crossentropy', metrics=['acc'])
    return model



def get_vgg_model():
    #we cann add batch_normalization
    from tensorflow.keras.applications.vgg16 import VGG16
    # create new empty model with desired input_shape
    base_model = VGG16(weights=None, input_shape=input_shape, include_top=False)
    # you can also set this as: weights='imagenet' if you want transfer learning from imagenet
    base_model.add(GlobalAveragePooling2D())
    base_model.add(Dense(1024, activation='relu'))
    base_model.add(Dense(41, activation='softmax'))
    model.compile(optimizer ='adam', loss='categorical_crossentropy', metrics=['acc'])
    return model


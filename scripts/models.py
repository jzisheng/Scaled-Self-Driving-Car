import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Lambda, ELU
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers import Cropping2D
from keras.optimizers import Adadelta


def get_cnn1_model():
    model = keras.models.Sequential()
    #model.add(keras.layers.Dense(100, input_dim=input_size, activation='tanh'))
    model.add(keras.layers.Cropping2D(cropping=((20,0), (0,0)), input_shape=(120, 160, 3)))
    model.add(keras.layers.Conv2D( 32, (5, 5), strides=(2,2), padding='same'))
    model.add(keras.layers.ELU())
 
    model.add(keras.layers.Conv2D( 32, (3, 3), strides=(2,2), padding='same'))
    model.add(keras.layers.ELU())
    
    model.add(keras.layers.Conv2D( 16,( 3, 3), strides=(1,1), padding='same'))
    model.add(keras.layers.ELU())
    
    model.add(keras.layers.Dropout(0.5))
    
    model.add(keras.layers.Flatten())
    
    model.add(keras.layers.Dense(2048))
    model.add(keras.layers.ELU())
    
    model.add(keras.layers.Dense(1024))
    model.add(keras.layers.Dropout(0.4))
    model.add(keras.layers.ELU())
    
    model.add(keras.layers.Dense(526))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.ELU())
    
    model.add(keras.layers.Dense(128))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.ELU())
    
    model.add(keras.layers.Dense(1, activation='relu'))
    
    return model

def get_nvidia_model(num_outputs):
    '''
    this model is inspired by the NVIDIA paper
    https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
    Activation is ELU
    '''
    
    model = Sequential()

    model.add(Cropping2D(cropping=((20,0), (0,0)), input_shape=(120, 160, 3)))

    model.add(Lambda(lambda x: x/127.5 - 1.))
    model.add(Conv2D(24, (5, 5), strides=(2, 2), padding="same"))
    model.add(ELU())
    model.add(Conv2D(32, (5, 5), strides=(2, 2), padding="same"))
    model.add(ELU())
    model.add(Conv2D(64, (5, 5), strides=(2, 2), padding="same"))
    model.add(ELU())
    model.add(Conv2D(64, (3, 3), strides=(2, 2), padding="same"))
    model.add(ELU())
    model.add(Dropout(.5))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), padding="same"))
    model.add(Flatten())
    model.add(Dense(1000))
    model.add(Dropout(.1))
    model.add(ELU())
    model.add(Dense(128))
    model.add(ELU())
    model.add(Dense(num_outputs))

    model.compile(optimizer=Adadelta(), loss="mse")
    return model
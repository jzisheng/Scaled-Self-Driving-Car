import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Lambda, ELU
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers import Cropping2D
from keras.optimizers import Adadelta

def default_categorical():
    from keras.layers import Input, Dense, merge
    from keras.models import Model
    from keras.layers import Convolution2D, MaxPooling2D, Reshape, BatchNormalization
    from keras.layers import Activation, Dropout, Flatten, Dense
    
    img_in = Input(shape=(120, 160, 3), name='img_in')  
    # First layer, input layer, Shape comes from camera.py resolution, RGB
    x = img_in
    x = Convolution2D(24, (5,5), strides=(2,2), activation='relu')(x) 
    # 24 features, 5 pixel x 5 pixel kernel (convolution, feauture) window, 2wx2h stride, relu activation
    x = Convolution2D(32, (5,5), strides=(2,2), activation='relu')(x)      
    # 32 features, 5px5p kernel window, 2wx2h stride, relu activatiion
    x = Convolution2D(64, (5,5), strides=(2,2), activation='relu')(x)     
    # 64 features, 5px5p kernal window, 2wx2h stride, relu
    x = Convolution2D(64, (3,3), strides=(2,2), activation='relu')(x)     
    # 64 features, 3px3p kernal window, 2wx2h stride, relu
    x = Convolution2D(64, (3,3), strides=(1,1), activation='relu')(x)     
    # 64 features, 3px3p kernal window, 1wx1h stride, relu

    # Possibly add MaxPooling (will make it less sensitive to position in image).  
    # Camera angle fixed, so may not to be needed

    x = Flatten(name='flattened')(x)                                  
    # Flatten to 1D (Fully connected)
    x = Dense(100, activation='relu')(x)                                 
    # Classify the data into 100 features, make all negatives 0
    x = Dropout(.1)(x)                                                   
    # Randomly drop out (turn off) 10% of the neurons (Prevent overfitting)
    x = Dense(50, activation='relu')(x)                                  
    # Classify the data into 50 features, make all negatives 0
    x = Dropout(.1)(x)                                                   
    # Randomly drop out 10% of the neurons (Prevent overfitting)
    #categorical output of the angle
    angle_out = Dense(15, activation='softmax', name='angle_out')(x)      
    # Connect every input with every output and output 15 hidden units. 
    # Use Softmax to give percentage. 15 categories and find best one based off percentage 0.0-1.0
    
    #continous output of throttle
    throttle_out = Dense(1, activation='relu', name='throttle_out')(x)      # Reduce to 1 number, Positive number only
    
    model = Model(inputs=[img_in], outputs=[angle_out, throttle_out])
    model.compile(optimizer='adam',
                  loss={'angle_out': 'categorical_crossentropy', 
                        'throttle_out': 'mean_absolute_error'},
                  loss_weights={'angle_out': 0.9, 'throttle_out': .001})

    return model



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
    
    model.add(keras.layers.Dense(1, activation='linear'))
    
    return model

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
    
    model.add(keras.layers.Dense(15, activation='relu'))
    
    return model

def get_nvidia_model(num_outputs):
    
    model = Sequential()

    model.add(Cropping2D(cropping=((30,0), (0,0)), input_shape=(120, 160, 3)))

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

    model.compile(optimizer="adam", loss="mse")
    return model
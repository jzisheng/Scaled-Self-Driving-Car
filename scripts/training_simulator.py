from os import walk
import numpy
import keras
import sys
import PIL
import utils
from PIL import Image

class DataLoader(object):
    max_train = 10
    index_start = 0
    
    filepath = ""
    def parse_img_name(self, fstr):
        """ 
        Parses image file name and returns steering angle
        """
        data = fstr.split("_")
        if(len(data)>2):
            #print(float(data[3])/25.0)
            result =float(data[3])/30.0
            return result
        else:
            return -1
        
    def prepare_files(self, path):
        """ 
        Walks through data directory, converts images
        to NP arrays and stores steering angles
        """
        # Numpy array containing images
        imgs = []
        # List containing steering angles
        steerings = []
        files = self.get_image_paths(path)
        # Prepare training data
        count = 0
        index_count = 0
        for fname in files:
            if index_count < self.index_start:
                index_count +=1 # Starts loading file from this index point
            # Starting from file index index_start
            else:
                parsed = self.parse_img_name(fname)
                if not(parsed == -1):        # Make sure the file is valid
                    steerings.append( utils.linear_bin(parsed) ) # Append steering
                    img = self.load_image(path+fname)
                    # img = self.normalize(self.load_image(path+fname))   # Load the image
                    imgs.append(img)                          # Append image to images array
                    count+=1              
                    if(count > self.max_train):
                        break
        return imgs, steerings

    def get_image_paths(self, path):
        # Walk through all the images in the folder path
        f = []
        for (dirpath, dirnames, filenames) in walk(path):
            f.extend(filenames)
        return f

    def load_image(self, infilename):
        size = 160,120
        img = Image.open(infilename)
        img.thumbnail(size, Image.ANTIALIAS)
        data = numpy.asarray(img,dtype="uint8")
        return data

    def normalize(self, arr):
        """
        Linear normalization
        http://en.wikipedia.org/wiki/Normalization_%28image_processing%29
        """
        arr = arr.astype('float')
        for i in range(3):
            minval = arr[...,i].min()
            maxval = arr[...,i].max()
            if minval != maxval:
                arr[...,i] -= minval
                arr[...,i] *= (255.0/(maxval-minval))
        return arr
    
    def load(self):       
        imgs, steerings = self.prepare_files(self.filepath)
        return numpy.array(imgs), numpy.array(steerings)
    
    def test(self,x):
        print(x)
    
    def __init__(self,filepath,max_train=10000,index_start = 0):
        self.data = []
        self.filepath = filepath
        self.max_train = max_train
        self.index_start=index_start
        
        

def generator(samples, batch_size=32, perc_to_augment=0.5):
    '''
    Rather than keep all data in memory, we will make a function that keeps
    it's state and returns just the latest batch required via the yield command.
    
    As we load images, we can optionally augment them in some manner that doesn't
    change their underlying meaning or features. This is a combination of
    brightness, contrast, sharpness, and color PIL image filters applied with random
    settings. Optionally a shadow image may be overlayed with some random rotation and
    opacity.
    We flip each image horizontally and supply it as a another sample with the steering
    negated.
    '''
    num_samples = len(samples)   
    
    while 1: # Loop forever so the generator never terminates
        samples = shuffle(samples)
        #divide batch_size in half, because we double each output by flipping image.
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            
            images = []
            controls = []
            for fullpath in batch_samples:
                try:
                    data = parse_img_filepath(fullpath)
                
                    steering = data["steering"]
                    throttle = data["throttle"]

                    try:
                        image = Image.open(fullpath)
                    except:
                        image = None

                    if image is None:
                        print('failed to open', fullpath)
                        continue

                    #PIL Image as a numpy array
                    image = np.array(image)

                    if len(shadows) > 0 and random.uniform(0.0, 1.0) < perc_to_augment:
                        image = augment.augment_image(image, shadows)

                    center_angle = steering
                    images.append(image)
                    
                    if conf.num_outputs == 2:
                        controls.append([center_angle, throttle])
                    elif conf.num_outputs == 1:
                        controls.append([center_angle])
                    else:
                        print("expected 1 or 2 ouputs")

                except:
                    print("we threw an exception on:", fullpath)
                    yield [], []


            # final np array to submit to training
            X_train = np.array(images)
            y_train = np.array(controls)
            yield X_train, y_train

            
""" The two functions below work exclusively for the simulator """
def make_generators(inputs, limit=None, batch_size=32, aug_perc=0.0):
    '''
    load the job spec from the csv and create some generator for training
    '''
    # get the image/steering pairs from the csv files
    # lines = get_files(inputs)
    # print("found %d files" % len(lines))

    if limit is not None:
        lines = lines[:limit]
        print("limiting to %d files" % len(lines))
    
    train_samples, validation_samples = train_test_split(lines, test_perc=0.2)
    
    print("num train/val", len(train_samples), len(validation_samples))
    
    # compile and train the model using the generator function
    train_generator = generator(train_samples, batch_size=batch_size, perc_to_augment=aug_perc)
    validation_generator = generator(validation_samples, batch_size=batch_size, perc_to_augment=0.0)
    
    n_train = len(train_samples)
    n_val = len(validation_samples)
    
    return train_generator, validation_generator, n_train, n_val

def go(model_name, epochs=50, inputs='./log/*.jpg', limit=None, aug_mult=1, aug_perc=0.0):

    print('working on model', model_name)
    
    '''
    modify config.json to select the model to train.
    '''
    model = models.get_nvidia_model(conf.num_outputs)

    '''
    display layer summary and weights info
    '''
    models.show_model_summary(model)

    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=conf.training_patience, verbose=0),
        keras.callbacks.ModelCheckpoint(model_name, monitor='val_loss', save_best_only=True, verbose=0),
    ]
    
    batch_size = conf.training_batch_size


    #Train on session images
    train_generator, validation_generator, n_train, n_val = \
        make_generators(inputs, limit=limit, batch_size=batch_size, aug_perc=aug_perc)

    if n_train == 0:
        print('no training data found')
        return

    steps_per_epoch = n_train // batch_size
    validation_steps = n_val // batch_size

    print("steps_per_epoch", steps_per_epoch, "validation_steps", validation_steps)

    history = model.fit_generator(train_generator, 
        steps_per_epoch = steps_per_epoch,
        validation_data = validation_generator,
        validation_steps = validation_steps,
        epochs=epochs,
        verbose=1,
        callbacks=callbacks)
    
    try:
        if do_plot:
            # summarize history for loss
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper left')
            plt.savefig('loss.png')
    except:
        print("problems with loss graph")
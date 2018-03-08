from os import walk
import numpy
import keras
import sys
import PIL
from PIL import Image

class DataLoader(object):
    max_train = 10
    filepath = ""
    def parse_img_name(self, fstr):
        """ 
        Parses image file name and returns steering angle
        """
        data = fstr.split("_")
        if(len(data)>2):
            return(data[3])
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
        for fname in files:
            parsed = self.parse_img_name(fname)
            if not(parsed == -1):                         # Make sure the file is valid
                steerings.append( self.parse_img_name(fname) ) # Append steering
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
    
    def __init__(self,filepath,max_train):
        self.data = []
        self.filepath = filepath
        self.max_train = max_train
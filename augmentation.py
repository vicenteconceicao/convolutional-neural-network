# example of zoom image augmentation
from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot

import matplotlib.pyplot as plt

from PIL import Image
import numpy as np

train_path = "train.txt"
data_path = "./data/"
aug_train_path = "aug_train.txt"
data_aug_path = "./data_augmented/"


new_test_file = open(aug_train_path, 'w+')

# loading images
with open(train_path, "r") as rfile:
    for line in rfile:
        print(line)
        file_name, file_class = line.split(" ")
        img = load_img(data_path+file_name)
        # Saving image to a new directory
        img.save(data_aug_path+file_name)
        # convert to numpy array
        data = img_to_array(img)
        # writing test file
        new_test_file.write(line)
        # expand dimension to one sample
        samples = expand_dims(data, 0)
        # create image data augmentation generator
        datagen = ImageDataGenerator(zoom_range=[0.9,1.5], rotation_range=90)
        # prepare iterator
        it = datagen.flow(samples, batch_size=1)
        # generate samples and plot
        for i in range(6):
            # define subplot
            pyplot.subplot(330 + 1 + i)
            # generate batch of images
            batch = it.next()
            # convert to unsigned integers for viewing
            image = batch[0].astype('uint8')
            # plot raw pixel data
            pyplot.imshow(image)
            # File name
            new_file_name = str(i)+file_name
            # Convert to image
            im = Image.fromarray(image)
            # Saving file
            im.save(data_aug_path+new_file_name)
            new_test_file.write(new_file_name+" "+file_class)

new_test_file.close()

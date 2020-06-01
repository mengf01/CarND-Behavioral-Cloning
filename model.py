import os
import csv
import cv2
from scipy import ndimage
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.models import Sequential, Model
from keras.layers import Lambda, Cropping2D, Dense, Dropout, Flatten, ELU, Activation
from keras.layers.convolutional import Convolution2D
import math

data_root_paths = ['/opt/carnd_p3/data', '/opt/train_data_car_sim_2']
# data_root_paths = ['/opt/carnd_p3/data']

samples = []
for data_root_path in data_root_paths:
    with open(data_root_path+'/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)  # skip the headers
        for line in reader:
            samples.append(line)
print("Total num of samples:")
print(len(samples))

train_samples, validation_samples = train_test_split(samples, test_size=0.2)

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                if batch_sample[0].startswith('IMG/center_2016'):
                    img_path = data_root_paths[0]
                else:
                    img_path = data_root_paths[1]
                center_name = img_path+'/IMG/'+batch_sample[0].split('/')[-1]
                left_name = img_path+'/IMG/'+batch_sample[1].split('/')[-1]
                right_name = img_path+'/IMG/'+batch_sample[2].split('/')[-1]
                center_image = ndimage.imread(center_name)
                left_image = ndimage.imread(left_name)
                right_image = ndimage.imread(right_name)
                # create adjusted steering measurements for the side camera images
                correction = 0.2 # this is a parameter to tune
                steering_center = float(batch_sample[3])
                steering_left = steering_center + correction
                steering_right = steering_center - correction
                # append data
#                 images.extend([center_image, left_image, right_image, cv2.flip(center_image, 1)])
#                 angles.extend([steering_center, steering_left, steering_right, steering_center * -1.0])
                images.extend([center_image, cv2.flip(center_image, 1)])
                angles.extend([steering_center, steering_center * -1.0])
            # add data to train
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# Set our batch size
batch_size = 32

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

ch, row, col = 3, 160, 320

# # using model architecture from the comma ai paper
# # https://github.com/commaai/research/blob/master/train_steering_model.py
# model = Sequential()
# # # Trim image to remove top and bottom
# model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(row, col, ch)))
# # Preprocess incoming data, centered around zero with small standard deviation
# model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(row, col, ch)))
# # model.add(Flatten(input_shape=(160, 320, 3)))
# model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same", input_shape=(ch, row, col)))
# model.add(ELU())
# model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
# model.add(ELU())
# model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
# model.add(Flatten())
# model.add(Dropout(.2))
# model.add(ELU())
# model.add(Dense(512))
# model.add(Dropout(.5))
# model.add(ELU())
# model.add(Dense(1))


# Nvidia model
model = Sequential()

model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape = (160, 320, 3)))
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))

model.add(Convolution2D(24, 5, 5, subsample=(2, 2)))
model.add(Activation('relu'))

model.add(Convolution2D(36, 5, 5, subsample=(2, 2)))
model.add(Activation('relu'))

model.add(Convolution2D(48, 5, 5, subsample=(2, 2)))
model.add(Activation('relu'))

model.add(Convolution2D(64, 3, 3, subsample=(1, 1)))
model.add(Activation('relu'))

model.add(Convolution2D(64, 3, 3, subsample=(1, 1))) 
model.add(Activation('relu'))

model.add(Flatten())    

model.add(Dense(100))
model.add(Activation('relu'))

model.add(Dense(50))
model.add(Activation('relu'))

model.add(Dense(10))
model.add(Activation('relu'))

model.add(Dense(1))

print (model.summary())
model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, \
            steps_per_epoch=math.ceil(len(train_samples)/batch_size), \
            validation_data=validation_generator, \
            validation_steps=math.ceil(len(validation_samples)/batch_size), \
            epochs=5, verbose=1)

model.save("./output_model/model_nvidia_v2.h5")

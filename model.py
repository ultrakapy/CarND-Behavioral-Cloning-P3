import os
import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Convolution2D, \
    MaxPooling2D, Activation, Dropout

samples = []
line_num = 0
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        if line_num > 0:
            samples.append(line)
        line_num += 1

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

import cv2
import numpy as np
import sklearn

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                # all 3 images and steering angles
                center_image_name = 'data/IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(center_image_name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

                correction = 0.4
                left_angle = center_angle + correction
                right_angle = center_angle - correction

                left_image_name = 'data/IMG/'+batch_sample[1].split('/')[-1]
                left_image = cv2.imread(left_image_name)
                images.append(left_image)
                angles.append(left_angle)

                right_image_name = 'data/IMG/'+batch_sample[2].split('/')[-1]
                right_image = cv2.imread(right_image_name)
                images.append(right_image)
                angles.append(right_angle)

                # flipped versions of the original 3 images and steering angles
                center_image_flipped = cv2.flip(center_image, 1)
                center_angle_flipped = -1.0*center_angle
                images.append(center_image_flipped)
                angles.append(center_angle_flipped)

                left_image_flipped = cv2.flip(left_image, 1)
                left_angle_flipped = -1.0*left_angle
                images.append(left_image_flipped)
                angles.append(left_angle_flipped)

                right_image_flipped = cv2.flip(right_image, 1)
                right_angle_flipped = -1.0*right_angle
                images.append(right_image_flipped)
                angles.append(right_angle_flipped)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

#ch, row, col = 3, 80, 320  # Trimmed image format
ch, row, col = 3, 160, 320

model = Sequential()
model.add(Lambda(lambda x: x / 255 - 0.5,
                 input_shape=(row, col, ch),
                 output_shape=(row, col, ch)))
model.add(Cropping2D(cropping=((70, 25), (0, 0))))

# NVIDIA model
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(48,5,5,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Dense(1))

# LeNet model
#model.add(Convolution2D(6, 5, 5))
#model.add(MaxPooling2D())
#model.add(Activation('relu'))
#model.add(Convolution2D(6, 5, 5))
#model.add(MaxPooling2D())
#model.add(Activation('relu'))
#model.add(Flatten())
#model.add(Dense(120))
#model.add(Dense(84))
#model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator, samples_per_epoch= \
                    len(train_samples), validation_data=validation_generator, \
                    nb_val_samples=len(validation_samples), nb_epoch=4)

model.save('model.h5')

### plot the training and validation loss for each epoch
#from keras.models import Model
#import matplotlib.pyplot as plt

#plt.plot(history_object.history['loss'])
#plt.plot(history_object.history['val_loss'])
#plt.title('model mean squared error loss')
#plt.ylabel('mean squared error loss')
#plt.xlabel('epoch')
#plt.legend(['training set', 'validation set'], loc='upper right')
#plt.show()

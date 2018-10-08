#!/home/eudie/miniconda3/envs/carnano/bin/python
# -*- coding: utf-8 -*-
# Author: Eudie

"""
This file is for Behavioural Cloning project of Udacity Self Driving Car Nanodegree. Here we are creating keras model
to mimic steering control to keep simulated car at the center of the road.
"""

# Required Modules
import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from sklearn.model_selection import train_test_split
from random import shuffle
import sklearn


# Reading driving log as list
lines = []

with open('../Behavioral_Cloning-Raw-FIles/extra/data/data/driving_log.csv') as csv_file:
    reader = csv.reader(csv_file)
    for line in reader:
        lines.append(line)

lines = lines[1:]  # Because first line is column names


train_samples, validation_samples = train_test_split(lines, test_size=0.2)


def generator(lines, batch_size=32):
    """
    To feed data in keras by batch
    :param lines: driving log lines
    :param batch_size: number of lines want to pick for each iteration
    :return: pair of image array and steering measurements
    """
    correction_factor = 0.2

    num_lines = len(lines)
    reminder = num_lines % batch_size
    lines = lines[:-reminder]  # To make length of list list multiple of batch size
    num_lines = len(lines)
    while 1:
        shuffle(lines)
        for offset in range(0, num_lines, batch_size):
            batch_lines = lines[offset:offset+batch_size]

            images = []
            measurements = []

            for i in range(batch_size):
                line = batch_lines[i]

                # For center image
                source_path_center = line[0]
                filename_center = source_path_center.split('/')[-1]
                current_path_center = '../Behavioral_Cloning-Raw-FIles/extra/data/data/IMG/' + filename_center
                image_center = cv2.imread(current_path_center)
                images.append(image_center)
                measurement_center = float(line[3])
                measurements.append(measurement_center)

                # For left image
                source_path_left = line[1]
                filename_left = source_path_left.split('/')[-1]
                current_path_left = '../Behavioral_Cloning-Raw-FIles/extra/data/data/IMG/' + filename_left
                image_left = cv2.imread(current_path_left)
                images.append(image_left)
                measurement_left = measurement_center + correction_factor
                measurements.append(measurement_left)

                # For right image
                source_path_right = line[2]
                filename_right = source_path_right.split('/')[-1]
                current_path_right = '../Behavioral_Cloning-Raw-FIles/extra/data/data/IMG/' + filename_right
                image_right = cv2.imread(current_path_right)
                images.append(image_right)
                measurement_right = measurement_center - correction_factor
                measurements.append(measurement_right)

            augmented_image, augmented_measurement = [], []

            for image, measurement in zip(images, measurements):
                augmented_image.append(image)
                augmented_measurement.append(measurement)
                augmented_image.append(cv2.flip(image, 1))
                augmented_measurement.append(measurement * -1.0)

            x_train = np.array(augmented_image)
            y_train = np.array(augmented_measurement)

            yield sklearn.utils.shuffle(x_train, y_train)


train_generator = generator(train_samples, batch_size=16)
validation_generator = generator(validation_samples, batch_size=16)

# Building model
model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))  # Normalizing
model.add(Cropping2D(cropping=((70, 25), (0, 0))))  # Because main features are in the bottom half of the image

# Convolution layers
model.add(Convolution2D(8, 5, 5, activation='relu'))
model.add(MaxPooling2D())

model.add(Convolution2D(32, 5, 5, activation='relu'))
model.add(MaxPooling2D())

model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(MaxPooling2D())

model.add(Convolution2D(128, 3, 3, activation='relu'))
model.add(MaxPooling2D())

# Fully connected layers
model.add(Flatten())
model.add(Dense(512))
model.add(Dropout(0.8))
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(1))


model.compile(loss='mse', optimizer='adam')
print(model.summary())
quit()
model.fit_generator(train_generator, samples_per_epoch=len(train_samples),
                    validation_data=validation_generator,
                    nb_val_samples=len(validation_samples), nb_epoch=1)

model.save('model.h5')

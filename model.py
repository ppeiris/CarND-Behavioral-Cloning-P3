"""
Behavioral cloning
Prabath Peiris
peiris.prabath@gmail.com
Udacity : Self Driving Car (Term 1)
"""

import os
import csv
import pandas as pd
import numpy as np
import cv2
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import sklearn

# Configs
dloc = [
        # "data/set_1_steering/data",
        # "data/set_2_moredata_steering/data",
        # "data/data_track1_center_counter_drive/data",
        "data/data_track1_center_drive/data",
        "data/track1_center",
        "data/track1_smooth_corners",
        "data/track1_recovery_from_sides",
        "data/track1_counterclock",
        # "data/track2_center"
    ]






dcolumns = [
    'image_center',
    'image_left',
    'image_right',
    'steering_center',
    'throttle',
    'brake',
    'speed'
]

""" load data

Load data from the log file in to a dataframe
"""
def loadData():

    totalData = pd.DataFrame(columns=dcolumns)
    for loc in dloc:
        # print(loc)
        # Load the data log file in to a dataframe
        data = pd.read_csv("%s/driving_log.csv" %(loc), names=dcolumns)

        # Change the image path locations
        data['image_center'] = data['image_center'].apply(lambda x: "%s/IMG/%s" %(loc,x.split('/')[-1]))
        data['image_left'] = data['image_left'].apply(lambda x: "%s/IMG/%s" % (loc,x.split('/')[-1]))
        data['image_right'] = data['image_right'].apply(lambda x: "%s/IMG/%s" % (loc,x.split('/')[-1]))
        # steeting for left image
        data['steering_left'] = data['steering_center']
        # steering for right image
        data['steering_right'] = data['steering_center']
        totalData = pd.concat([totalData, data])


    # shuffle the data
    totalData = shuffle(totalData)
    # make some adjustment to the steeting
    totalData = steeringAdjustments(totalData)
    # print(data.tail())
    # print(len(totalData))
    # print('-----------------------')
    return totalData


def getSamples():
    data = loadData()
    train_samples, validation_samples = train_test_split(data, test_size=0.2)
    return [train_samples, validation_samples]

def generator(samples, batch_size=32):
    num_samples = len(samples)

    while 1:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            # load all images
            # center image
            image_center = batch_samples["image_center"].apply(lambda x: cv2.imread(x))
            # left image
            image_left = batch_samples["image_left"].apply(lambda x: cv2.imread(x))
            # right image
            image_right = batch_samples["image_right"].apply(lambda x: cv2.imread(x))
            # flip the center image
            image_center_flip = batch_samples["image_center"].apply(lambda x: np.fliplr(cv2.imread(x)))

            images = list(image_center) + \
                        list(image_left) + \
                        list(image_right) + \
                        list(image_center_flip)

            lables = list(batch_samples["steering_center"]) + \
                        list(batch_samples["steering_left"]) + \
                        list(batch_samples["steering_right"]) + \
                        list(-batch_samples["steering_center"])

            yield [np.array(images), np.array(lables)]



""" make adjustment to steeting angle

for left and right images we have to make some adjustment for the steeting angle
"""
def steeringAdjustments(data):
    data['steering_left'] += 0.2
    data['steering_right'] -= 0.2
    return data


def getNvideaModel():

    model = Sequential()
    # normalize images
    model.add(Lambda(lambda x: (x/255.0) - 0.5, input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((70, 25),(0,0))))
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation="relu"))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation="relu"))
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation="relu"))
    model.add(Convolution2D(64, 3, 3, activation="relu"))
    model.add(Convolution2D(64, 3, 3, activation="relu"))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    return model


def main():
    model = getNvideaModel()
    train_samples, validation_samples = getSamples()

    train_generator = generator(train_samples, batch_size=32)
    validation_generator = generator(validation_samples, batch_size=32)



    model.fit_generator(
        train_generator,
        samples_per_epoch=len(train_samples),
        validation_data=validation_generator,
        nb_val_samples=len(validation_samples),
        nb_epoch=3
    )

    model.save('model_steering_nvidia_new.h5')

if __name__ == '__main__':
    main()


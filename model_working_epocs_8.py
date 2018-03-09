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
from keras.layers import Flatten, Dropout, Dense, Lambda, Cropping2D, BatchNormalization
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import sklearn

# Training data
dloc = [
    "newdata/track1_center",
    "newdata/track1_corners",
    "newdata/track1_counter_center",
    "newdata/track1_counter_corners",
    "newdata/track1_recovery_corners",
    # "data/data_samples",
    "data/track2_center",
    "data/track1_center",
    "data/track1_smooth_corners",
    "data/track1_recovery_from_sides",
    "data/track1_counterclock",
    "data/track2_center",
    "newdata/track1_center_2",
    # "data/track1_more_corners",
    # "data/track1_counter_corners"
];

# columns for pandas dataframe
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

Load data from the log files in to a dataframe
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

    return totalData

"""

split the data in to training and validation sets
"""
def getSamples():
    data = loadData()
    train_samples, validation_samples = train_test_split(data, test_size=0.2)
    return [train_samples, validation_samples]

"""

generator that return the image data as an numpy array
"""
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


"""

Deep Learning Network
"""
def getNvideaModel():

    model = Sequential()
    # normalize images
    model.add(Lambda(lambda x: (x/255.0) - 0.5, input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((70, 25),(0,0))))
    # model.add(BatchNormalization())
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation="relu"))
    # model.add(BatchNormalization())
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation="relu"))
    # model.add(BatchNormalization())
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation="relu"))
    # model.add(BatchNormalization())
    model.add(Convolution2D(64, 3, 3, activation="relu"))
    # model.add(BatchNormalization())
    model.add(Convolution2D(64, 3, 3, activation="relu"))
    # model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(100))
    # model.add(Dropout(0.8))
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

    history_object = model.fit_generator(
        train_generator,
        samples_per_epoch=len(train_samples),
        validation_data=validation_generator,
        nb_val_samples=len(validation_samples),
        nb_epoch=3
    )

    model.save('model_2.h5')


    print(history_object.history.keys())
    print('Loss')
    print(history_object.history['loss'])
    print('Validation Loss')
    print(history_object.history['val_loss'])


if __name__ == '__main__':
    main()


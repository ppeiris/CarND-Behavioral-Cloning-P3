import csv
import pandas as pd
import numpy as np
import cv2
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D, MaxPooling2D

# load the data from csv file

data = pd.read_csv(
	"data/set_2_moredata_steering/data/driving_log.csv",
	names=['image_center', 'image_left', 'image_right', 'steering', 'throttle', 'brake', 'speed']
);

# correct the image path
# load the image path to an array
data['image_center']=data['image_center'].apply(lambda x: 'data/set_2_moredata_steering/data/IMG/' + x.split('/')[-1])
data['image_left']=data['image_left'].apply(lambda x: 'data/set_2_moredata_steering/data/IMG/' + x.split('/')[-1])
data['image_right']=data['image_right'].apply(lambda x: 'data/set_2_moredata_steering/data/IMG/' + x.split('/')[-1])

# load steering data to an array
# data is in the 'steeting'

# load images
# getting the center image
images_center_1 = data['image_center'].apply(lambda x: cv2.imread(x))
# flip the center image
images_center_2 = data['image_center'].apply(lambda x: np.fliplr(cv2.imread(x)))

# left images
images_left_1 = data['image_left'].apply(lambda x: cv2.imread(x))

# right images
images_right_1 = data['image_right'].apply(lambda x: cv2.imread(x))

images = list(images_center_1) + list(images_center_2) + list(images_left_1) + list(images_right_1)

# print(type(images[0]))
lables_left = data['steering'] + 0.2
lables_right = data['steering'] - 0.2

lables = list(data['steering']) + list(-data['steering']) + list(lables_left) + list(lables_right)

# print(lables)

# convert data to numpy arrays
x_train = np.array(images)
y_train = np.array(lables)
# print(x_train[0].shape)

# create a model
# print(x_train.shape)
# print(y_train.shape)


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
model.fit(
	x_train,
	y_train,
	nb_epoch=3,
	validation_split=0.2,
	shuffle=True,
	verbose=1
)

model.save('model_steering_nvidia.h5')



# train the model

# save the model



import csv
import pandas as pd 
import numpy as np 
import cv2
from keras.models import Sequential 
from keras.layers import Flatten, Dense 



''' 
# load the data from csv file 

data = pd.read_csv(
	"data/set_1_steering/data/driving_log.csv",
	names=['image_center', 'image_left', 'image_right', 'steering', 'throttle', 'brake', 'speed']
);

# correct the image path 
# load the image path to an array 
data['image_center']=data['image_center'].apply(lambda x: 'data/set_1_steering/data/IMG/' + x.split('/')[-1])
data['image_left']=data['image_left'].apply(lambda x: 'data/set_1_steering/data/IMG/' + x.split('/')[-1])
data['image_right']=data['image_right'].apply(lambda x: 'data/set_1_steering/data/IMG/' + x.split('/')[-1])

# load steering data to an array 
# data is in the 'steeting'

# load images 
images = data['image_center'].apply(lambda x: cv2.imread(x))
# print(type(images[0]))

lables = data['steering']
# print(lables)

# convert data to numpy arrays 
x_train = np.array(images)
y_train = np.array(lables)
# print(x_train[0].shape)

# create a model 


print(x_train[0].shape)

print(y_train.shape)
'''



lines = []
with open('data/set_1_steering/data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)


images = []
measurements = []

for line in lines:
	spath = line[0]
	filename = spath.split('/')[-1]
	cpath = 'data/set_1_steering/data/IMG/' + filename 
	image = cv2.imread(cpath)
	images.append(image)
	m = float(line[3])
	measurements.append(m)


x_train = np.array(images)
y_train = np.array(measurements)


print(x_train.shape)
print(y_train.shape)



'''
model = Sequential()
model.add(Flatten(input_shape=(160, 320, 3)))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, validation_split=0.2, shuffle=True)
'''
# train the model 

# save the model



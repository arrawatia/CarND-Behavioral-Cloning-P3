import csv

import cv2
import numpy as np
import os

# Get csv data

DATA_DIR = "data"


def load_data(data_dir):
    """
    Reads the dataset. Assumes the following layout
    data_dir
        driving_log.csv
        IMG
            left_2016_12_01_13_39_25_499.jpg
            ...

    Returns a list of images and a list of corresponding steering angle measurements
    """
    lines = []
    with open(os.path.join(DATA_DIR, 'driving_log.csv')) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

    images = []
    measurements = []

    for line in lines[1:]:
        source_path = line[0]
        filename = source_path.split("/")[-1]
        current_path = os.path.join(DATA_DIR, "IMG", filename)
        image = cv2.imread(current_path)
        images.append(image)
        measurement = float(line[3])
        measurements.append(measurement)

    return images, measurements


images, measurements = load_data(DATA_DIR)
X_train = np.array(images)
y_train = np.array(measurements)


from keras.models import Sequential
from keras.layers import Flatten, Dense

model = Sequential()
model.add(Flatten(input_shape=(160, 320, 3)))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=2)

model.save('model.h5')
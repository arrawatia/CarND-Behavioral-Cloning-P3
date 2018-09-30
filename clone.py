# Get csv data
from keras.optimizers import Adam
import cv2
import numpy as np
import os
import csv

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

augmented_images, augmented_measurements = [], []
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)

    augmented_images.append(cv2.flip(image, 1))
    augmented_measurements.append(measurement * -1.0)

X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)




samples = []
with open(os.path.join(DATA_DIR, 'driving_log.csv')) as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split

train_samples, validation_samples = train_test_split(samples, test_size=0.2)

import cv2
import numpy as np
import sklearn


def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:  # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = './IMG/' + batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Convolution2D, ELU, Dropout, Activation


def model():
    # ch, row, col = 3, 80, 320  # Trimmed image format
    ch, row, col = 160, 320, 3

    model = Sequential()
    # Normalize (divide by 255) and
    # mean center the images (substract 0.5 to make them range from [-0.5, 0.5] instead of [0, 1]
    model.add(Lambda(lambda x: x / 127.5 - 1., input_shape=(ch, row, col), output_shape=(ch, row, col)))
    # model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((70, 25), (0, 0)), input_shape=(3, 160, 320)))

    model.add(Flatten(input_shape=(160, 320, 3)))
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')

    return model


# https://github.com/commaai/research/blob/master/train_steering_model.py
def comma_ai_model():
    # ch, row, col = 3, 160, 320  # camera format
    ch, row, col = 160, 320, 3

    model = Sequential()
    model.add(Lambda(lambda x: x / 127.5 - 1., input_shape=(ch, row, col), output_shape=(ch, row, col)))

    # model.add(Lambda(lambda x: x / 127.5 - 1.,
    #                  input_shape=(ch, row, col),
    #                  output_shape=(ch, row, col)))
    model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Flatten())
    model.add(Dropout(.2))
    model.add(ELU())
    model.add(Dense(512))
    model.add(Dropout(.5))
    model.add(ELU())
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mse")

    return model


def nvidia_model():
    # ch, row, col = 3, 80, 320  # Trimmed image format
    ch, row, col = 160, 320, 3
    model = Sequential()
    # model.add(Lambda(lambda x: x / 255 - 0.5, input_shape=(160, 320, 3)))
    model.add(Lambda(lambda x: x / 127.5 - 1., input_shape=(ch, row, col), output_shape=(ch, row, col)))
    model.add(Cropping2D(cropping=((70, 25), (0, 0))))
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation="relu"))
    # model.add(Conv2D(24, (5, 5), strides=(2, 2), kernel_regularizer='l2', dim_ordering='tf'))
    # model.add(Activation('relu'))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation="relu"))

    # model.add(Conv2D(36, (5, 5), strides=(2, 2), kernel_regularizer='l2', dim_ordering='tf'))
    # model.add(Activation('relu'))
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation="relu"))

    # model.add(Conv2D(48, (5, 5), strides=(2, 2), kernel_regularizer='l2', dim_ordering='tf'))
    # model.add(Activation('relu'))

    model.add(Convolution2D(64, 3, 3, activation="relu"))

    # model.add(Conv2D(64, (3, 3), strides=(1, 1), kernel_regularizer='l2', dim_ordering='tf'))
    # model.add(Activation('relu'))

    model.add(Convolution2D(64, 3, 3, activation="relu"))

    # model.add(Conv2D(64, (3, 3), strides=(1, 1), kernel_regularizer='l2', dim_ordering='tf'))
    # model.add(Activation('relu'))

    model.add(Flatten())
    model.add(Dense(100, name='fc1'))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))

    model.add(Dense(50, name='fc2'))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))

    model.add(Dense(10, name='fc3'))
    model.add(Activation('relu'))
    model.add(Dense(1, name='output'))

    # for a mean squared error regression problem
    model.compile(optimizer=Adam(lr=0.0001), loss='mean_squared_error')

    return model


model = comma_ai_model()

model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=2, verbose=1)

model.save('comma_ai_model.h5')

model = nvidia_model()

model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=2, verbose=1)

model.save('nvidia_model.h5')


# import matplotlib.pyplot as plt
#
# history_object = model.fit_generator(train_generator,
#                                      samples_per_epoch=len(train_samples),
#                                      validation_data=validation_generator,
#                                      nb_val_samples=len(validation_samples),
#                                      nb_epoch=5, verbose=1)


### print the keys contained in the history object
# print(history_object.history.keys())
#
# ### plot the training and validation loss for each epoch
# plt.plot(history_object.history['loss'])
# plt.plot(history_object.history['val_loss'])
# plt.title('model mean squared error loss')
# plt.ylabel('mean squared error loss')
# plt.xlabel('epoch')
# plt.legend(['training set', 'validation set'], loc='upper right')
# plt.show()

# Get csv data
import csv
import os

import cv2
import numpy as np
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

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

    return lines[1:]


def shuffle_and_split(lines):
    ## shuffle the observations
    lines_shuffled = shuffle(lines)

    ## split into training and validation data sets
    train_lines, validation_lines = train_test_split(lines_shuffled, test_size=0.2)

    return np.array(train_lines), np.array(validation_lines)


def parse(lines):
    images = []
    measurements = []

    steering_correction = 0.2

    for line in lines:
        center = os.path.join(DATA_DIR, "IMG", line[0].split("/")[-1])
        left = os.path.join(DATA_DIR, "IMG", line[1].split("/")[-1])
        right = os.path.join(DATA_DIR, "IMG", line[2].split("/")[-1])

        # Based on NVidia paper
        color_conversion = cv2.COLOR_BGR2YUV

        images.append(cv2.cvtColor(cv2.imread(center), color_conversion))
        images.append(cv2.cvtColor(cv2.imread(left), color_conversion))
        images.append(cv2.cvtColor(cv2.imread(right), color_conversion))

        measurements.append(float(line[3]))
        measurements.append(float(line[3]) + steering_correction)
        measurements.append(float(line[3]) - steering_correction)

    return images, measurements


def distribute_data(observations, min_needed=500, max_needed=750):
    ''' create a relatively uniform distribution of images
    Arguments
        observations: the array of observation data that comes from the read input function
        min_needed: minimum number of observations needed per bin in the histogram of steering angles
        max_needed:: maximum number of observations needed per bin in the histogram of steering angles
    Returns
        observations_output: output of augmented data observations
    '''

    observations_output = np.asarray(observations.copy())

    print(observations_output[70:100])
    print(type(observations_output[:]))
    print(observations_output[:, 3])
    ## create histogram to know what needs to be added
    steering_angles = np.asarray(observations_output[:, 3], dtype='float')
    print(steering_angles)
    num_hist, idx_hist = np.histogram(steering_angles, 20)

    to_be_added = np.empty([1, 7])
    to_be_deleted = np.empty([1, 1])

    for i in range(1, len(num_hist)):
        if num_hist[i - 1] < min_needed:

            ## find the index where values fall within the range
            match_idx = np.where((steering_angles >= idx_hist[i - 1]) & (steering_angles < idx_hist[i]))[0]

            ## randomly choose up to the minimum needed
            need_to_add = observations_output[np.random.choice(match_idx, min_needed - num_hist[i - 1]), :]

            to_be_added = np.vstack((to_be_added, need_to_add))

        elif num_hist[i - 1] > max_needed:

            ## find the index where values fall within the range
            match_idx = np.where((steering_angles >= idx_hist[i - 1]) & (steering_angles < idx_hist[i]))[0]

            ## randomly choose up to the minimum needed
            to_be_deleted = np.append(to_be_deleted, np.random.choice(match_idx, num_hist[i - 1] - max_needed))

    ## delete the randomly selected observations that are overrepresented and append the underrepresented ones
    observations_output = np.delete(observations_output, to_be_deleted, 0)
    observations_output = np.vstack((observations_output, to_be_added[1:, :]))

    return observations_output

def augment(images, measurements):
    augmented_images, augmented_measurements = [], []
    for image, measurement in zip(images, measurements):
        augmented_images.append(image)
        augmented_measurements.append(measurement)

        augmented_images.append(cv2.flip(image, 1))
        augmented_measurements.append(measurement * -1.0)

    return augmented_images, augmented_measurements




import cv2
import numpy as np
import sklearn


def generator(samples, batch_size=128):
    num_samples = len(samples)
    while 1:  # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]

            images, measurements = parse(batch_samples)
            augmented_images, augmented_measurements = augment(images, measurements)

            X_train = np.array(augmented_images)
            y_train = np.array(augmented_measurements)
            yield sklearn.utils.shuffle(X_train, y_train)



from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Convolution2D, ELU, Dropout, Activation, Conv2D


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
    model.add(Cropping2D(cropping=((70, 25), (0, 0))))

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
    row, col, ch = 160, 320, 3
    model = Sequential()
    # model.add(Lambda(lambda x: x / 255 - 0.5, input_shape=(160, 320, 3)))
    model.add(Lambda(lambda x: x / 127.5 - 1., input_shape=(row, col, ch), output_shape=(row, col, ch)))
    model.add(Cropping2D(cropping=((70, 25), (0, 0))))
    # model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation="relu"))
    model.add(Conv2D(24, (5, 5), strides=(2, 2), kernel_regularizer='l2', dim_ordering='tf'))
    model.add(Activation('relu'))
    # model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation="relu"))

    model.add(Conv2D(36, (5, 5), strides=(2, 2), kernel_regularizer='l2', dim_ordering='tf'))
    model.add(Activation('relu'))
    # model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation="relu"))
    #
    model.add(Conv2D(48, (5, 5), strides=(2, 2), kernel_regularizer='l2', dim_ordering='tf'))
    model.add(Activation('relu'))
    #
    # model.add(Convolution2D(64, 3, 3, activation="relu"))
    #
    model.add(Conv2D(64, (3, 3), strides=(1, 1), kernel_regularizer='l2', dim_ordering='tf'))
    model.add(Activation('relu'))
    #
    # model.add(Convolution2D(64, 3, 3, activation="relu"))
    #
    model.add(Conv2D(64, (3, 3), strides=(1, 1), kernel_regularizer='l2', dim_ordering='tf'))
    model.add(Activation('relu'))

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

images, measurements = parse(load_data(DATA_DIR)[1:100])
print("image size = ", images[1].shape)
# augmented_images, augmented_measurements = augment(images, measurements)
# X_train = np.array(augmented_images)
# y_train = np.array(augmented_measurements)


data = load_data(DATA_DIR)
batch_size = 128
train_samples, validation_samples = train_test_split(data)
# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

# model = comma_ai_model()

# model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=1, verbose=1)
#
# history_object = model.fit_generator(train_generator,
#                                      steps_per_epoch=len(train_samples)/batch_size,
#                                      validation_data=validation_generator,
#                                      validation_steps=len(validation_samples)/batch_size,
#                                      epochs=10, verbose=1)
#
# model.save('comma_ai_model.h5')
#
model = nvidia_model()

# model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=1, verbose=1)

history_object = model.fit_generator(train_generator,
                                     steps_per_epoch=len(train_samples) / batch_size,
                                     validation_data=validation_generator,
                                     validation_steps=len(validation_samples) / batch_size,
                                     epochs=10, verbose=1)

model.save('nvidia_model.h5')

# import matplotlib.pyplot as plt
#
# history_object = model.fit_generator(train_generator,
#                                      samples_per_epoch=len(train_samples),
#                                      validation_data=validation_generator,
#                                      nb_val_samples=len(validation_samples),
#                                      nb_epoch=5, verbose=1)
#
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

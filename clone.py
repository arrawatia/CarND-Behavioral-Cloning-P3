import sklearn
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Convolution2D, ELU, Dropout, Activation, Conv2D
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import numpy as np

from data import *


def shuffle_and_split(lines):
    # shuffle the observations
    lines_shuffled = shuffle(lines)

    # split into training and validation data sets
    train_lines, validation_lines = train_test_split(lines_shuffled, test_size=0.2)

    return np.array(train_lines), np.array(validation_lines)


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


def simple_model():
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


def main():
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

    # print the keys contained in the history object
    print(history_object.history.keys())

    import matplotlib.pyplot as plt

    # plot the training and validation loss for each epoch
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.savefig("training_log.jpeg")
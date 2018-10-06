import csv
import os
import sklearn

import cv2

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


def augment(images, measurements):
    augmented_images, augmented_measurements = [], []
    for image, measurement in zip(images, measurements):
        augmented_images.append(image)
        augmented_measurements.append(measurement)

        augmented_images.append(cv2.flip(image, 1))
        augmented_measurements.append(measurement * -1.0)

    return augmented_images, augmented_measurements


import numpy as np
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


def main():

        fig, axes = plt.subplots(2, 1, figsize=(8, 8))
        plt.subplots_adjust(left=0, right=0.95, top=0.9, bottom=0.25)
        ax0, ax1 = axes.flatten()

        ax0.hist(y_train, bins=n_classes, histtype='bar', color='blue', rwidth=0.6, label='train')
        ax0.set_title('Number of training')
        ax0.set_xlabel('Steering Angle')
        ax0.set_ylabel('Total Image')

        ax1.hist(y_valid, bins=n_classes, histtype='bar', color='red', rwidth=0.6, label='valid')
        ax1.set_title('Number of validation')
        ax1.set_xlabel('Steering Angle')
        ax1.set_ylabel('Total Image')

        fig.tight_layout()
        plt.show()

    # show
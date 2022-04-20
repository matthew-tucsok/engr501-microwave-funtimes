import numpy as np
import glob
import os
import pandas as pd
from PIL import Image
import string
import random
import tensorflow as tf
from sklearn.preprocessing import normalize


class MicrowaveDataloader:
    def __init__(self, source='./data/microwave_dataset', window_size=100, batch_size=1):
        self.source = source
        self.window_size = window_size
        self.classes = ['water', 'ethanol']
        materials = glob.glob(self.source + '/*/')

        x_train = []
        y_train = []
        x_val = []
        y_val = []
        x_test = []
        y_test = []
        for material in materials:
            class_count = 0
            liquids = glob.glob(material + '/*/')
            for liquid in liquids:
                liquid_path_split = liquid.split('\\')
                liquid_name = liquid_path_split[-2]
                liquid_class = self.classes.index(liquid_name)
                positions = glob.glob(liquid + '/*/')
                for position in positions:
                    runs = glob.glob(position + '*.csv')
                    train_count = 0
                    val_count = 0
                    test_count = 0
                    for run in runs:
                        if train_count < 6:
                            x_train.append(load_csv(run))  # Returns each csv as a 2d matrix with first column as frequencies and rest are sensor readings
                            y_train.append(int(liquid_class))
                            train_count += 1
                            continue

                        if val_count < 2:
                            x_val.append(load_csv(run))  # Returns each csv as a 2d matrix with first column as frequencies and rest are sensor readings
                            y_val.append(int(liquid_class))
                            val_count += 1
                            continue

                        if test_count < 2:
                            x_test.append(load_csv(run))  # Returns each csv as a 2d matrix with first column as frequencies and rest are sensor readings
                            y_test.append(int(liquid_class))
                            test_count += 1
                            continue

        train_data_pairs = list(zip(x_train, y_train))
        random.shuffle(train_data_pairs)
        x_train, y_train = (zip(*train_data_pairs))

        val_data_pairs = list(zip(x_val, y_val))
        random.shuffle(val_data_pairs)
        x_val, y_val = zip(*val_data_pairs)

        test_data_pairs = list(zip(x_test, y_test))
        random.shuffle(test_data_pairs)
        x_test, y_test = zip(*test_data_pairs)

        x_train = np.array(x_train).astype('float32')
        y_train = np.array(y_train).astype('float32')
        x_val = np.array(x_val).astype('float32')
        y_val = np.array(y_val).astype('float32')
        x_test = np.array(x_test).astype('float32')
        y_test = np.array(y_test).astype('float32')

        self.train_set = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        self.train_set = self.train_set.batch(batch_size)
        self.val_set = tf.data.Dataset.from_tensor_slices((x_val, y_val))
        self.val_set = self.val_set.batch(batch_size)
        self.test_set = tf.data.Dataset.from_tensor_slices((x_test, y_test))
        self.test_set = self.val_set.batch(batch_size)


def load_csv(path):
    data = pd.read_csv(path, skiprows=7)
    data = data.drop(index=data.index[-1], axis=0)
    numpy_data = data.to_numpy()
    numpy_data = np.delete(numpy_data, 0, axis=1)
    # for row in range(numpy_data.shape[0]):
        # silly_string = numpy_data[row][0]
        # freq = float(silly_string.strip(string.ascii_letters))
        # numpy_data[row][0] = freq
    numpy_data = normalize(numpy_data, axis=1, norm='l2')
    proper_data_size = np.empty((2000, 32, 1))
    proper_data_size[:, :, 0] = numpy_data
    return proper_data_size

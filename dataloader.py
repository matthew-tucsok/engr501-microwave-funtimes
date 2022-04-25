"""
ENGR 501 Course Project
Matthew Tucsok 37924164
Jack McClelland 31046162

Dataloader to load, reformat, normalize, shuffle, and divide our data correctly.
"""

import numpy as np
import glob
import pandas as pd
import random
import tensorflow as tf
from sklearn.preprocessing import normalize


class MicrowaveDataloader:
    def __init__(self, source='./data/microwave_dataset', batch_size=1):
        self.source = source
        self.classes = ['water', 'ethanol']  # These are the classes we have in our current dataset
        materials = glob.glob(self.source + '/*/')

        x_train = []
        y_train = []
        x_val = []
        y_val = []
        x_test = []
        y_test = []
        """
        The following nested for loops navigate our dataset filesystem and ensure an equal number of each class ends up
        in train/val/test sets. 
        """
        for material in materials:
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
                        if train_count < 6:  # Since we know there are 10 rotations in a position, 60 percent is 6/10
                            x_train.append(load_csv(run))  # Returns each csv as a 2d matrix with first column as frequencies and rest are sensor readings
                            y_train.append(int(liquid_class))
                            train_count += 1
                            continue

                        if val_count < 2:  # Since we know there are 10 rotations in a position, 20 percent is 2/10
                            x_val.append(load_csv(run))  # Returns each csv as a 2d matrix with first column as frequencies and rest are sensor readings
                            y_val.append(int(liquid_class))
                            val_count += 1
                            continue

                        if test_count < 2:  # Since we know there are 10 rotations in a position, 20 percent is 2/10
                            x_test.append(load_csv(run))  # Returns each csv as a 2d matrix with first column as frequencies and rest are sensor readings
                            y_test.append(int(liquid_class))
                            test_count += 1
                            continue

        # Shuffling each of the sets of data
        train_data_pairs = list(zip(x_train, y_train))
        random.shuffle(train_data_pairs)
        x_train, y_train = (zip(*train_data_pairs))

        val_data_pairs = list(zip(x_val, y_val))
        random.shuffle(val_data_pairs)
        x_val, y_val = zip(*val_data_pairs)

        test_data_pairs = list(zip(x_test, y_test))
        random.shuffle(test_data_pairs)
        x_test, y_test = zip(*test_data_pairs)

        # Converting data types to work with TensorFlow
        x_train = np.array(x_train).astype('float32')
        y_train = np.array(y_train).astype('float32')
        x_val = np.array(x_val).astype('float32')
        y_val = np.array(y_val).astype('float32')
        x_test = np.array(x_test).astype('float32')
        y_test = np.array(y_test).astype('float32')

        # Converting data sets into TensorFlow-friendly objects
        self.train_set = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        self.train_set = self.train_set.batch(batch_size)
        self.val_set = tf.data.Dataset.from_tensor_slices((x_val, y_val))
        self.val_set = self.val_set.batch(batch_size)
        self.test_set = tf.data.Dataset.from_tensor_slices((x_test, y_test))
        self.test_set = self.test_set.batch(batch_size)


# This function takes care of loading a csv file and ensuring only useful normalized information is kept in a numpy format
def load_csv(path):
    data = pd.read_csv(path, skiprows=7)
    data = data.drop(index=data.index[-1], axis=0)  # Getting rid of the "END" at the bottom of csv file
    numpy_data = data.to_numpy()
    numpy_data = np.delete(numpy_data, 0, axis=1)  # Deleting frequencies
    numpy_data = normalize(numpy_data, axis=1, norm='l2')  # Column-wise normalization of data using L2 nrom
    proper_data_size = np.zeros((2000, 32, 1))
    proper_data_size[:, :, 0] = numpy_data  # Deleting frequency data
    return proper_data_size

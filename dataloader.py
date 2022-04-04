import numpy as np
import glob
import os
import pandas as pd
from PIL import Image
import random
import tensorflow as tf


class MicrowaveDataloader:
    def __init__(self, source='./data/microwave_dataset', window_size=100, percent_train=0.7):
        self.source = source
        self.window_size = window_size
        self.percent_train = percent_train
        self.train_sensor_readings = []
        self.val_sensor_readings = []
        self.train_ground_truths = []
        self.val_ground_truths = []

        materials = glob.glob(self.source + '/*/')
        for material in materials:
            class_count = 0
            liquids = glob.glob(material + '/*/')
            for liquid in liquids:
                positions = glob.glob(liquid + '/*/')
                for position in positions:
                    runs = glob.glob(position + '*.csv')
                    for run in runs:
                        run_data = load_csv(run)  # Returns each csv as a 2d matrix with first column as frequencies and rest are sensor readings
                        


        if train_with_composites:
            print('Loading composite shapes...')
            comp_source = source + '/composites'
            object_folders = next(os.walk(comp_source))[1]
            train_comp_views, val_comp_views, train_comp_ground_truths, val_comp_ground_truths = load_samples(
                comp_source, object_folders, voxel_dims,
                view_size)
            train_sensor_readings += train_comp_views
            val_sensor_readings += val_comp_views
            train_ground_truths += train_comp_ground_truths
            val_ground_truths += val_comp_ground_truths

        if load_single is not None:
            object_folder = load_single.split('/')[-1]
            load_single = load_single.replace('/' + object_folder, '')
            object_folders = [object_folder]
            train_spec_views, val_spec_views, train_spec_ground_truths, val_spec_ground_truths = load_samples(
                load_single, object_folders,
                voxel_dims, view_size)
            train_sensor_readings += train_spec_views
            val_sensor_readings += val_spec_views
            train_ground_truths += train_spec_ground_truths
            val_ground_truths += val_spec_ground_truths
        else:
            prim_source = source + '/primitives'
            object_folders = next(os.walk(prim_source))[1]
            print('Loading primitive shapes...')
            train_prim_views, val_prim_views, train_prim_ground_truths, val_prim_ground_truths = load_samples(
                prim_source, object_folders, voxel_dims,
                view_size)
            train_sensor_readings += train_prim_views
            val_sensor_readings += val_prim_views
            train_ground_truths += train_prim_ground_truths
            val_ground_truths += val_prim_ground_truths

        # if shuffle:
        #     train_data_pairs = list(zip(train_sensor_readings, train_ground_truths))
        #     random.shuffle(train_data_pairs)
        #     train_sensor_readings, train_ground_truths = zip(*train_data_pairs)
        #
        #     val_data_pairs = list(zip(val_sensor_readings, val_ground_truths))
        #     random.shuffle(val_data_pairs)
        #     val_sensor_readings, val_ground_truths = zip(*val_data_pairs)

        x_train = np.array(train_sensor_readings)
        x_val = np.array(val_sensor_readings)
        y_train = np.array(train_ground_truths)
        y_val = np.array(val_ground_truths)
        self.train_set = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        self.train_set = self.train_set.batch(batch_size)
        self.val_set = tf.data.Dataset.from_tensor_slices((x_val, y_val))
        self.val_set = self.val_set.batch(batch_size)


def load_csv(path):
    data = pd.read_csv(path_name, skiprows=7)
    return data.to_numpy()
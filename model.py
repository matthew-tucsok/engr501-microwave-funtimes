"""
ENGR 501 Course Project
Matthew Tucsok 37924164
Jack McClelland 31046162

Creation of deep learning models.
"""

import tensorflow as tf
from tensorflow.keras import layers


# Generic network template useful for inheritance
class GenericNet:
    def __init__(self, weights_path=None):
        self.tf_model = self.create_model()
        self.weights_loaded = False
        if weights_path is not None:
            try:
                self.tf_model.load_weights(weights_path)  # Load in weights if they exist
                print('Successfully loaded existing weights!')
                self.weights_loaded = True
            except Exception as e:
                _ = e
                print('Unable to load existing weights at', weights_path)

    @staticmethod
    def create_model():
        pass


class ConvNet(GenericNet):
    @staticmethod
    def create_model():
        x = layers.Input(shape=(2000, 32, 1))
        fx = layers.Conv2D(filters=1, kernel_size=3, padding='valid')(x)
        fx = layers.BatchNormalization()(fx)
        fx = layers.LeakyReLU(alpha=0.3)(fx)
        fx = layers.Conv2D(filters=2, kernel_size=3, padding='valid')(fx)
        fx = layers.BatchNormalization()(fx)
        fx = layers.LeakyReLU(alpha=0.3)(fx)
        fx = layers.Conv2D(filters=4, kernel_size=3, padding='valid')(fx)
        fx = layers.BatchNormalization()(fx)
        fx = layers.LeakyReLU(alpha=0.3)(fx)
        fx = layers.Conv2D(filters=8, kernel_size=3, padding='valid')(fx)
        fx = layers.BatchNormalization()(fx)
        fx = layers.LeakyReLU(alpha=0.3)(fx)
        fx = layers.Flatten()(fx)
        fx = layers.Dense(16, activation='relu')(fx)
        fx = layers.Dense(1, activation='sigmoid')(fx)
        model = tf.keras.Model(inputs=x, outputs=fx, name='MicrowaveLiquidConvClassifier')
        model.summary()
        return model


class TinyConvNet(GenericNet):
    @staticmethod
    def create_model():
        x = layers.Input(shape=(2000, 32, 1))
        fx = layers.Conv2D(filters=1, kernel_size=3, padding='valid')(x)
        fx = layers.BatchNormalization()(fx)
        fx = layers.LeakyReLU(alpha=0.3)(fx)
        fx = layers.Conv2D(filters=2, kernel_size=3, padding='valid')(fx)
        fx = layers.BatchNormalization()(fx)
        fx = layers.LeakyReLU(alpha=0.3)(fx)
        fx = layers.Conv2D(filters=4, kernel_size=3, padding='valid')(fx)
        fx = layers.BatchNormalization()(fx)
        fx = layers.LeakyReLU(alpha=0.3)(fx)
        fx = layers.Conv2D(filters=8, kernel_size=3, padding='valid')(fx)
        fx = layers.BatchNormalization()(fx)
        fx = layers.LeakyReLU(alpha=0.3)(fx)
        fx = layers.Flatten()(fx)
        fx = layers.Dense(1, activation='sigmoid')(fx)
        model = tf.keras.Model(inputs=x, outputs=fx, name='MicrowaveLiquidConvClassifier')
        model.summary()
        return model


class FCNet(GenericNet):
    @staticmethod
    def create_model():
        x = layers.Input(shape=(2000, 32, 1))
        fx = layers.Dense(10, activation='relu')(x)
        fx = layers.Dense(10, activation='relu')(fx)
        fx = layers.Dense(10, activation='relu')(fx)
        fx = layers.Dense(10, activation='relu')(fx)
        fx = layers.Dense(10, activation='relu')(fx)
        fx = layers.Flatten()(fx)
        fx = layers.Dense(16, activation='relu')(fx)
        fx = layers.Dense(1, activation='sigmoid')(fx)
        model = tf.keras.Model(inputs=x, outputs=fx, name='FCNet')
        model.summary()
        return model


class TinyFCNet(GenericNet):
    @staticmethod
    def create_model():
        x = layers.Input(shape=(2000, 32, 1))
        fx = layers.Dense(10, activation='relu')(x)
        fx = layers.Dense(10, activation='relu')(fx)
        fx = layers.Dense(10, activation='relu')(fx)
        fx = layers.Dense(10, activation='relu')(fx)
        fx = layers.Dense(10, activation='relu')(fx)
        fx = layers.Flatten()(fx)
        fx = layers.Dense(1, activation='sigmoid')(fx)
        model = tf.keras.Model(inputs=x, outputs=fx, name='TinyFCNet')
        model.summary()
        return model

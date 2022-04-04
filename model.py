import tensorflow as tf
from tensorflow.keras import layers


class MicrowaveLSTM:
    def __init__(self, window_size, num_classes, weights_path=None):
        self.window_size = window_size
        self.num_classes = num_classes
        self.tf_model = self.create_model()
        self.weights_loaded = False
        if weights_path is not None:
            try:
                self.tf_model.load_weights(weights_path)
                print('Successfully loaded existing weights!')
                self.weights_loaded = True
            except Exception as e:
                _ = e
                print('Unable to load existing weights at', weights_path)

    def create_model(self):
        x = layers.Input(shape=(self.window_size, 16))
        fx = layers.Conv2D(filters=64, kernel_size=7, padding='same')(x)
        # Put model here
        fx = layers.Dense(self.num_classes, activation='softmax')
        model = tf.keras.Model(inputs=x, outputs=fx, name='MicrowaveLiquidClassifier')
        model.summary()
        return model




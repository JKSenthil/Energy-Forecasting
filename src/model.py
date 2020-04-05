import numpy as np
import tensorflow as tf

tf.keras.backend.set_floatx('float64')

class MLP(tf.keras.Model):
    def __init__(self, n_out, lr=1e-4):
        super(MLP, self).__init__()
        
        # initialize optimizer
        self.opt = tf.keras.optimizers.Adam(learning_rate=lr)

        # initialize neural network layers TODO: change this
        self.dense1 = tf.keras.layers.Dense(128, activation=tf.nn.leaky_relu)
        self.dense2 = tf.keras.layers.Dense(n_out, activation='linear')

    def call(self, inputs, curr_weather):
        """
        inputs will be of shape (batch_size, input_size, sequence_length)
        """
        pass
    
    def loss(self, y_true, y_pred):
        """
        Mean Squared error loss
        """
        return tf.keras.losses.MSE(y_true, y_true)

class RNN(tf.keras.Model):
    def __init__(self, n_out, lr=1e-4, units=256):
        super(RNN, self).__init__()
        
        # initialize optimizer
        self.opt = tf.keras.optimizers.Adam(learning_rate=lr)

        # initialize neural network layers
        self.gru = tf.keras.layers.GRU(units, input_shape=(91, 154))
        self.dense1 = tf.keras.layers.Dense(128, activation=tf.nn.leaky_relu)
        self.dense2 = tf.keras.layers.Dense(n_out, activation='linear')

    def call(self, inputs, curr_weather):
        """
        inputs will be of shape (batch_size, input_size, sequence_length)
        """
        gru_output = self.gru(inputs)
        out1 = self.dense1(tf.concat([gru_output, curr_weather], axis=1))
        return self.dense2(out1)
    
    def loss(self, y_true, y_pred):
        """
        Mean Squared error loss
        """
        return tf.keras.losses.MSE(y_true, y_true)
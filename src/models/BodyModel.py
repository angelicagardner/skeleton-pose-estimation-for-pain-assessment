import tensorflow as tf

from tensorflow.keras import models
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras import optimizers

"""
    This class represents the Body Modality Model. 
    It creates an LSTM that receives skeleton data from PoseNet as input.
"""


class BodyModel():

    def __init__(self, n_timesteps, n_features, n_outputs):
        model = models.Sequential()
        model.add(LSTM(256, input_shape=(
            n_timesteps, n_features)))
        model.add(Dense(1024, activation='relu'))
        model.add(Dense(n_outputs, activation='softmax'))
        lr_schedule = optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.001,
            decay_steps=50,
            decay_rate=0.1)
        optimizer = optimizers.Adam(learning_rate=lr_schedule)
        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizer, metrics=['accuracy'])
        self.model = model

    def model_info(self):
        return self.model.summary()

    def train(self, trainX, trainy, epochs, batch_size):
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='loss', patience=2)
        history = self.model.fit(trainX, trainy, epochs=epochs, callbacks=[early_stopping],
                                 batch_size=batch_size, verbose=2)
        return history

    def evaluate(self, testX, testy, batch_size):
        _, accuracy = self.model.evaluate(
            testX, testy, batch_size=batch_size, verbose=0)
        return accuracy

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.layers import Input, LSTM, Dense, Flatten, TimeDistributed, Conv1D, MaxPooling1D, Concatenate, BatchNormalization
from tensorflow.keras import Model
from tensorflow.keras.initializers import Constant
from keras.layers.advanced_activations import PReLU


class RCNN():
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=50,
        restore_best_weights=True,
    )

    def __init__(self, n_features, second_n_features, n_length, n_outputs, fusion=False):
        if fusion:
            input_1 = Input(shape=(1, n_length, n_features))
            conv1d_1 = TimeDistributed(Conv1D(filters=64, kernel_size=3, activation=PReLU(
                alpha_initializer=Constant(value=0.25))))(input_1)
            bn_1 = TimeDistributed(BatchNormalization())(conv1d_1)
            maxpool_1 = TimeDistributed(MaxPooling1D(
                pool_size=2, data_format='channels_first'))(bn_1)
            flatten_1 = Flatten()(maxpool_1)
            dense_1 = Dense(64)(flatten_1)

            input_2 = Input(shape=(1, n_length, second_n_features))
            conv1d_2 = TimeDistributed(Conv1D(filters=64, kernel_size=3, activation=PReLU(
                alpha_initializer=Constant(value=0.25))))(input_2)
            bn_2 = TimeDistributed(BatchNormalization())(conv1d_2)
            maxpool_2 = TimeDistributed(MaxPooling1D(
                pool_size=2, data_format='channels_first'))(bn_2)
            flatten_2 = Flatten()(maxpool_2)
            dense_2 = Dense(64)(flatten_2)

            concat = Concatenate()([dense_1, dense_2])
            output = Dense(units=n_outputs, activation='sigmoid')(concat)
            model = Model(inputs=[input_1, input_2], outputs=[output])
        else:
            input = Input(shape=(1, n_length, n_features))
            conv1d_1 = TimeDistributed(Conv1D(filters=64, kernel_size=3, activation=PReLU(
                alpha_initializer=Constant(value=0.25))))(input)
            bn = TimeDistributed(BatchNormalization())(conv1d_1)
            conv1d_2 = TimeDistributed(Conv1D(filters=128, kernel_size=3, activation=PReLU(
                alpha_initializer=Constant(value=0.25))))(bn)
            maxpool = TimeDistributed(MaxPooling1D(
                pool_size=2, strides=2, data_format='channels_first'))(conv1d_2)
            flatten = Flatten()(maxpool)
            dense_1 = Dense(256, activation=PReLU(
                alpha_initializer=Constant(value=0.25)))(flatten)
            dense_2 = Dense(512, activation=PReLU(
                alpha_initializer=Constant(value=0.25)))(dense_1)
            output = Dense(units=n_outputs, activation='sigmoid')(dense_2)
            model = Model(inputs=input, outputs=output)
        model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), metrics=['accuracy', tf.keras.metrics.AUC(
        ), tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tfa.metrics.F1Score(num_classes=n_outputs, average='macro')])
        self.model = model

    def train(self, X_train, y_train, X_val, y_val, epochs, batch_size):
        history = self.model.fit(X_train, y_train, validation_data=(
            X_val, y_val), epochs=epochs, batch_size=batch_size, callbacks=[self.early_stopping], verbose=2)
        return history

    def trainFusioned(self, body_X_train, face_X_train, y_train, body_X_val, face_X_val, y_val, epochs, batch_size):
        history = self.model.fit([body_X_train, face_X_train], y_train, validation_data=(
            [body_X_val, face_X_val], y_val), epochs=epochs, batch_size=batch_size, callbacks=[self.early_stopping], verbose=2)
        return history

    def save(self, model_path):
        self.model.save(model_path)

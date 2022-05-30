import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.layers import Input, LSTM, Dense, Flatten, TimeDistributed, Conv1D, MaxPooling1D, Bidirectional, BatchNormalization, Dropout
from tensorflow.keras import Model


class CNNLSTM():
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=25,
        restore_best_weights=True,
    )

    def __init__(self, n_features, n_length, n_outputs, multiclass=False):
        input = Input(shape=(1, n_length, n_features))
        conv1d_1 = Conv1D(filters=128, kernel_size=3,
                          activation='tanh')(input)
        bn_1 = BatchNormalization()(conv1d_1)
        conv1d_2 = Conv1D(filters=128, kernel_size=3,
                          activation='tanh')(bn_1)
        maxpool_1 = TimeDistributed(MaxPooling1D(
            pool_size=2, strides=2, data_format='channels_first'))(conv1d_2)
        conv1d_3 = Conv1D(filters=256, kernel_size=3,
                          activation='tanh')(maxpool_1)
        bn_2 = BatchNormalization()(conv1d_3)
        conv1d_4 = Conv1D(filters=256, kernel_size=3,
                          activation='tanh')(bn_2)
        maxpool_2 = TimeDistributed(MaxPooling1D(
            pool_size=2, strides=2, data_format='channels_first'))(conv1d_4)
        lstm_1 = TimeDistributed(Bidirectional(
            LSTM(units=350, return_sequences=True)))(maxpool_2)
        lstm_2 = TimeDistributed(Bidirectional(LSTM(units=350)))(lstm_1)
        flatten = Flatten()(lstm_2)
        dense_1 = Dense(256, activation='tanh')(flatten)
        dropout = Dropout(0.1)(dense_1)
        dense_2 = Dense(512, activation='tanh')(dropout)
        if multiclass:
            output = Dense(units=n_outputs, activation='softmax')(dense_2)
            model = Model(inputs=input, outputs=output)
            model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Nadam(learning_rate=0.0001), metrics=['accuracy', tf.keras.metrics.AUC(
            ), tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tfa.metrics.F1Score(num_classes=n_outputs, average='macro')])
            self.model = model
        else:
            output = Dense(units=n_outputs, activation='sigmoid')(dense_2)
            model = Model(inputs=input, outputs=output)
            model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Nadam(learning_rate=0.0001), metrics=['accuracy', tf.keras.metrics.AUC(
            ), tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tfa.metrics.F1Score(num_classes=n_outputs, average='macro')])
            self.model = model

    def train(self, X_train, y_train, X_val, y_val, epochs, batch_size, class_weight=None):
        history = self.model.fit(X_train, y_train, validation_data=(
            X_val, y_val), epochs=epochs, batch_size=batch_size, callbacks=[self.early_stopping], class_weight=class_weight, verbose=2)
        return history

    def save(self, model_path):
        self.model.save(model_path)

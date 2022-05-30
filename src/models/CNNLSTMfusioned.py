import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.layers import Input, LSTM, Dense, Flatten, TimeDistributed, Conv1D, MaxPooling1D, Bidirectional, BatchNormalization, Concatenate, Dropout
from tensorflow.keras import Model


class CNNLSTM_fusioned():
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=25,
        restore_best_weights=True,
    )

    def __init__(self, body_features, face_features, n_length, n_outputs, multiclass=False):
        input_1 = Input(shape=(1, n_length, body_features))
        conv1d_1_1 = Conv1D(filters=128, kernel_size=3,
                            activation='tanh')(input_1)
        bn_1_1 = BatchNormalization()(conv1d_1_1)
        conv1d_2_1 = Conv1D(filters=128, kernel_size=3,
                            activation='tanh')(bn_1_1)
        maxpool_1_1 = TimeDistributed(MaxPooling1D(
            pool_size=2, strides=2, data_format='channels_first'))(conv1d_2_1)
        conv1d_3_1 = Conv1D(filters=256, kernel_size=3,
                            activation='tanh')(maxpool_1_1)
        bn_2_1 = BatchNormalization()(conv1d_3_1)
        conv1d_4_1 = Conv1D(filters=256, kernel_size=3,
                            activation='tanh')(bn_2_1)
        maxpool_2_1 = TimeDistributed(MaxPooling1D(
            pool_size=2, strides=2, data_format='channels_first'))(conv1d_4_1)
        lstm_1_1 = TimeDistributed(Bidirectional(
            LSTM(units=350, return_sequences=True)))(maxpool_2_1)
        lstm_2_1 = TimeDistributed(Bidirectional(LSTM(units=350)))(lstm_1_1)
        flatten_1 = Flatten()(lstm_2_1)
        dense_1_1 = Dense(256, activation='tanh')(flatten_1)
        dropout_1 = Dropout(0.1)(dense_1_1)
        dense_2_1 = Dense(512, activation='tanh')(dropout_1)

        input_2 = Input(shape=(1, n_length, face_features))
        conv1d_1_2 = Conv1D(filters=128, kernel_size=3,
                            activation='tanh')(input_2)
        bn_1_2 = BatchNormalization()(conv1d_1_2)
        conv1d_2_2 = Conv1D(filters=128, kernel_size=3,
                            activation='tanh')(bn_1_2)
        maxpool_1_2 = TimeDistributed(MaxPooling1D(
            pool_size=2, strides=2, data_format='channels_first'))(conv1d_2_2)
        conv1d_3_2 = Conv1D(filters=256, kernel_size=3,
                            activation='tanh')(maxpool_1_2)
        bn_2_2 = BatchNormalization()(conv1d_3_2)
        conv1d_4_2 = Conv1D(filters=256, kernel_size=3,
                            activation='tanh')(bn_2_2)
        maxpool_2_2 = TimeDistributed(MaxPooling1D(
            pool_size=2, strides=2, data_format='channels_first'))(conv1d_4_2)
        lstm_1_2 = TimeDistributed(Bidirectional(
            LSTM(units=350, return_sequences=True)))(maxpool_2_2)
        lstm_2_2 = TimeDistributed(Bidirectional(LSTM(units=350)))(lstm_1_2)
        flatten_2 = Flatten()(lstm_2_2)
        dense_1_2 = Dense(256, activation='tanh')(flatten_2)
        dropout_2 = Dropout(0.1)(dense_1_2)
        dense_2_2 = Dense(512, activation='tanh')(dropout_2)

        concat = Concatenate()([dense_2_1, dense_2_2])
        if multiclass:
            output = Dense(units=n_outputs, activation='softmax')(concat)
            model = Model(inputs=[input_1, input_2], outputs=output)
            model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Nadam(learning_rate=0.0001), metrics=['accuracy', tf.keras.metrics.AUC(
            ), tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tfa.metrics.F1Score(num_classes=n_outputs, average='macro')])
            self.model = model
        else:
            output = Dense(units=n_outputs, activation='sigmoid')(concat)
            model = Model(inputs=[input_1, input_2], outputs=output)
            model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Nadam(learning_rate=0.0001), metrics=['accuracy', tf.keras.metrics.AUC(
            ), tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tfa.metrics.F1Score(num_classes=n_outputs, average='macro')])
            self.model = model

    def train(self, X_train, y_train, X_val, y_val, epochs, batch_size, class_weight=None):
        history = self.model.fit(X_train, y_train, validation_data=(
            X_val, y_val), epochs=epochs, batch_size=batch_size, callbacks=[self.early_stopping], class_weight=class_weight, verbose=2)
        return history

    def train(self, body_X_train, face_X_train, y_train, body_X_val, face_X_val, y_val, epochs, batch_size, class_weight=None):
        history = self.model.fit([body_X_train, face_X_train], y_train, validation_data=(
            [body_X_val, face_X_val], y_val), epochs=epochs, batch_size=batch_size, callbacks=[self.early_stopping], class_weight=class_weight, verbose=2)
        return history

    def save(self, model_path):
        self.model.save(model_path)

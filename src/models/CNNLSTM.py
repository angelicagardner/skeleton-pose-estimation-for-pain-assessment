import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.layers import Input, LSTM, Dense, Flatten, TimeDistributed, Conv1D, MaxPooling1D, Bidirectional, BatchNormalization, Concatenate
from tensorflow.keras import Model


class CNNLSTM():
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=50,
        restore_best_weights=True,
    )

    def __init__(self, n_features, second_n_features, n_length, n_outputs, fusion=False):
        if fusion:
            input_1 = Input(shape=(1, n_length, n_features))
            conv1d_1 = TimeDistributed(
                Conv1D(filters=64, kernel_size=3, activation='tanh'))(input_1)
            maxpool_1 = TimeDistributed(MaxPooling1D(
                pool_size=2, data_format='channels_first'))(conv1d_1)
            flatten_1 = TimeDistributed(Flatten())(maxpool_1)
            lstm_1 = LSTM(300)(flatten_1)
            dense_1 = Dense(64, activation='tanh')(lstm_1)

            input_2 = Input(shape=(1, n_length, second_n_features))
            conv1d_2 = TimeDistributed(
                Conv1D(filters=64, kernel_size=3, activation='tanh'))(input_2)
            maxpool_2 = TimeDistributed(MaxPooling1D(
                pool_size=2, data_format='channels_first'))(conv1d_2)
            flatten_2 = TimeDistributed(Flatten())(maxpool_2)
            lstm_2 = LSTM(300)(flatten_2)
            dense_2 = Dense(64, activation='tanh')(lstm_2)

            concat = Concatenate()([dense_1, dense_2])
            output = Dense(units=n_outputs, activation='sigmoid')(concat)
            model = Model(inputs=[input_1, input_2], outputs=[output])
        else:
            input = Input(shape=(1, n_length, n_features))
            conv1d_1 = TimeDistributed(
                Conv1D(filters=64, kernel_size=3, activation='tanh'))(input)
            bn = TimeDistributed(BatchNormalization())(conv1d_1)
            conv1d_2 = TimeDistributed(
                Conv1D(filters=128, kernel_size=3, activation='tanh'))(bn)
            maxpool = TimeDistributed(MaxPooling1D(
                pool_size=2, strides=2, data_format='channels_first'))(conv1d_2)
            flatten = TimeDistributed(Flatten())(maxpool)
            lstm = Bidirectional(LSTM(256, activation='tanh'))(flatten)
            dense = Dense(512, activation='tanh')(lstm)
            output = Dense(units=n_outputs, activation='sigmoid')(dense)
            model = Model(inputs=input, outputs=output)

        model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0001), metrics=['accuracy', tf.keras.metrics.AUC(
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

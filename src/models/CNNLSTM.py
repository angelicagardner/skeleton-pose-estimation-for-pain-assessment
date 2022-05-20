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

    def __init__(self, n_features, second_n_features, n_length, n_outputs, fusion=False, multiclass=False):
        if fusion:
            input_1 = Input(shape=(1, n_length, n_features))
            conv1d_1 = TimeDistributed(
                Conv1D(filters=64, kernel_size=3, activation='tanh'))(input_1)
            bn_1 = TimeDistributed(BatchNormalization())(conv1d_1)
            conv1d_11 = TimeDistributed(
                Conv1D(filters=128, kernel_size=3, activation='tanh'))(bn_1)
            maxpool_1 = TimeDistributed(MaxPooling1D(
                pool_size=2, data_format='channels_first'))(conv1d_11)
            conv1d_3 = TimeDistributed(
                Conv1D(filters=64, kernel_size=3, activation='tanh'))(maxpool_1)
            bn_11 = TimeDistributed(BatchNormalization())(conv1d_3)
            conv1d_33 = TimeDistributed(
                Conv1D(filters=128, kernel_size=3, activation='tanh'))(bn_11)
            maxpool_11 = TimeDistributed(MaxPooling1D(
                pool_size=2, data_format='channels_first'))(conv1d_33)
            flatten_1 = TimeDistributed(Flatten())(maxpool_11)
            lstm_1 = LSTM(300)(flatten_1)
            dense_1 = Dense(256, activation='tanh')(lstm_1)
            dense_11 = Dense(512, activation='tanh')(dense_1)

            input_2 = Input(shape=(1, n_length, second_n_features))
            conv1d_2 = TimeDistributed(
                Conv1D(filters=64, kernel_size=3, activation='tanh'))(input_2)
            bn_2 = TimeDistributed(BatchNormalization())(conv1d_1)
            conv1d_22 = TimeDistributed(
                Conv1D(filters=128, kernel_size=3, activation='tanh'))(bn_2)
            maxpool_2 = TimeDistributed(MaxPooling1D(
                pool_size=2, data_format='channels_first'))(conv1d_22)
            conv1d_4 = TimeDistributed(
                Conv1D(filters=64, kernel_size=3, activation='tanh'))(input_2)
            bn_22 = TimeDistributed(BatchNormalization())(conv1d_4)
            conv1d_44 = TimeDistributed(
                Conv1D(filters=128, kernel_size=3, activation='tanh'))(bn_22)
            maxpool_22 = TimeDistributed(MaxPooling1D(
                pool_size=2, data_format='channels_first'))(conv1d_44)
            flatten_2 = TimeDistributed(Flatten())(maxpool_22)
            lstm_2 = LSTM(300)(flatten_2)
            dense_2 = Dense(256, activation='tanh')(lstm_2)
            dense_22 = Dense(512, activation='tanh')(dense_2)

            concat = Concatenate()([dense_11, dense_22])
            if multiclass:
                output = Dense(units=n_outputs, activation='softmax')(concat)
            else:
                output = Dense(units=n_outputs, activation='sigmoid')(concat)
            model = Model(inputs=[input_1, input_2], outputs=[output])
        else:
            input = Input(shape=(1, n_length, n_features))
            conv1d_1 = Conv1D(filters=64, kernel_size=3,
                              activation='relu')(input)
            maxpool_1 = TimeDistributed(MaxPooling1D(
                pool_size=2, strides=2, data_format='channels_first'))(conv1d_1)
            conv1d_2 = Conv1D(filters=128, kernel_size=3,
                              activation='relu')(maxpool_1)
            conv1d_3 = Conv1D(filters=128, kernel_size=3,
                              activation='relu')(conv1d_2)
            maxpool_2 = TimeDistributed(MaxPooling1D(
                pool_size=2, strides=2, data_format='channels_first'))(conv1d_3)
            conv1d_4 = Conv1D(filters=256, kernel_size=3,
                              activation='relu')(maxpool_2)
            conv1d_5 = Conv1D(filters=256, kernel_size=3,
                              activation='relu')(conv1d_4)
            conv1d_6 = Conv1D(filters=256, kernel_size=3,
                              activation='relu')(conv1d_5)
            maxpool_3 = TimeDistributed(MaxPooling1D(
                pool_size=2, strides=2, data_format='channels_first'))(conv1d_6)
            conv1d_7 = Conv1D(filters=512, kernel_size=3,
                              activation='relu')(maxpool_3)
            conv1d_8 = Conv1D(filters=512, kernel_size=3,
                              activation='relu')(conv1d_7)
            conv1d_9 = Conv1D(filters=512, kernel_size=3,
                              activation='relu')(conv1d_8)
            if multiclass:
                output = Dense(units=n_outputs, activation='softmax')(dense_2)
            else:
                output = Dense(units=n_outputs, activation='sigmoid')(dense_2)
            model = Model(inputs=input, outputs=output)

        if multiclass:
            model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Nadam(learning_rate=0.0001), metrics=['accuracy', tf.keras.metrics.AUC(
            ), tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tfa.metrics.F1Score(num_classes=n_outputs, average='macro')])
            self.model = model
        else:
            model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Nadam(learning_rate=0.0001), metrics=['accuracy', tf.keras.metrics.AUC(
            ), tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tfa.metrics.F1Score(num_classes=n_outputs, average='macro')])
            self.model = model

    def train(self, X_train, y_train, X_val, y_val, epochs, batch_size, class_weight=None):
        history = self.model.fit(X_train, y_train, validation_data=(
            X_val, y_val), epochs=epochs, batch_size=batch_size, callbacks=[self.early_stopping], class_weight=class_weight, verbose=2)
        return history

    def trainFusioned(self, body_X_train, face_X_train, y_train, body_X_val, face_X_val, y_val, epochs, batch_size, class_weight=None):
        history = self.model.fit([body_X_train, face_X_train], y_train, validation_data=(
            [body_X_val, face_X_val], y_val), epochs=epochs, batch_size=batch_size, callbacks=[self.early_stopping], class_weight=class_weight, verbose=2)
        return history

    def save(self, model_path):
        self.model.save(model_path)

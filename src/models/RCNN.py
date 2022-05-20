import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.layers import Input, Dense, Flatten, TimeDistributed, Conv1D, MaxPooling1D, Concatenate, BatchNormalization, Dropout
from tensorflow.keras import Model
from tensorflow.keras.initializers import Constant
from keras.layers.advanced_activations import PReLU


class RCNN():
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=50,
        restore_best_weights=True,
    )

    def __init__(self, n_features, second_n_features, n_length, n_outputs, fusion=False, multiclass=False):
        if fusion:
            input_1 = Input(shape=(1, n_length, n_features))
            conv1d_1 = TimeDistributed(Conv1D(filters=64, kernel_size=3, activation=PReLU(
                alpha_initializer=Constant(value=0.25))))(input_1)
            bn_1 = TimeDistributed(BatchNormalization())(conv1d_1)
            conv1d_2 = TimeDistributed(Conv1D(filters=128, kernel_size=3, activation=PReLU(
                alpha_initializer=Constant(value=0.25))))(bn_1)
            maxpool_1 = TimeDistributed(MaxPooling1D(
                pool_size=2, data_format='channels_first'))(conv1d_2)
            flatten_1 = Flatten()(maxpool_1)
            dense_1 = Dense(128, activation=PReLU(
                alpha_initializer=Constant(value=0.25)))(flatten_1)
            dense_11 = Dense(512, activation=PReLU(
                alpha_initializer=Constant(value=0.25)))(dense_1)

            input_2 = Input(shape=(1, n_length, second_n_features))
            conv1d_2 = TimeDistributed(Conv1D(filters=64, kernel_size=3, activation=PReLU(
                alpha_initializer=Constant(value=0.25))))(input_2)
            bn_2 = TimeDistributed(BatchNormalization())(conv1d_2)
            conv1d_22 = TimeDistributed(Conv1D(filters=128, kernel_size=3, activation=PReLU(
                alpha_initializer=Constant(value=0.25))))(bn_2)
            maxpool_2 = TimeDistributed(MaxPooling1D(
                pool_size=2, data_format='channels_first'))(conv1d_22)
            flatten_2 = Flatten()(maxpool_2)
            dense_2 = Dense(128, activation=PReLU(
                alpha_initializer=Constant(value=0.25)))(flatten_1)
            dense_22 = Dense(512, activation=PReLU(
                alpha_initializer=Constant(value=0.25)))(dense_1)

            concat = Concatenate()([dense_11, dense_22])
            if multiclass:
                output = Dense(units=n_outputs, activation='softmax')(concat)
            else:
                output = Dense(units=n_outputs, activation='sigmoid')(concat)
            model = Model(inputs=[input_1, input_2], outputs=[output])
        else:
            input = Input(shape=(1, n_length, n_features))
            conv1 = Conv1D(filters=128, kernel_size=1, padding='same',)
            stack1 = conv1(input)
            stack2 = BatchNormalization()(stack1)
            stack3 = PReLU()(stack2)
            conv2 = Conv1D(filters=128, kernel_size=3,
                           padding='same', kernel_initializer='he_normal')
            stack4 = conv2(stack3)
            stack5 = Concatenate()([stack1, stack4])
            stack6 = BatchNormalization()(stack5)
            stack7 = PReLU()(stack6)
            conv3 = Conv1D(filters=128, kernel_size=3,
                           padding='same')
            stack8 = conv3(stack7)
            stack9 = Concatenate()([stack1, stack8])
            stack10 = BatchNormalization()(stack9)
            stack11 = PReLU()(stack10)
            conv4 = Conv1D(filters=128, kernel_size=3, padding='same')
            stack12 = conv4(stack11)
            stack13 = Concatenate()([stack1, stack12])
            stack14 = BatchNormalization()(stack13)
            stack15 = PReLU()(stack14)
            stack16 = TimeDistributed(MaxPooling1D(
                (2), strides=2, data_format='channels_first'))(stack15)
            stack17 = Dropout(0.1)(stack16)
            flatten = Flatten()(stack17)
            if multiclass:
                output = Dense(units=n_outputs, activation='softmax')(flatten)
            else:
                output = Dense(units=n_outputs, activation='sigmoid')(flatten)
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

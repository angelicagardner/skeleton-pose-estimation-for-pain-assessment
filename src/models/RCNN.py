import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.layers import Input, Dense, Flatten, TimeDistributed, Conv1D, MaxPooling1D, Concatenate, BatchNormalization, Dropout
from tensorflow.keras import Model
from keras.layers.advanced_activations import PReLU


class RCNN():
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=30,
        restore_best_weights=True,
    )

    def __init__(self, n_features, n_length, n_outputs, multiclass=False):
        input = Input(shape=(1, n_length, n_features))
        conv1 = Conv1D(filters=128, kernel_size=1, padding='same')
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
        stack16 = TimeDistributed(MaxPooling1D(
            (2), strides=2, data_format='channels_first'))(stack11)
        stack17 = Dropout(0.1)(stack16)
        flatten = Flatten()(stack17)
        if multiclass:
            output = Dense(units=n_outputs, activation='softmax')(flatten)
            model = Model(inputs=input, outputs=output)
            model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Nadam(learning_rate=0.0001), metrics=['accuracy', tf.keras.metrics.AUC(
            ), tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tfa.metrics.F1Score(num_classes=n_outputs, average='macro')])
            self.model = model
        else:
            output = Dense(units=n_outputs, activation='sigmoid')(flatten)
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

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.layers import Input, Dense, Flatten, TimeDistributed, Conv1D, MaxPooling1D, Concatenate, BatchNormalization, Dropout
from tensorflow.keras import Model
from keras.layers.advanced_activations import PReLU


class RCNN_fusioned():
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=50,
        restore_best_weights=True,
    )

    def __init__(self, body_features, face_features, n_length, n_outputs, multiclass=False):
        input_1 = Input(shape=(1, n_length, body_features))
        conv1_1 = Conv1D(filters=128, kernel_size=1, padding='same',)
        stack1_1 = conv1_1(input_1)
        stack2_1 = BatchNormalization()(stack1_1)
        stack3_1 = PReLU()(stack2_1)
        conv2_1 = Conv1D(filters=128, kernel_size=3,
                         padding='same', kernel_initializer='he_normal')
        stack4_1 = conv2_1(stack3_1)
        stack5_1 = Concatenate()([stack1_1, stack4_1])
        stack6_1 = BatchNormalization()(stack5_1)
        stack7_1 = PReLU()(stack6_1)
        conv3_1 = Conv1D(filters=128, kernel_size=3,
                         padding='same')
        stack8_1 = conv3_1(stack7_1)
        stack9_1 = Concatenate()([stack1_1, stack8_1])
        stack10_1 = BatchNormalization()(stack9_1)
        stack11_1 = PReLU()(stack10_1)
        conv4_1 = Conv1D(filters=128, kernel_size=3, padding='same')
        stack12_1 = conv4_1(stack11_1)
        stack13_1 = Concatenate()([stack1_1, stack12_1])
        stack14_1 = BatchNormalization()(stack13_1)
        stack15_1 = PReLU()(stack14_1)
        stack16_1 = TimeDistributed(MaxPooling1D(
            (2), strides=2, data_format='channels_first'))(stack15_1)
        stack17_1 = Dropout(0.1)(stack16_1)
        flatten_1 = Flatten()(stack17_1)

        input_2 = Input(shape=(1, n_length, face_features))
        conv1_2 = Conv1D(filters=128, kernel_size=1, padding='same',)
        stack1_2 = conv1_2(input_2)
        stack2_2 = BatchNormalization()(stack1_2)
        stack3_2 = PReLU()(stack2_2)
        conv2_2 = Conv1D(filters=128, kernel_size=3,
                         padding='same', kernel_initializer='he_normal')
        stack4_2 = conv2_2(stack3_2)
        stack5_2 = Concatenate()([stack1_2, stack4_2])
        stack6_2 = BatchNormalization()(stack5_2)
        stack7_2 = PReLU()(stack6_2)
        conv3_2 = Conv1D(filters=128, kernel_size=3,
                         padding='same')
        stack8_2 = conv3_2(stack7_2)
        stack9_2 = Concatenate()([stack1_2, stack8_2])
        stack10_2 = BatchNormalization()(stack9_2)
        stack11_2 = PReLU()(stack10_2)
        conv4_2 = Conv1D(filters=128, kernel_size=3, padding='same')
        stack12_2 = conv4_2(stack11_2)
        stack13_2 = Concatenate()([stack1_2, stack12_2])
        stack14_2 = BatchNormalization()(stack13_2)
        stack15_2 = PReLU()(stack14_2)
        stack16_2 = TimeDistributed(MaxPooling1D(
            (2), strides=2, data_format='channels_first'))(stack15_2)
        stack17_2 = Dropout(0.1)(stack16_2)
        flatten_2 = Flatten()(stack17_2)

        concat = Concatenate()([flatten_1, flatten_2])

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

    def train(self, body_X_train, face_X_train, y_train, body_X_val, face_X_val, y_val, epochs, batch_size, class_weight=None):
        history = self.model.fit([body_X_train, face_X_train], y_train, validation_data=(
            [body_X_val, face_X_val], y_val), epochs=epochs, batch_size=batch_size, callbacks=[self.early_stopping], class_weight=class_weight, verbose=2)
        return history

    def save(self, model_path):
        self.model.save(model_path)

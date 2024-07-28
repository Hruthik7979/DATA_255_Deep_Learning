import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential, load_model, Model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout, GlobalAveragePooling2D
from keras.applications import VGG16
from keras.optimizers import Adam
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from keras_tuner.tuners import Hyperband
from keras_tuner.engine.hyperparameters import HyperParameters
import kerastuner as kt
import IPython
from keras.layers import MaxPooling2D

input_shape = (32, 32, 3)
num_classes = 43

class TrafficSignClassifier:
    def __init__(self, model_type='sequential'):
        self.model_type = model_type
        self.model = None
        self.best_model = None

    def build_model(self, hp):
        raise NotImplementedError("Subclasses must implement the build_model method")

    def compile_model(self, optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def train_model(self, X_train, y_train, X_val, y_val, epochs=10, batch_size=256):
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=3,
            restore_best_weights=True
        )

        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping]
        )

        return history

    def evaluate_model(self, X_test, y_test):
        y_pred = self.best_model.predict(X_test)
        classes_x = np.argmax(y_pred, axis=1)
        print("Test Accuracy: ", accuracy_score(y_test, classes_x))
        accuracy = accuracy_score(y_test, classes_x)
        print(f'Accuracy: {accuracy}')

        # Calculate precision
        precision = precision_score(y_test, classes_x, average='weighted')
        print(f'Precision: {precision}')

        # Calculate recall
        recall = recall_score(y_test, classes_x, average='weighted')
        print(f'Recall: {recall}')

        # Calculate F1 score
        f1 = f1_score(y_test, classes_x, average='weighted')
        print(f'F1 Score: {f1}')

    def plot_history(self, history):
        plt.figure(0)
        plt.plot(history.history['accuracy'], label='training accuracy')
        plt.plot(history.history['val_accuracy'], label='val accuracy')
        plt.title('Accuracy')
        plt.xlabel('epochs')
        plt.ylabel('accuracy')
        plt.legend()
        plt.show()

        plt.plot(history.history['loss'], label='training loss')
        plt.plot(history.history['val_loss'], label='val loss')
        plt.title('Loss')
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.legend()
        plt.show()

    def predict_on_test_set(self, X_test):
        return self.best_model.predict(X_test)

    def summary(self):
        self.model.summary()

    def test_on_img(self, img):
        data = []
        image = Image.open(img)
        image = image.resize((32, 32))
        image = np.array(image) / 255.0
        data.append(np.array(image))
        data = np.array(data)
        y_pred_img = self.best_model.predict(data)
        y_pred_img = np.argmax(y_pred_img, axis=1)
        return image, y_pred_img

    def test_on_image(self, img_path, class_names):
        plot, prediction = self.test_on_img(img_path)
        print("Predicted traffic sign is: ", class_names[prediction[0]])
        plt.imshow(plot)
        plt.show()

    def tune_hyperparameters(self, X_train, y_train, X_val, y_val, epochs=10, num_classes=None):
        tuner = Hyperband(
            self.build_model,
            objective='val_accuracy',
            max_epochs=epochs,
            factor=5,
            directory='my_dir',
            project_name=f'{self.model_type}_hyperband'
        )

        class ClearTrainingOutput(tf.keras.callbacks.Callback):
            def on_train_end(*args, **kwargs):
                if IPython:
                    IPython.display.clear_output(wait=True)
                print('\nTraining completed.')

        tuner.search(
            X_train, y_train,
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=[ClearTrainingOutput(), tf.keras.callbacks.EarlyStopping(patience=3)],
            verbose=1,
            batch_size=256
        )

        # Print or log details of each trial
        tuner.results_summary()

        # Get the optimal hyperparameters
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

        # Build the model with the best hyperparameters
        self.best_model = self.build_model(best_hps)

        return best_hps, self.best_model

class VGG16Model(TrafficSignClassifier):
    def build_model(self, hp):
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)

        # Freeze convolutional layers
        for layer in base_model.layers:
            layer.trainable = False

        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(hp.Float('dropout', min_value=0.0, max_value=0.5, step=0.25))(x)
        x = Dense(128, activation='relu')(x)
        predictions = Dense(num_classes, activation='softmax')(x)

        model = Model(inputs=base_model.input, outputs=predictions)

        for layer in model.layers[:10]:
            layer.trainable = True

        model.compile(
            optimizer=tf.keras.optimizers.legacy.Adam(
                learning_rate=hp.Choice('learning_rate', values=[1e-3, 1e-4]),
                decay=hp.Choice('l2_decay', values=[1e-4, 1e-5])
            ),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

class CustomCNNModel(TrafficSignClassifier):
    def build_model(self, hp):
        model = Sequential()

        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
        model.add(MaxPool2D(2, 2))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPool2D(2, 2))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(MaxPool2D(2, 2))

        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(hp.Float('dropout', min_value=0.2, max_value=0.5, step=0.1)))
        model.add(Dense(num_classes, activation='softmax'))

        model.compile(
            optimizer=tf.keras.optimizers.legacy.Adam(
                learning_rate=hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4]),
                decay=hp.Choice('l2_decay', values=[1e-4, 1e-5])
            ),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

class AlexNetModel(TrafficSignClassifier):
    def build_model(self, hp):
        model = Sequential()

        model.add(Conv2D(
            hp.Int('conv1_units', min_value=32, max_value=128, step=32),
            (11, 11),
            strides=(4, 4),
            activation='relu',
            input_shape=input_shape,
            padding='same'
        ))
        model.add(MaxPooling2D(2, 2))

        model.add(Conv2D(
            hp.Int('conv2_units', min_value=64, max_value=256, step=64),
            (5, 5),
            padding='same',
            activation='relu'
        ))
        model.add(MaxPooling2D(2, 2))

        model.add(Conv2D(
            hp.Int('conv3_units', min_value=128, max_value=512, step=128),
            (3, 3),
            padding='same',
            activation='relu'
        ))

        model.add(Conv2D(
            hp.Int('conv4_units', min_value=128, max_value=512, step=128),
            (3, 3),
            padding='same',
            activation='relu'
        ))

        model.add(Conv2D(
            hp.Int('conv5_units', min_value=128, max_value=512, step=128),
            (3, 3),
            padding='same',
            activation='relu'
        ))
        model.add(MaxPooling2D(2, 2))

        model.add(Flatten())
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(hp.Float('dropout1', min_value=0.2, max_value=0.5, step=0.1)))
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(hp.Float('dropout2', min_value=0.2, max_value=0.5, step=0.1)))
        model.add(Dense(num_classes, activation='softmax'))

        model.compile(
            optimizer=tf.keras.optimizers.legacy.Adam(
                learning_rate=hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4]),
                decay=hp.Choice('l2_decay', values=[1e-4, 1e-5])
            ),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

class LeNetModel(TrafficSignClassifier):
    def build_model(self, hp):
        model = Sequential()

        model.add(Conv2D(
            hp.Int('conv1_units', min_value=8, max_value=32, step=8),
            (5, 5),
            activation='relu',
            input_shape=input_shape
        ))
        model.add(MaxPooling2D(2, 2))

        model.add(Conv2D(
            hp.Int('conv2_units', min_value=16, max_value=64, step=16),
            (5, 5),
            activation='relu'
        ))
        model.add(MaxPooling2D(2, 2))

        model.add(Flatten())
        model.add(Dense(120, activation='relu'))
        model.add(Dense(84, activation='relu'))
        model.add(Dropout(hp.Float('dropout', min_value=0.2, max_value=0.5, step=0.1)))
        model.add(Dense(num_classes, activation='softmax'))

        model.compile(
            optimizer=tf.keras.optimizers.legacy.Adam(
                learning_rate=hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4]),
                decay=hp.Choice('l2_decay', values=[1e-4, 1e-5])
            ),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

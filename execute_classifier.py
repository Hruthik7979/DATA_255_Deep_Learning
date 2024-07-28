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
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
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
from traffic_sign_classifier import VGG16Model, CustomCNNModel, AlexNetModel, LeNetModel
from data_loading_and_preprocessing import load_and_augment_data, preprocess_data, testing
from class_names import class_names

def main(model_type):
    traffic_sign_classifier = None

    if model_type == 'vgg16':
        traffic_sign_classifier = VGG16Model(model_type)
    elif model_type == 'customcnn':
        traffic_sign_classifier = CustomCNNModel(model_type)
    elif model_type == 'alexnet':
        traffic_sign_classifier = AlexNetModel(model_type)
    elif model_type == 'lenet':
        traffic_sign_classifier = LeNetModel(model_type)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    data, labels = load_and_augment_data('/Users/pari/Documents/3rd sem/Data 255/Project/Traffic_Sign_Dataset')
    X_train, X_val, y_train, y_val, class_names = preprocess_data(data, labels)


    # Tune hyperparameters
    best_hyperparameters, best_model = traffic_sign_classifier.tune_hyperparameters(
        X_train, y_train, X_val, y_val, epochs=5, num_classes=len(class_names)
    )

    # Train the best model
    history = best_model.fit(X_train, y_train, epochs=5, validation_data=(X_val, y_val))

    print("Best Hyperparameters: ", best_hyperparameters.values)
    

    best_model.save(f'best_model_{model_type}_hyperband.h5')
    best_model = load_model(f'best_model_{model_type}_hyperband.h5')
    
    best_model.summary()

    X_test, y_test = testing('Test.csv')

    # Evaluate the best model
    traffic_sign_classifier.evaluate_model(X_test, y_test)

    # Plots
    traffic_sign_classifier.plot_history(history)

    # Test on a single image
    test_image_path = '/Users/pari/Documents/3rd sem/Data 255/Project/Traffic_Sign_Dataset/Test/00500.png'
    traffic_sign_classifier.test_on_image(test_image_path, class_names)


if __name__ == "__main__":
    model_type_input = input("Enter the model type ('vgg16', 'customcnn', 'alexnet', or 'lenet'): ")
    main(model_type_input)

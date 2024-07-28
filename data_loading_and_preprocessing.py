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
import kerastuner as kt
import IPython
from keras.layers import MaxPooling2D
from class_names import class_names

def load_and_augment_data(path, classes=43, image_shape=(32, 32, 3)):
    os.chdir(path)
    cur_path = os.getcwd()
    data = []
    labels = []

    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=False,
        vertical_flip=False
    )

    for i in range(classes):
        path = os.path.join(cur_path, 'train', str(i))
        images = os.listdir(path)
        
        for a in images:
            try:
                image_path = os.path.join(path, a)
                print("Opening:", image_path)
                
                # Load image and resize to 32x32
                image = Image.open(image_path)
                image = image.resize((32, 32))
                image = np.array(image)
                image = image.reshape((1,) + image_shape) 
                augmented_images = datagen.flow(image, batch_size=1)
                
                # Normalize the image to the range [0, 1]
                image = np.squeeze(image, axis=0)
                image = image / 255.0
                
                data.append(image)
                labels.append(i)
    
                # Augment and add augmented images
                for j in range(4):  # assuming you want to add 4 augmented versions
                    augmented_image = augmented_images.next()[0]
                    augmented_image = augmented_image / 255.0
                    data.append(augmented_image)
                    labels.append(i)
    
            except Exception as e:
                print(e)
    
    # Convert lists to NumPy arrays
    data = np.array(data)
    labels = np.array(labels)

    return data, labels

def preprocess_data(data, labels, test_size=0.2):
    unique_labels = np.unique(labels)

    y_one_hot = to_categorical(labels, num_classes=len(unique_labels))
    X_train, X_val, y_train, y_val = train_test_split(data, y_one_hot, test_size=test_size, random_state=0)

    return X_train, X_val, y_train, y_val, class_names

def testing(testcsv):
    y_test = pd.read_csv(testcsv)
    
    label = y_test["ClassId"].values
    imgs = y_test["Path"].values
    data=[]
    for img in imgs:
        image = Image.open(img)
        image = image.resize((32,32))
        image = np.array(image) / 255.0
        data.append(image)
    X_test=np.array(data)
    return X_test,label

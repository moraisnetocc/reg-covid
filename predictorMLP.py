# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn

# Import necessary modules
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
from PIL import Image
import cv2


# Keras specific
import keras
from keras.models import Sequential
from keras.layers import Dense


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


def open_image(filepath, size):
    image = Image.open(filepath, 'r')
    image = image.resize((size, size), Image.ANTIALIAS)
    # img = cv2.resize(cv2.imread(filepath, cv2.IMREAD_COLOR), (size, size)).astype(np.float32)
    # return img.transpose((2, 0, 1))
    return rgb2gray(np.array(image.convert('RGB'))).tolist()


def create_image_data_set(n, size):
    filepath = "/Users/moraisneto/PycharmProjects/covidAI/suspeitos/"
    images = []
    for i in range(1, n + 1):
        images.append(open_image(filepath + "{}.jpg".format(i), size))
    print('Imagens carregadas')
    return images


images = create_image_data_set(10, 200)

model = Sequential()
model.add(Dense(500, input_dim=4, activation= "relu"))
model.add(Dense(100, activation= "relu"))
model.add(Dense(50, activation= "relu"))
model.add(Dense(1))

model.compile(loss= "mean_squared_error" , optimizer="adam", metrics=["mean_squared_error"])
model.fit(images[0], images[1], epochs=20)
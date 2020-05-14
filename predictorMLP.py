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
    return rgb2gray(np.array(image.convert('RGB')))#.tolist()


def create_image_data_set(n, size):
    filepath = "/Users/moraisneto/PycharmProjects/covidAI/suspeitos/"
    images = []
    for i in range(1, n + 1):
        images.append(open_image(filepath + "{}.jpg".format(i), size))
    print('Imagens carregadas')
    return np.asarray(images)


images = create_image_data_set(15, 200)

print(images.shape)
x = images[0:10].reshape(images[:10].shape[0], images[:10].shape[1], 1)

y = images[1:11].reshape(images[1:11].shape[0], images[1:11].shape[1], 1)

X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=40)

model = Sequential()
# model.add(Dense(500, input_dim=1, activation= "relu"))
model.add(Dense(500, input_shape=(15, 150, 200), activation= "relu"))
model.add(Dense(100, activation= "relu"))
model.add(Dense(50, activation= "relu"))
model.add(Dense(1))

model.compile(loss= "mean_squared_error" , optimizer="adam", metrics=["mean_squared_error"])

model.fit(X_train, y_train, epochs=20)
# pred_train= model.predict(X_train)

pred= model.predict(X_test)
print(np.sqrt(mean_squared_error(y_test,pred)))

# print(pred_train)
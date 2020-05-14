
#matplotlib inline
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from sklearn.model_selection import train_test_split
from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten, Activation
from keras_preprocessing import image

from helper import get_class_names, get_train_data, get_test_data, plot_images, plot_model

# matplotlib.style.use('ggplot')


IMAGE_SIZE = 200
CHANNELS = 3


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


def open_image(filepath, size):
    x = image.load_img(filepath, target_size=(size, size))
    x = image.img_to_array(x)
    x = x.reshape((1,) + x.shape)
    return x/255


def create_image_data_set(n, size):
    filepath = "/Users/moraisneto/PycharmProjects/covidAI/ImagesV2/"
    images = []
    for i in range(1, n + 1):
        images = np.concatenate(images, open_image(filepath + "{}.png".format(i), size))
    print('Imagens carregadas')
    return images


def cnn_model():
    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(512, activation='relu'))
    # model.add(Dropout(0.5))

    # model.add(Dense(num_classes, activation='softmax'))

    model.summary()

    return model


model = cnn_model()
images = create_image_data_set(15, 200)

model.compile(loss= "mean_squared_error" , optimizer="adam", metrics=["mean_squared_error"])

x = images.reshape(images[0], images[1], 1)
y = images[1]

xtrain, xtest, ytrain, ytest=train_test_split(x, y, test_size=0.15)

# model.fit(x[0], y[0])
# images_test, labels_test, class_test = get_test_data()
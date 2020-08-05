import numpy
import matplotlib.pyplot as plt
import pandas
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return numpy.array(dataX), numpy.array(dataY)


def organize_sum(dataset):
    new_data_set = []
    current_value = 0
    for value in dataset:
        current_value += value
        new_data_set.append(current_value)
    return new_data_set


# fix random seed for reproducibility
numpy.random.seed(7)

# load the dataset
dataframe = pandas.read_csv('natal.csv', usecols=[1], engine='python')
dataset = dataframe.values
dataset = dataset.astype('float32')

# we can see if this line above are necessary
# dataset = organize_sum(dataset)

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
train_size = int(len(dataset))
train = dataset

# reshape into X=t and Y=t+1
look_back = 1
trainX, trainY = create_dataset(train, look_back)

# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))


# create and fit the LSTM network
model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)


# make predictions
trainPredict = model.predict(trainX)

day1 = model.predict(numpy.reshape(trainPredict[train_size-3], (1, 1, 1)))
day2 = model.predict(numpy.reshape(day1, (1, 1, 1)))
day3 = model.predict(numpy.reshape(day2, (1, 1, 1)))
day4 = model.predict(numpy.reshape(day3, (1, 1, 1)))
day5 = model.predict(numpy.reshape(day4, (1, 1, 1)))
day6 = model.predict(numpy.reshape(day5, (1, 1, 1)))
day7 = model.predict(numpy.reshape(day6, (1, 1, 1)))
day8 = model.predict(numpy.reshape(day7, (1, 1, 1)))
day9 = model.predict(numpy.reshape(day8, (1, 1, 1)))
day10 = model.predict(numpy.reshape(day9, (1, 1, 1)))

# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])

day1 = scaler.inverse_transform(day1)
day2 = scaler.inverse_transform(day2)
day3 = scaler.inverse_transform(day3)
day4 = scaler.inverse_transform(day4)
day5 = scaler.inverse_transform(day5)
day6 = scaler.inverse_transform(day6)
day7 = scaler.inverse_transform(day7)
day8 = scaler.inverse_transform(day8)
day9 = scaler.inverse_transform(day9)
day10 = scaler.inverse_transform(day10)

output = [
    day1.tolist()[0][0],
    day2.tolist()[0][0],
    day3.tolist()[0][0],
    day4.tolist()[0][0],
    day5.tolist()[0][0],
    day6.tolist()[0][0],
    day7.tolist()[0][0],
    day8.tolist()[0][0],
    day9.tolist()[0][0],
    day10.tolist()[0][0],
]


print(trainPredict)
print(output)
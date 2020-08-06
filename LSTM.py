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
dataframe = pandas.read_csv('brasil.csv', usecols=[1], engine='python')
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

times = 10
output = []
# make predictions
trainPredict = model.predict(trainX)
increment = 1/trainPredict.size
day = 1

while bool(model.predict(numpy.reshape(day, (1, 1, 1))) < 1):
    day += increment

for i in range(times):
    output.append(scaler.inverse_transform(model.predict(numpy.reshape(day, (1, 1, 1)))).tolist()[0][0])
    day += increment

# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])


print(trainPredict)
print(output)
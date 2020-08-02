from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import TimeDistributed
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("A5M.txt")

dataset = df.values
scaler = MinMaxScaler(feature_range=(0,1))
dataset = scaler.fit_transform(dataset)

def split_sequences(sequence, n_steps, size_output):
    X, y = list(), list()
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence)-size_output:
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:end_ix+size_output]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)

n_steps = 12
size_training = 0.5
num_epochs = 1000
units = 50
size_output = 2
# split into samples
x_train_uni = dataset[0:int(size_training*len(dataset))]

X, y = split_sequences(x_train_uni, n_steps,size_output)
y = y.reshape((y.shape[0],size_output))
X = X.reshape((X.shape[0], 2, 6, 1))

model = Sequential()
model.add(TimeDistributed(Conv1D(filters=64, kernel_size=1, activation='relu'), input_shape=(None, 6, 1)))
model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
model.add(TimeDistributed(Flatten()))
model.add(LSTM(units, activation='relu'))
model.add(Dense(size_output))
model.compile(optimizer='adam', loss='mse')

model.fit(X, y, epochs=num_epochs, verbose=1)


act, pred, time = list(), list(), list()
for i in range(int(size_training*len(dataset)),int(len(dataset))-(n_steps*2),size_output):
	if len(dataset[i+(n_steps):i+(n_steps)+size_output]) < size_output:
		break
	x_input = dataset[i:i+(n_steps)]
	x_input = x_input.reshape((1, 2, n_steps, 1))
	yhat = model.predict(x_input, verbose=1)
	act.append(dataset[i+(n_steps):i+(n_steps)+size_output])
	pred.append(yhat[0])

plotAct = array(act).reshape(array(act).shape[1]*array(act).shape[0],1)
plotAct = scaler.inverse_transform(plotAct)
plotPred = array(pred).reshape(array(pred).shape[1]*array(pred).shape[0],1)
plotPred = scaler.inverse_transform(plotPred)


fig, ax = plt.subplots()
ax.ticklabel_format(axis='y', scilimits=(9,9))
plt.ylabel("Gigabits/5m")
plt.xlabel("Timesteps")
fig.set_size_inches(12, 8)
ax.plot(list(range(0,len(plotAct))),plotAct,'r-',label="Actual")
ax.plot(list(range(0,len(plotPred))),plotPred,'b-',label="Predicted")
ax.legend()

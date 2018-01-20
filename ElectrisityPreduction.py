import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# read file
data = pd.read_excel("data.xlsx")

# Take the values
data = data.values

# Make suitable format
temp_data = []
for i in range(0, len(data)):
    date = data[i][0]
    time = data[i][1]
    temperature = data[i][2]
    dewPoint = data[i][3]
    dayOfWeek = data[i][4]
    apparentTemperature = data[i][5]
    kwh = data[i][6]

    temp_data.append([kwh, date.year, date.month, date.day, time, dewPoint, temperature, apparentTemperature, dayOfWeek])

data = temp_data

# dataset splitting
data_train, data_test = train_test_split(data, test_size=0.2)

# Scale data
scaler = MinMaxScaler(feature_range=(-1, 1))
scaler.fit(data_train)
data_train = scaler.transform(data_train)
data_test = scaler.transform(data_test)

# Dividing the data
X_train = data_train[:, 1:]
y_train = data_train[:, 0]
X_test = data_test[:, 1:]
y_test = data_test[:, 0]

# Number of inputs in training data
n_input_dimens = X_train.shape[1]

# Neurons
n_neurons_1 = 1024
n_neurons_2 = 512
n_neurons_3 = 256
n_neurons_4 = 128

# Session
net = tf.InteractiveSession()

# Placeholder
X = tf.placeholder(dtype=tf.float32, shape=[None, n_input_dimens])
Y = tf.placeholder(dtype=tf.float32, shape=[None])

# Initializers
sigma = 1
weight_initializer = tf.variance_scaling_initializer(mode="fan_avg", distribution="uniform", scale=sigma)
bias_initializer = tf.zeros_initializer()

# Hidden weights
W_hidden_1 = tf.Variable(weight_initializer([n_input_dimens, n_neurons_1]))
bias_hidden_1 = tf.Variable(bias_initializer([n_neurons_1]))
W_hidden_2 = tf.Variable(weight_initializer([n_neurons_1, n_neurons_2]))
bias_hidden_2 = tf.Variable(bias_initializer([n_neurons_2]))
W_hidden_3 = tf.Variable(weight_initializer([n_neurons_2, n_neurons_3]))
bias_hidden_3 = tf.Variable(bias_initializer([n_neurons_3]))
W_hidden_4 = tf.Variable(weight_initializer([n_neurons_3, n_neurons_4]))
bias_hidden_4 = tf.Variable(bias_initializer([n_neurons_4]))

# Output weights
W_out = tf.Variable(weight_initializer([n_neurons_4, 1]))
bias_out = tf.Variable(bias_initializer([1]))

# Hidden layer
hidden_1 = tf.nn.relu(tf.add(tf.matmul(X, W_hidden_1), bias_hidden_1))
hidden_2 = tf.nn.relu(tf.add(tf.matmul(hidden_1, W_hidden_2), bias_hidden_2))
hidden_3 = tf.nn.relu(tf.add(tf.matmul(hidden_2, W_hidden_3), bias_hidden_3))
hidden_4 = tf.nn.relu(tf.add(tf.matmul(hidden_3, W_hidden_4), bias_hidden_4))

# Output layer (transpose!)
out = tf.transpose(tf.add(tf.matmul(hidden_4, W_out), bias_out))

# Cost function
mse = tf.reduce_mean(tf.squared_difference(out, Y))

# Optimizer
opt = tf.train.AdamOptimizer().minimize(mse)

# Init
net.run(tf.global_variables_initializer())

batch_size = 256

# Train neural network
epochs = 20
for e in range(epochs):

    # Shuffle training data
    shuffle_indices = np.random.permutation(np.arange(len(y_train)))
    X_train = X_train[shuffle_indices]
    y_train = y_train[shuffle_indices]

    # Minibatch training
    for i in range(0, len(y_train) // batch_size):
        start = i * batch_size
        batch_x = X_train[start:start + batch_size]
        batch_y = y_train[start:start + batch_size]
        # Run optimizer with batch
        net.run(opt, feed_dict={X: batch_x, Y: batch_y})

# read prediction file
data_predict = pd.read_excel("predict.xlsx")

# Making an array
data_predict = data_predict.values

# Making data into suitable format
temp_data = []
for i in range(0, len(data_predict)):
    date = data_predict[i][0]
    time = data_predict[i][1]
    temperature = data_predict[i][2]
    dewPoint = data_predict[i][3]
    dayOfWeek = data_predict[i][4]
    apparentTemperature = data_predict[i][5]
    kwh = data_predict[i][6]

    temp_data.append([kwh, date.year, date.month, date.day, time, dewPoint, temperature, apparentTemperature, dayOfWeek])

data_predict = temp_data

# Scale data
scaler = MinMaxScaler(feature_range=(-1, 1))
scaler.fit(data_predict)
data_predict = scaler.transform(data_predict)

# Dividng the dataset
X_predict = data_predict[:, 1:]
y_predict = data_predict[:, 0]

batch_size = 1

# Prediction of neural network
for i in range(0, len(data_predict) // batch_size):
    start = i * batch_size
    batch_x = X_predict[start:start + batch_size]
    pred = net.run(out, feed_dict={X: batch_x})
    y_predict[i] = pred

# Reverse scaling
data_predict = scaler.inverse_transform(data_predict)

# Plot data
dates = []
kwhs = []

for i in range(len(data_predict)):
    dates.append(str((int(round(data_predict[i][4])))).zfill(2) + ":00")
    kwhs.append(data_predict[i][0])

plt.plot(dates, kwhs)
plt.show()
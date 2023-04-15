import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import yfinance as yf
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf

from pandas_datareader.data import DataReader

tf.keras.backend.clear_session()
tf.random.set_seed(69)
np.random.seed(69)


end = datetime.now()
start = datetime(2015, end.month, end.day)
df = yf.download("AAPL", start, end)

print(df)

tstart = 2015
tend = 2020

def train_test_plot(df, tstart, tend):
    df.loc[f"{tstart}":f"{tend}", "High"].plot(figsize=(16, 4), legend=True)
    df.loc[f"{tend+1}":, "High"].plot(figsize=(16, 4), legend=True)
    plt.legend([f"Train (Before {tend+1})", f"Test ({tend+1} and beyond)"])
    plt.title("APPLE stock price")
    plt.show()

train_test_plot(df,tstart,tend)

def train_test_split(df, tstart, tend):
    train = df.loc[f"{tstart}":f"{tend}", "High"]
    test = df.loc[f"{tend+1}":, "High"]
    return train, test

def train_test_split_values(df, tstart, tend):
    train, test =  train_test_split(df, tstart, tend)
    return train.values, test.values

training_set, test_set = train_test_split_values(df, tstart, tend)

sc = MinMaxScaler(feature_range=(0, 1))
training_set = training_set.reshape(-1, 1)
training_set_scaled = sc.fit_transform(training_set)

test_set = test_set.reshape(-1, 1)
test_set_scaled = sc.fit_transform(test_set)

def split_sequence(sequence, window):
    X, y = list(), list()
    for i in range(len(sequence)):
        end_ix = i + window
        if end_ix > len(sequence) - 1:
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

window_size = 60
features = 1

X_train, y_train = split_sequence(training_set_scaled, window_size)

X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],features)

model_lstm = tf.keras.models.Sequential([
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=125, activation="tanh"), input_shape=(window_size, features)),
    tf.keras.layers.Dense(20),
    tf.keras.layers.Dense(units=1)
])
model_lstm.compile(optimizer='adam', loss='mse', metrics='mae')

print(model_lstm.summary())
model_lstm.fit(X_train, y_train, epochs=15, batch_size=32)

df_total = df.loc[:,"High"]
inputs = df_total[len(df_total) - len(test_set) - window_size :].values
inputs = inputs.reshape(-1, 1)
inputs = sc.transform(inputs)

X_test, y_test = split_sequence(inputs, window_size)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], features)

predicted_stock_price = model_lstm.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)
y_test = sc.inverse_transform(y_test)

def plot_predictions(test, predicted):
    plt.plot(test, color="gray", label="Real")
    plt.plot(predicted, color="red", label="Predicted")
    plt.title("Stock Price Prediction w/ LSTM Layer")
    plt.xlabel("Time")
    plt.ylabel("Stock Price")
    plt.legend()
    plt.show()


def return_rmse(test, predicted):
    rse = mean_squared_error(test, predicted)
    rmse = np.sqrt(rse)
    print("With LSTM Layer the root mean squared error is {:.2f}.".format(rmse))
    
def return_mae(test, predicted):
    mae = mean_absolute_error(test, predicted)
    print("With LSTM Layer the mean absolute error is {:.2f}.".format(mae))
    
plot_predictions(y_test,predicted_stock_price)  

rse = return_rmse(y_test,predicted_stock_price)
mae = return_mae(y_test,predicted_stock_price)

tf.keras.backend.clear_session()
tf.random.set_seed(66)
np.random.seed(66)

window_size = 60
features = 1

X_train, y_train = split_sequence(training_set_scaled, window_size)

X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],features)

model_gru = tf.keras.models.Sequential([
    tf.keras.layers.Bidirectional(tf.keras.layers.GRU(units=125, activation="tanh"), input_shape=(window_size, features)),
    tf.keras.layers.Dense(units=1)
])
model_gru.compile(optimizer='adam', loss='mse', metrics=('mae', 'mse'))

print(model_gru.summary())
model_gru.fit(X_train, y_train, epochs=32, batch_size=32)

X_test, y_test = split_sequence(inputs, window_size)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], features)
predicted_stock_price = model_gru.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)
y_test = sc.inverse_transform(y_test)

def plot_predictions(test, predicted):
    plt.plot(test, color="gray", label="Real")
    plt.plot(predicted, color="red", label="Predicted")
    plt.title("Stock Price Prediction w/ GRU Layer")
    plt.xlabel("Time")
    plt.ylabel("Stock Price")
    plt.legend()
    plt.show()


def return_rmse(test, predicted):
    rse = mean_squared_error(test, predicted)
    rmse = np.sqrt(rse)
    print("With GRU Layer the root mean squared error is {:.2f}.".format(rmse))
    
def return_mae(test, predicted):
    mae = mean_absolute_error(test, predicted)
    print("With GRU Layer the mean absolute error is {:.2f}.".format(mae))
    
plot_predictions(y_test,predicted_stock_price)

rse = return_rmse(y_test,predicted_stock_price)
mae = return_mae(y_test,predicted_stock_price)  
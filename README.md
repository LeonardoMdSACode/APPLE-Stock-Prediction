# Apple Stock Price Prediction

Thanks to yfinance it is possible to work with the dataframe of stock prices and ETFs.
![Apple_stocks](Apple_stock.png)
With this dataframe we got a time series problem. It was tested GRU and LSTM layers for this solution after a preliminary testing these two types performed the best.

Using a LSTM Bidirectional layer followed by dense relu layer and the output dense layer its possible to achieve root mean squared error is 3.58 and mean absolute error is 2.78.

![LSTM](LSTM_pred.png)

With GRU we can have even faster training per epoch and better results, with 15 epochs it is already possible to achieve better results compared to LSTM.
With GRU Layer the root mean squared error is 2.93 and the mean absolute error is 2.30.

![GRU](GRU_pred.png)

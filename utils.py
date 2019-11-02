import numpy as np
import pandas as pd
import math
import sklearn
import sklearn.preprocessing
import os
import matplotlib.pyplot as plt

def plot_raw(df, included_volume=True):
    ''' plot graph of raw data
    '''
    plt.plot(df['Open'], color='red', label='open')
    plt.plot(df['Close'], color='green', label='close')
    plt.plot(df['High'], color='blue', label='high')
    plt.plot(df['Low'], color='black', label='low')
    plt.title('Stock price')
    plt.xlabel('Time [days]')
    plt.ylabel('Price')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()

    if included_volume :
        plt.plot(df['Volume'], color='black', label='volume')
        plt.title('Stock volume')
        plt.xlabel('Time [days]')
        plt.ylabel('Volume')
        plt.legend(loc='best')
        plt.grid(True)
        plt.show()


def normalize_data(df):
    ''' normalize by using only min max scaler
    '''
    min_max_scaler = sklearn.preprocessing.MinMaxScaler()
    df['Open'] = min_max_scaler.fit_transform(df['Open'].to_numpy().reshape(-1, 1))
    df['Low'] = min_max_scaler.fit_transform(df['Low'].to_numpy().reshape(-1, 1))    
    df['High'] = min_max_scaler.fit_transform(df['High'].to_numpy().reshape(-1, 1))    
    df['Close'] = min_max_scaler.fit_transform(df['Close'].to_numpy().reshape(-1, 1))  
    #df['Volume'] = min_max_scaler.fit_transform(df['Volume'].to_numpy().reshape(-1, 1))  
    return df

def load_data(stock, seq_len):
    data_raw = stock.as_matrix()
    data = list()
    for index in range(len(data_raw) - seq_len):
        data.append(data_raw[index: index+seq_len])
    data = np.array(data)
    val_set_size = int(np.round(val_set_size_percentage/100*data.shape[0]))
    test_set_size = int(np.round(test_set_size_percentage/100*data.shape[0]))
    train_set_size = data.shape[0] - (val_set_size + test_set_size)
    
    x_train = data[:train_set_size, :-1, :]
    y_train = data[:train_set_size, -1, :]
    
    x_val = data[train_set_size:train_set_size + val_set_size, :-1, :]
    y_val = data[train_set_size:train_set_size + val_set_size, -1, :]
    
    x_test = data[train_set_size+val_set_size:,:-1,:]
    y_test = data[train_set_size+val_set_size:,-1,:]
    
    return [x_train, y_train, x_val, y_val, x_test, y_test]

def plot_result(y_train, y_val, y_test, y_train_pred, y_val_pred, y_test_pred, ft=0, included_test=True):
    ''' plot graph to compare between prediction and ground truth
    '''

    plt.plot(np.arange(y_train.shape[0]), y_train[:, ft], color='blue', label='train target')

    plt.plot(np.arange(y_train.shape[0], y_train.shape[0] + y_val.shape[0]), y_val[:, ft], 
             color='gray', label='val target')
    plt.plot(np.arange(y_train.shape[0] + y_val.shape[0], y_train.shape[0] + y_val.shape[0] + y_test.shape[0]),
            y_test[:, ft], color='black', label='test target')
    plt.plot(np.arange(y_train_pred.shape[0]), y_train_pred[:, ft],
            color='red', label='train prediction') 
    plt.plot(np.arange(y_train_pred.shape[0], y_train_pred.shape[0] + y_val_pred.shape[0]),
            y_val_pred[:, ft], color='orange', label='val prediction')
    plt.plot(np.arange(y_train_pred.shape[0] + y_val_pred.shape[0],
            y_train_pred.shape[0] + y_val_pred.shape[0] + y_test_pred.shape[0]),
            y_test_pred[:, ft], color ='green', label='test prediction')
    plt.title('past and future stock price')
    plt.xlabel('Time [Days]')
    plt.ylabel('Normalized price')
    plt.grid(True)
    plt.legend(loc='best')
    plt.show()

    if included_test:
        plt.plot(np.arange(y_train.shape[0], y_train.shape[0] + y_test.shape[0]),
                y_test[:, ft], color='black', label='test target')
        plt.plot(np.arange(y_train.shape[0], y_train.shape[0] + y_test_pred.shape[0]),
                y_test_pred[:, ft], color='green', label='test prediction')
        plt.title('future stock prices')
        plt.xlabel('Time [Days]')
        plt.ylabel('Normalized price')
        plt.legend(loc='best')
        plt.grid(True)
        plt.show()

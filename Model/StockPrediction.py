import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM
import Model.preprocessing as preprocessing

def run_stock_prediction(filepath):
    # FOR REPRODUCIBILITY
    np.random.seed(7)

    # IMPORTING DATASET
    dataset = pd.read_csv(filepath, usecols=[1, 2, 3, 4])
    dataset = dataset.reindex(index=dataset.index[::-1])

    # TAKING DIFFERENT INDICATORS FOR PREDICTION
    OHLC_avg = dataset.mean(axis=1)
    obs = np.arange(1, len(dataset) + 1, 1)

    # SCALING DATA
    OHLC_avg = np.reshape(OHLC_avg.values, (len(OHLC_avg), 1))  
    scaler = MinMaxScaler(feature_range=(0, 1))
    OHLC_avg = scaler.fit_transform(OHLC_avg)

    # TRAIN-TEST SPLIT
    train_size = int(len(OHLC_avg) * 0.75)
    train_OHLC, test_OHLC = OHLC_avg[0:train_size, :], OHLC_avg[train_size:len(OHLC_avg), :]

    # TIME-SERIES DATASET
    step_size = 1
    trainX, trainY = preprocessing.new_dataset(train_OHLC, step_size)
    testX, testY = preprocessing.new_dataset(test_OHLC, step_size)

    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

    # LSTM MODEL
    model = Sequential()
    model.add(LSTM(64, input_shape=(1, step_size), return_sequences=True))
    model.add(LSTM(32, return_sequences=True))
    model.add(LSTM(16))
    model.add(Dense(1))
    model.add(Activation('linear'))

    model.compile(loss='mean_squared_error', optimizer='adam')  
    model.fit(trainX, trainY, epochs=50, batch_size=32, verbose=2)

    # PREDICTION
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)

    trainPredict = scaler.inverse_transform(trainPredict)
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])
    trainY = scaler.inverse_transform([trainY])

    # Calculate RMSE
    trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
    testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
    print('Train RMSE: %.2f' % trainScore)
    print('Test RMSE: %.2f' % testScore)

    # Predict Next Day
    last_value = test_OHLC[-1]
    last_value = np.reshape(last_value, (1, 1, 1))
    next_day_scaled = model.predict(last_value)
    next_day_value = scaler.inverse_transform(next_day_scaled)[0][0]
    print("Predicted Next Day Stock Price: {:.2f}".format(next_day_value))

    # PLOTTING GRAPH WITH NEXT DAY PREDICTION
    OHLC_avg = scaler.inverse_transform(OHLC_avg)
    plt.figure(figsize=(12, 6))
    plt.plot(OHLC_avg, 'g', label='Original dataset')
    plt.plot(np.arange(step_size, len(trainPredict) + step_size), trainPredict, 'r', label='Training predictions')
    plt.plot(np.arange(len(trainPredict) + step_size, len(trainPredict) + step_size + len(testPredict)), 
             testPredict, 'b', label='Test predictions')
    plt.axvline(x=len(OHLC_avg) - 1, color='k', linestyle='--', label='Prediction Point')
    plt.scatter(len(OHLC_avg), next_day_value, color='purple', label='Next Day Prediction')
    plt.legend(loc='upper left')
    plt.title('Stock Price Prediction with Next Day Value')
    plt.xlabel('Time in Days')
    plt.ylabel('OHLC Value of Stocks')
    plt.grid()

    # Save the plot
    output_path = 'static/output_graph.png'
    plt.savefig(output_path)
    plt.close()

    # Return the next day's prediction and graph path
    return next_day_value, output_path

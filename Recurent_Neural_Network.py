import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Prediksi saham google dari 2012 - 2016

# Importing the training set
dataset_train = pd.read_csv("assets/Google_Stock_Price_Train.csv")
training_set = dataset_train.iloc[:,1:2].values

# Feature Scaling

SC = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = SC.fit_transform(training_set)

# print(training_set_scale)

# Creating a data structure with 60 timestamps and 1 output

x_train = []
y_train = []

for i in range(60, len(training_set_scaled)):
    # akan berisi tumpukan array 1258 kolum 60 row
    x_train.append(training_set_scaled[i - 60:i, 0])

    y_train.append(training_set_scaled[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

# Reshaping dari 2d array ke 3d array ( untuk memvisualisasikan ke indikator )
# 3D Array (Tensor) => (batch_size, timesteps, prediktor/indikator)
# 2D Array (Matrics) => (batch_size, prediktor/indikator)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Building the RNN

RNN = Sequential([
    # LSTM(hidden_layer, return_sequences, input_shape) (Cukup 2D Array)
    # (default valuenye false) true karena kita membangun lstm yang bertumpuk dan akan memiliki beberapa lapisan lstm

    LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)),

    # Dropout(rate)
    # rate => jumlah neuron/hidden layer yang ingin di jatuhkan/abaikan dalam layer untuk melakukan regularasi

    Dropout(0.2),

    LSTM(50, return_sequences=True),
    Dropout(0.2),

    LSTM(50, return_sequences=True),
    Dropout(0.2),

    LSTM(50),
    Dropout(0.2),

    # Karena kita menghasilkan output bilangan rill jadi cukup 1 neuron saja
    Dense(1),
])

# mean_squared_error => mengukur Rata-rata Kesalahan kuadrat antara nilai aktual dan nilai peramalan.
# Metode Mean Squared Error secara umum digunakan untuk mengecek estimasi berapa nilai kesalahan pada peramalan
# RMSprop dan adam yang biasa di gunakan di RNN

RNN.compile(optimizer="adam", loss='mean_squared_error', metrics=['accuracy'])

# Pitting the rnn to training_set ( Jika loss sangat kecil 10 < bisa overfitting )

EPOCS = 100
RNN.fit(x_train, y_train, epochs=EPOCS, batch_size=32, verbose=1)

# Making the predictions and the visualization the result

# Getting the real stock price of 2017
dataset_test = pd.read_csv("assets/Google_Stock_Price_Test.csv")
real_stock_price = dataset_test.iloc[:,1:2].values

# Getting the predict stock price of 2017

# Gabungin 2 Column Open menjadi 1 column saja jadi => 2012 sampai januari 2017
dataset_total = pd.concat((dataset_train["Open"], dataset_test["Open"]), axis=0) # Concat 2 dataset

# Mengambil data dari batas 60 (data 2016) bawah sampai january 31 2017
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1, 1)

# Karena udh di fit jadi langsung di transform aja

inputs = SC.transform(inputs) # harusnya akan di hitung menjadi bilangan rill

x_test = []

for i in range(60, len(inputs)):
    x_test.append(inputs[i - 60:i, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predicted_price_stok = RNN.predict(x_test)
print("\nPredict : ", predicted_price_stok)

# inverse_transform => Untuk mengembalikan nilai yang sudah di scale MinMax menjadi nilai awal dataset
# Contoh => 0.89266247 menjadi 758.9796

predicted_price_stok = SC.inverse_transform(predicted_price_stok)
print("\nPredict With Inverse MinMax : ", predicted_price_stok)

# Visualising the result

plt.plot(real_stock_price, color="red", label="Real Google Stock Price")
plt.plot(predicted_price_stok, color="blue", label="Prediction Google Stock Price")
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()

# Conclusion
# Pada plot warna biru kenapa di akhir tidak mengikuti plot merah ada lonjakan besar seperti singularitas
# karena model kita bereaksi terhadap perubahan nonlinear ( tapi itu normal kok ) yang cepat
# di mana model kita tidak dapat bereaksi dengan baik ( tapi ga masalah )
# menurut konsep matematika Gerak Brown dalam teknik keuangan
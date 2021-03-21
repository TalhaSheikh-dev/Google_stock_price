# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 20:59:08 2018

@author: Ayyaz -ul- Haq
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os 
#os.chdir("D:\\Learning\\AI\\Deep Learning Projects\\Google Stock Price")
#Importing DataSet
data_frame_whole = pd.read_csv("HistoricalQuotes.csv")
data_frame_whole = data_frame_whole.iloc[:,3]

test_data_frame = data_frame_whole[:4]
data_frame = data_frame_whole[5:]

def reversing_colom (a):
    b=[]
    i= len(a)-1
    while i>=0:
        b.append(float(a.iloc[i][2:]))
        i-=1
    return b

"""
############################ PRE-PROCESSING ###################################
"""
#Function for getting the open column in reverse order (as 2014 to 2018)
def adding_dim(aa):
    aa = reversing_colom(aa)
    aa =np.expand_dims(aa, axis=1)
    return aa
#Preparing numpy array for NeuralNetwork by adding 1 dimension
training_set = adding_dim(data_frame)
test_data_frame = adding_dim(test_data_frame)
data_frame_whole = adding_dim(data_frame_whole)

# Feature Scaling (Normilization)
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range =(0,1))
training_set_scaled = sc.fit_transform(training_set)
test_data_frame = sc.transform(test_data_frame)
data_frame_whole = sc.transform(data_frame_whole)

#Creating x_train and y_train for 100 time steps
x_train = []
y_train = []
timesteps = 60

for i in range(timesteps,len(training_set)):
    x_train.append(training_set_scaled[i-timesteps:i,0])
    y_train.append(training_set_scaled[i,0])
    
x_train , y_train = np.array(x_train) , np.array(y_train)

#Reshape
x_train = np.reshape(x_train ,( x_train.shape[0],x_train.shape[1] ,1))


"""
############################# BUILDING RNN ####################################
"""
from keras.models import Sequential
from keras.layers import Dense,Dropout,LSTM

regressor = Sequential()

#Layer 1
regressor.add(LSTM(units = 1000 , return_sequences = True , input_shape = (x_train.shape[1] ,1 )))
regressor.add(Dropout(0.2))
#Layer 2
regressor.add(LSTM(units = 500 , return_sequences = True ))
regressor.add(Dropout(0.2))

#Layer 4
regressor.add(LSTM(units = 250 , return_sequences = True ))
regressor.add(Dropout(0.2))
#Layer 5
regressor.add(LSTM(units = 100 , return_sequences = True ))
regressor.add(Dropout(0.2))

#Layer 6
regressor.add(LSTM(units = 500))
regressor.add(Dropout(0.2))
#Final Layer
regressor.add(Dense(units =1))

#Optimizer
regressor.compile(optimizer = 'adam' , loss = 'mean_squared_error' )

"""
############################# Fitting Dataset #################################
"""
try:
    regressor.load_weights("stock_prize.hdf5")
except:
    # Fitting the RNN to the Training set
    regressor.fit(x_train, y_train, epochs = 100, batch_size = 32)
    regressor.save_weights("stock_prize.hdf5")



"""
###############################################################################
############################### PREDICTIONS ###################################
###############################################################################
"""

# Part 3 - Making the predictions and visualising the results

# Getting the predicted stock price of 2017
dataset_total = pd.DataFrame(data_frame_whole)
inputs = dataset_total[len(data_frame_whole) - len(test_data_frame) - 60:].values

X_test = []
for i in range(61, 65):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)
real_stock_price = sc.inverse_transform(test_data_frame)
# Visualising the results
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price-35, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()




























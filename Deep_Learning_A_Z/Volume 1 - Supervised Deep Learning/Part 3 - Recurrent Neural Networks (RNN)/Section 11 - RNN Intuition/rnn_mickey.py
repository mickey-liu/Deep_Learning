#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 21:29:57 2018

@author: mickey.liu
"""

"""RNN intuition:
    RNN is kind of like short term memory. Placing heavier weights on more recent observations. Used to represent time-series.
    Every node in an RNN represents a whole layer of nodes
    Types of RNNs:
        -one to many: (e.g. input = image, output = a sentence that describes the image)
        -many to one: (e.g. input = a sentence, output = sentiment of the sentence)
        -many to many (type I): (e.g. language translation (need short term info abt the prev word to determine the next word e.g. languages where gender matters))
        -many to many (type II): predicting next image in a movie
        
    The Vanishing Gradient Problem:
        -During backprop of an RNN, all the weights in the previous time-series that contributed to the current time-series output need to be updated. 
        -Because the Wrec (weight recurring) is close to 0 and during backprop, as we pass gradient backwards, it is used to update the weights and begins to 
            decrease since it's being multiplied to Wrec. The lower the gradient, the slower it'll update the weights. Hence the earlier part of the network is not 
            as trained as the recent parts of the RNN.
            If Wrec is large, then we have an exploding gradient problem
    
    Solution:
        1. Exploding Gradient
            -Truncated backprop, stop after a while
            -Penalties: have gradient be penalized and artificially reduced
            -Gradient Clipping: set a max gradient and never have it go over this value
        2. Vanishing Gradient
            -Weight initialization: initialize them in a smart way
            -Echo State Networks
            -LSTM networks (POPULAR)
            
    LSTM:
        -Wrec > 1: exploding gradient
        -Wrec < 1: vanishing gradient
        Solution: Wrec = 1
        
    LSTM module has:
        3 inputs:
            -Ct-1 (memory, aka cell state)
            -ht-1 output from previous module
            -Xt input 
        2 outputs:
            -ht 
            -Ct
        each input and ouput is in the form of a vector
        
        NN layers (simgoid and tanh)
        
        Several pointwise operations:
            -forget valve ft: controlled by sigmoid to determine what info from the previous cell state we keep or throw away by looking at ht-1 and Xt. ft * Ct-1
            -memory (input gate) valve it: controlled by sigmoid and tanh to decide what new info we're going to store in the current cell state. Sigmoid layer 
                           decides which values we'll update, tanh creates a new vector of values C~t that could be added to the state. it * C~t
                           Ct = (ft * Ct-1) + (it * C~t)
            -output valve ot: controlled by sigmoid to determine what part of the cell to we're going to output, then ->
            -tangent (tanh) operation applied on the Ct to push values between -1 and 1 to comput ht 
                           ht = ot * tanh(Ct) 
                           
        http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-chain.png
   
    LSTM Variations:
        1. Adding peepholes 
        2. Connected forget value and memory valve
        3. Gated Recurring Units: instead of having C, just have output h
    """
    
"""Predicting Stock Price with RNN and LSTM"""


#Part 1 - Data Preprocessing

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

#Import the training set 
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values #initialized open prices as a numpy array

#Feature Scaling
#Apply normalization (Recommended whenever sigmoid functions are used)
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1)) #all our scaled stock prices will be better 0 and 1
training_set_scaled = sc.fit_transform(training_set)

#Creating a data structure with 60 timesteps and 1 output. 60 timesteps: at each time t, RNN will look at 60 (Prev 3 months) stock prices before time t 
#and based on that trend, try to predict the next output t+1. This is based on experimentation.
X_train = [] #contain the 60 prices before current t
y_train = [] #next day stock price at t+1
for i in range(60, 1258):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i,0])

#turn X_train and y_train from lists into numpy arrays
X_train, y_train = np.array(X_train), np.array(y_train)

#Reshaping (for adding a dimension to any numpy array) since Keras RNN input shape is 3D ((batch_size, timesteps, input_dim))
#The dimension added is the unit: the number of predictors we can use to predict what we want。 Additional predictors could help predict the stock price even better
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1)) #here the last 1 means that we're only using 1 indicator: the open price 
#X_train shape before: (1198, 60)
#X_train shape after: (1198, 60, 1)


#Part 2 - Building RNN
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

#initializing the RNN
regressor = Sequential() #regression to predict continuous value

#Add first LSTM layer and some Dropout Regularization
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1))) 
#units = # of cells/neurons, 
#return_sequences: Boolean. Whether to return the last output in the output sequence, or the full sequence.
#return_sequences to TRUE since we're adding several LSTM layers, when you're adding your last one, this will be false or DEFAULT
#input_shape: shape of the input in 3 dimensions (batch_size, timesteps, indicators (input_dim)). We just need to inclue the timesteps and indicators in the args.
regressor.add(Dropout(0.2))


#Add second LSTM layer and some Dropout Regularization
regressor.add(LSTM(units = 50, return_sequences = True)) 
regressor.add(Dropout(0.2))

#Add third LSTM layer and some Dropout Regularization
regressor.add(LSTM(units = 50, return_sequences = True)) 
regressor.add(Dropout(0.2))

#Add fourth LSTM layer and some Dropout Regularization
regressor.add(LSTM(units = 50, return_sequences = False)) 
regressor.add(Dropout(0.2))
"""_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
lstm_1 (LSTM)                (None, 60, 50)            10400     
_________________________________________________________________
dropout_25 (Dropout)         (None, 60, 50)            0         
_________________________________________________________________
lstm_2 (LSTM)                (None, 60, 50)            20200     
_________________________________________________________________
dropout_26 (Dropout)         (None, 60, 50)            0         
_________________________________________________________________
lstm_3 (LSTM)                (None, 60, 50)            20200     
_________________________________________________________________
dropout_27 (Dropout)         (None, 60, 50)            0         
_________________________________________________________________
lstm_4 (LSTM)                (None, 50)                20200     
_________________________________________________________________
dropout_28 (Dropout)         (None, 50)                0         
=================================================================
Total params: 71,000
Trainable params: 71,000
Non-trainable params: 0
_________________________________________________________________"""

#Adding output layer
regressor.add(Dense(units = 1)) #unit = dimension of output layer

#Compiling RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
#usually rmsprop recommended for RNN

#Fitting regressor to X_train and y_train NOT dataset_train
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)
# loss at epoch 100 is ~ 0.0015 which is good. Because if loss is too small, then might be a sign of overfitting.


#Part 3 - Making predictions and visualizing results

#Get the real stock price of Jan 2017
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv') #this csv has all the stock info for Google for Jan 2017
real_stock_price = dataset_test.iloc[:, 1:2].values

#Getting the predicted stock price of Jan 2017
#We expect the prediction to get the general trend of the real stock price

#3 Key Points:
#1. To predict the price at each time t in Jan 2017, we'll still use the previous 60 stock prices 
#2. To accomplish #1, we'll use both the train and test sets
#3. Simply concat training_set and real_stock_price sets will result in a problem since we'll have to rescale the new aggregated set and thus the actual test values will change.
##  Instead, we should concat the dataset_train and dataset_test, then we'll scale the 60 prev inputs for each time t. This way, we'll only scale the input rather than changing actual test values.
##  Scaling is needed for consistency since our model was trained on scaled inputs
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0) #{0/’index’, 1/’columns’}, default 0
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values #first financial day of Jan minus 60 to last financial day of Jan
#current shape of inputs is (80,), need to reshape into np array
inputs = inputs.reshape(-1,1)
#inputs shape = (80, 1)

#scale inputs
inputs = sc.transform(inputs)

#Creating a data structure with 60 timesteps and 1 output. 60 timesteps: at each time t, RNN will look at 60 (Prev 3 months) stock prices before time t 
#and based on that trend, try to predict the next output t+1. This is based on experimentation.
X_test = [] #contain the 60 prices before current t
for i in range(60, 80): #get the 60 prev inputs for each day in Jan 2017
    X_test.append(inputs[i-60:i, 0])
#turn X_train and y_train from lists into numpy arrays
X_test = np.array(X_test)
#X_test size (20, 60)

#reshape to 3D
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

predicted_stock_price = regressor.predict(X_test)

#inverse scaling 
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

#Visualizing results
plt.plot(real_stock_price, color = 'green', label = 'Real Google Stock Price (Jan \'17)')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price (Jan \'17)')
plt.title('Google Stock Price: Real vs Predicted (Jan \'17)')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()


import math
from sklearn.metrics import mean_squared_error
rmse = math.sqrt(mean_squared_error(real_stock_price, predicted_stock_price))
##rmse = 13.881827820615973

"""you can do some Parameter Tuning on the RNN model we implemented.

Remember, this time we are dealing with a Regression problem because we predict a continuous outcome (the Google Stock Price).

Parameter Tuning for Regression is the same as Parameter Tuning for Classification which you learned in Part 1 - Artificial Neural Networks, the only difference is that you have to replace:

scoring = 'accuracy'  

by:

scoring = 'neg_mean_squared_error' 

in the GridSearchCV class parameters."""



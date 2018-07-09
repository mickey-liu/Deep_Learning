#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 15:59:53 2018

@author: mickey.liu
"""

"""Purpose here is to know how to combine two models. Hence ignore simplicity of this model"""

#Mega Case Study - Making a hybrid model

#Goal is to make a SOM and also predict the probability that customer is fraudulant 

import pandas as pd 
import numpy as np 

#Part 1 - Identify the Frauds with the SOM

# Importing the dataset
dataset = pd.read_csv('Credit_Card_Applications.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
X = sc.fit_transform(X)

# Training the SOM
from minisom import MiniSom
som = MiniSom(x = 10, y = 10, input_len = 15, sigma = 1.0, learning_rate = 0.5)
som.random_weights_init(X)
som.train_random(data = X, num_iteration = 100)

# Visualizing the results
from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o', 's']
colors = ['r', 'g']
for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.5,
         w[1] + 0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
show()

# Finding the frauds
mappings = som.win_map(X)
frauds = np.concatenate((mappings[(6,4)], mappings[(6,5)], mappings[(6,6)], mappings[(8,2)]), axis = 0)
frauds = sc.inverse_transform(frauds)

#Part 2 - going from unsupervised to supervised DL

#Creating a matrix of features
customers = dataset.iloc[:, 1:].values #creating a matrix that contains all dataset columns including whether application was approved or not, but excluding customer ID


#Creating the dependent variable
#We need a dependent variable that is a binary value of whether the customer was marked as fraudulant or not
is_fraud = np.zeros(len(dataset)) #initialize a vector of the same size as dataset and set them to 0

#loop over all dataset and see if the customer_id is flagged as fraud or not
for i in range(len(dataset)):
    if dataset.loc[i, "CustomerID"] in frauds:
        is_fraud[i] = True

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
customers = sc.fit_transform(customers)

# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 2, kernel_initializer = 'uniform', activation = 'relu', input_dim = 15))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(customers, is_fraud, batch_size = 1, epochs = 2)
##Accuracy after 2 epochs is already 91%. Loss did reduce from 50% to 30%


# Predicting the probabilities of frauds
y_pred = classifier.predict(customers)
#merge the y_pred with the customerIDs
predictions = np.concatenate((dataset.iloc[:,0:1].values, y_pred), axis=1)
# Sort the predictions
predictions = predictions[predictions[:,1].argsort()]
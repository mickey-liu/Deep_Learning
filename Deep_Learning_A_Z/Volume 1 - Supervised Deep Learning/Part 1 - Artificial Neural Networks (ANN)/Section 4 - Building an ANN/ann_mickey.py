
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 13:43:51 2018

@author: mickey.liu
"""

"""ANN Deep Learning (Same as Machine Learning course)

"""

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

#Part 1 - Data Preprocessing

dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

#Encoding Categorical data
from sklearn.preprocessing import OneHotEncoder, LabelEncoder #class objects, need to be initialized
labelencoder_X_1 = LabelEncoder()
X[:,1] = labelencoder_X_1.fit_transform(X[:,1]) #encodes the Country
labelencoder_X_2 = LabelEncoder()
X[:,2] = labelencoder_X_1.fit_transform(X[:,2]) #encodes the gender
onehotencoder = OneHotEncoder(categorical_features = [1]) #only done for country column since there's 2+ categories
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:] #removes dummy var trap for country columns by removing 1 of the 3 country columns from X


#Splitting dataset into training and test sets
from sklearn.model_selection import train_test_split #method (to be applied on the objects)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Part 2 - Creating ANN
import keras 
from keras.models import Sequential #To initialize ANN
from keras.layers import Dense #to create layers
from keras.layers import Dropout

#initialize ANN
classifier = Sequential()
#Add First Layer + Hidden Layer with Dropout
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11)) #Dense takes care of randomly initialized weights close to 0 
#Dense implements the operation: output = activation(dot(input, kernel) + bias) where  activation is the element-wise activation 
#function passed as the activation argument, kernel is a weights matrix created by the layer, and bias is a bias vector created 
#by the layer (only applicable if use_bias is True).
classifier.add(Dropout(rate = 0.1)) #keras.layers.Dropout(rate (% of neurons to be dropped, starting with 10%, then 20%. Too high would result in underfitting)
#, noise_shape=None, seed=None)

#Add Hidden Layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
classifier.add(Dropout(rate = 0.1))

#Add Output Layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

#Compile the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
"""Before training model, need to configure learning process, which is done via compile method. It receives 3 args:

1. An optimizer. This could be the string identifier of an existing optimizer (such as rmsprop or  adagrad), or an instance of the Optimizer class. See: optimizers.
2. A loss function. This is the objective that the model will try to minimize. It can be the string identifier of an existing loss function (such as categorical_crossentropy or mse), or it can be an objective function. See: losses.
3. A list of metrics. For any classification problem you will want to set this to  metrics=['accuracy']. A metric could be the string identifier of an existing metric or a custom metric function."""

#Fit model to training set
classifier.fit(X_train, 
               y_train, 
               batch_size = 50, #updates weights after each batch
               epochs = 500 #A round in which the whole training set passes the ANN
               )
#Keras models are trained on Numpy arrays of input data and labels. For training a model, you will typically use the  fit function. 
#fit(self, x=None, y=None, batch_size=None, epochs=1, verbose=1, callbacks=None, validation_split=0.0, validation_data=None, 
#shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None)

classifier.fit(X_train, 
               y_train, 
               batch_size = 50, #updates weights after each batch
               epochs = 500 #A round in which the whole training set passes the ANN
               )


# Part 3 - Making predictions and evaluating the model
# Predicting the Test set results
y_pred = classifier.predict(X_test) 
#since y_pred are probabilities, we'll need to decide on a threshold to categorize the y_pred as 0 or 1.
y_pred = (y_pred > 0.5) #if y_pred > 0.5, return true

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
"""acc_test: 0.856"""


"""Predicting on the following customer: 
    Geography: France
Credit Score: 600
Gender: Male
Age: 40 years old
Tenure: 3 years
Balance: $60000
Number of Products: 2
Does this customer have a credit card ? Yes
Is this customer an Active Member: Yes
Estimated Salary: $50000"""

X_new = np.matrix(X[0, :])
X_new[0,0] = 0
X_new[0,2] = 600
X_new[0,3] = 1
X_new[0,4] = 40
X_new[0,5] = 3
X_new[0,6] = 60000
X_new[0,7] = 2
X_new[0,8] = 1
X_new[0,9] = 1
X_new[0,10] = 50000
X_new = sc.transform(X_new)
y_new = classifier.predict(X_new) > 0.5
"""False"""

#Array needs to be in a horizontal vector
#X_new_pred = np.array([[0,0,600,1,40,3,60000,2,1,1,50000]]) #Double-brackets indicate horizontal array
new_prediction = classifier.predict(sc.transform(np.array([[0,0,600,1,40,3,60000,2,1,1,50000]])))
new_prediction = (new_predict > 0.5)

###################################

#Part 4 - Evaluating Performance with K-Fold CV
"""Bias-Variance trade-off:
    We want the accuracy to be high and the variance to be low (aka everytime we train the model, the accuracy is about the same).
    K-Fold fixes this by breaking up training sets into k-folds, usually 10, and train on 9 of them and test on 1 of them. Then take the average"""
    
#K-Fold will include the ANN so we only need to execute the code up to the above ANN initialization.
#Need to combine sklearn (k-fold) and keras (ANN) together using a keras wrapper that'll wrap the k-fold sklearn into a keras model.
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential #To initialize ANN
from keras.layers import Dense #to create layers
#Build the ANN architecture with a function
def build_classifier(): #This classifier is a local variable, without training
    classifier = Sequential()
    classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11)) #Dense takes care of randomly initialized weights close to 0 
    classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
    classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 100) #global classifier that defines the classifier and the batch_size/# of epochs
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = 1)
mean = accuracies.mean() #~84%
variance = accuracies.std() #~1%
#Can we do better in terms of accuracy?

#Improving ANN 
#Dropout regularization to reduce overfitting if needed (overfitting: high variance from accuracies vector)
#Randomly disable neurons at each hidden layer (Implemented above in the ANN)

#Tuning the ANN:
#Param-Tuning: tuning the hyperparameters (batch_size, n_epochs, optimizer, # of neurons in the layers). Maybe we can get better results with other values
#Use GridSearch
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV #instead of cross_val_score
from keras.models import Sequential 
from keras.layers import Dense 
#Build the ANN architecture with a function
def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11)) #Dense takes care of randomly initialized weights close to 0 
    classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
    classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier) #removed batch_size, epochs to move them into the GridSearch args
#create dictionary (structure of data that contains keys, with each key can contain several values) of hyperparams to optimize
parameters = {'batch_size': [25, 32], 
              'epochs': [100, 500],
              'optimizer': ['adam', 'rmsprop'] } #need to add arg into the build_classifier function to be able to pass the optimizer

#Implement GridSearch with K-Fold CV
grid_search = GridSearchCV(estimator = classifier, 
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)
grid_search = grid_search.fit(X_train, y_train)
#Get the following 2 attributes of grid_search
best_accuracy = grid_search.best_score_ 
best_parameters = grid_search.best_params_
#more attributes
best_cv_results = grid_search.cv_results_
best_estimator = grid_search.best_estimator_
best_index = grid_search.best_index_
"""best_accuracy ~ 85% with
    best_params = batch_size=25, epochs = 500, optimizer = rmsprop.
    This is good, but we can do better. Possibly by changing the architecture of the ANN."""
    
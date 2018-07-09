#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 14:22:53 2018

@author: mickey.liu
"""

"""Self-Organizing Maps intuition:
    Used for reducing dimensionality. End up with a 2D output from a multi-dimension input layer. Helps visualize.
    This is an unsupervised DL
    
    Refresh on K-Means clustering:
        -Helps understanding of SOMs and an unsupervised algorithm
        -K-means can work with multi-dimensions
        
        Step 1: Choose # of clusters K
        Step 2: Select at random K points, centroids (not necessarily from dataset)
        Step 3: Assign each datapoint to the closest (euclidean or whichever distance fit for the business problem) centroid, forming K clusters
        Step 4: Compute and place new centroid of each cluster
        Step 5: Reassign each datapoint to the closest centroid. If any reassignment took place, repeat from step 4. Otherwise FIN
        
        -Use K-means++ algorithm to avoid random initialization trap to avoid non-desirable clustering
        -Use elbow method of the Within Cluster SUm of Squares to get the optimal # of clusters.
    
    How do SOMs learn? 
    -Using a NN
    -Input and output nodes with the output nodes being the maps
    -Input layer may be 3-D, multiple features and multiple layers
    -Output is 2-D 
    -Terms that were used in ANN/CNN/RNN have slightly diff meanings in SOMs
    -Weights in SOMs are a characteristic of the node itself e.g. output node 1: (W1,1; W1,2; W1,3) similar to coordinates. 
        Weights are not multiplied by the input node, no activation function
    -Output nodes are trying to see where they can fit in the input space
    -For each observation (row in the dataset), we find the output node that's closest to it (best matching unit). Distance = sqrt(sum(xi - wj,i)^2)
        The distance will typically be close to 1 since the input values should be scaled (0 < x < 1)
    -Then, we reduce the weights of the that bmu and also reduce the weight of the nearby nodes within a certain radius. The closer the nodes are to
        the BMU, the heavier their weights are reduced. Think of a drag.
    -We do this "pull" for each of the observation's bmu's
    -The radiuses are the same for all the bmu's of an epoch
    -After each epoch, the radius decreases, hence the bmu's are pulling on fewer nodes.
    -SOM's slowly turn into a representation
    
    Actual Steps:
        1. Start w/ dataset composed of n_features ind var
        2. create a grid composed of nodes, each one having a weights vector of n_feature elements
        3. randomly init weights close to 0
        4. select 1 random observation from the dataset
        5. compute euclidean distances from this point to diff neurons in the network
        6. select the neuron with the min distance to the point. That neuron is the winning node (BMU)
        7. update weights of winning node to move it closer to the point
        8. Using a Gaussian neighboring function of meean the winning node, update weights of winning node neighbors to move them closer 
            to the point. Neighboring radius is the sigma in the Gaussian function
        9. Repeat 1-5 and update weights after each observation or after a batch of observations, until network converges to a point where 
            the neighborhoods stop learning
    
    
    
    
    IMPORTANT:
        -SOM's retain topology of the input set
        -SOM's reveal correlations that are not easily identified
        -SOM's are unsupervised, no labels needed
        -No target vector -> no backprop
        -No lateral connection btw output nodes (no activation functions connecting output nodes)
        
    Resources:
        -D3.js for creating SOMs in javascript
    """

### Explict objective: Detect fraud from list of bank applicants applying for a CC
    
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('Credit_Card_Applications.csv')
#dataset has a combo of categorical and continuous ind var and the last column indicating whether CC application was approved or not.
#SOM will do some customer segmentation to ID segments of customers. And 1 of the segments will be for customers that cheated. 
#Frauds will be the outlying neurons in the SOM

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
#we did to make distinctions in the end. But since SOM is unsupervised, we'll only use X to train the model

#Feature Scaling - Normalization
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
X = sc.fit_transform(X)

#Training the SOM
#We'll use an Minisom 1.0 open-source library here (minisom.py)
from minisom import MiniSom
som = MiniSom(x = 10, y = 10, input_len = 15, sigma = 1.0, learning_rate = 0.5) #x,y = grid of the map, input_len # of features in dataset X,
# sigma = radius of neighborhoods in the grid (default 1.0), learning_rate: how much the weights are updated (default 0.5), decay_function)

#initialize the weights to small numbers close to 0
som.random_weights_init(X)

som.train_random(data = X, num_iteration = 100) #data = X, num_iteration: # of iterations we want to repeat steps 4-9

#Visualizing results
#See all the final winning nodes and we'll get the Mean Interneuron Distance for each winning node
#MID for a winning node is the mean of the distances of all the neurons around the winning node inside the neighborhood defined by sigma
#higher MID means the more winning node will be far away from its neighbors, aka winning node is outlier (fraud)
from pylab import bone, pcolor, colorbar, plot, show

#initialize the figure (window that contains map)
bone()
pcolor(som.distance_map().T) #take the transpose of the distance maps of the MID
colorbar()
#add markers to the customers with winning nodes (fraud) that were approved for the CC
markers = ['o', 's'] #o is circle (not approved), s is square (approved)
colors = ['r', 'g'] #assigning red to circle and green to square
for i, x in enumerate(X): #i = index of customers, x = vector of each customer
    w = som.winner(x) #som.winner(x) returns the coordinates of SOM grid for the winning node of the customer vector 
    plot(w[0] + 0.5, #take the x coordinate of winning node (+0.5 will take the center of that coordinate)
         w[1] + 0.5, #take the y coordinate of the winning node (+0.5 will take the center of that coordinate)
         markers[y[i]],
         markeredgecolor = colors[y[i]], #look up whether customer x got approved in the y matrix and set the marker/color 
         markerfacecolor = 'None', #no marker face since each node can have multiple markers
         markersize = 10,
         markeredgewidth = 2) 
show()

#Find the frauds
#use a dictionary to help find the frauds
mappings = som.win_map(X) #creates a dictionary of all nodes that contain customer vectors and the list of customers in each node

#get the winning nodes
frauds = np.concatenate([mappings[(6,4)], mappings[(1,4)]], axis=0)
frauds = sc.inverse_transform(frauds)
frauds_list = []
for i in range(frauds.shape[0]):
    for j in range(dataset.shape[0]):
        if int(frauds[i][0]) == dataset.loc[j,"CustomerID"]:
            frauds_list.append(dataset.iloc[j,[0,15]].values)
        
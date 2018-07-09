#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 15:39:13 2018

@author: mickey.liu
"""

"""Boltzman machine intution:
    
    ANN, CNN, RNN and SOMs are directive. aka we know where the information is flowing to. BM are diff, no arrows. 
    
    BM:
        1. Hidden Nodes
        2. Visible Nodes: are kind of like the input layer, except they don't just pass data to hidden layer. They also generate data/information
        No output nodes. All the hidden and visible nodes are connected to each other.
        
    The BM tries to learn how all of the nodes interact with each other and adjusts the weights of the connections of all the nodes.
    No outputs. The BM creates a model that describes our system.
    
    Used for recommendation systems
    2 types:
        1. energy-based models
        2. restricted boltzmann machines
    """
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 16:07:44 2020

@author: ankursrivastava
"""

# Artificial Neural Network using Keras
# Using Housing Data

import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix


dataset = pd.read_csv('Churn_Modelling.csv')

x = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# handle categorical data
labelEncoder_x_1 = LabelEncoder()
x[:, 1] = labelEncoder_x_1.fit_transform(x[:, 1])

labelEncoder_x_2 = LabelEncoder()
x[:, 2] = labelEncoder_x_2.fit_transform(x[:, 2])

oneHotEncoder = OneHotEncoder(categorical_features=[1])
x = oneHotEncoder.fit_transform(x).toarray()

x = x[:, 1:]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, 
                                                    random_state = 0)

# feature scaling
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)


# Keras
classifier = Sequential()
# Add two hidden layers
# Tip output_dim - avg of no of nodes in input layer and avg in output layer 
# which comes out = (11+1) / 2
# or use parameter tuning
classifier.add(Dense(activation = 'relu', units = 6, kernel_initializer = 'uniform'))
classifier.add(Dense(activation = 'relu', units = 6, kernel_initializer = 'uniform'))

# Add output layer
classifier.add(Dense(activation = 'sigmoid', units = 1, kernel_initializer = 'uniform'))

# Compile ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', 
                   metrics = ['accuracy'])

# Fit ANN to train data
classifier.fit(x_train, y_train, batch_size = 10, epochs = 100)

# Predict

y_pred = classifier.predict(x_test)
y_pred = (y_pred > 0.5)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
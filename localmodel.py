#Import
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Input

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import get_data
from sklearn.model_selection import train_test_split


#Get images and Labels
features,labels=get_data.load_data()

#Featuers -> Bilder (bekommt die Ki)
print(features)
print("----------------------------")
#Labels ->0-25 bzw. A-Z (soll di Ki zuordnen)
print(labels)


#Split data
train_features, test_features, train_labels, test_labels = train_test_split(
    features,
    labels,
    test_size=0.1,      # 10 % Testdaten
    random_state=42,    # reproduzierbar
    shuffle=True
)


#Model architecutre
model = Sequential([
    Input(shape=(32, 32)),     # images are 32x32
    Flatten(),                 # becomes 784
    Dense(64, activation='relu'), #?
    Dense(128, activation='relu'), #?
    Dense(26, activation='softmax')   #26 end layers
])
#Compile model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#Train model
print(type(train_features))
print("--------------")
print(type(train_labels))
model.fit(train_features, train_labels, epochs=50, batch_size=64)
#Test models accuracy
model.evaluate(test_features,test_labels)
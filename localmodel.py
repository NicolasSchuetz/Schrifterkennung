#Import
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Input, Dropout, Conv2D, MaxPooling2D

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import get_data
from sklearn.model_selection import train_test_split


#Get images and Labels
features,labels=get_data.load_tensorflow_data()

#Featuers -> Bilder (bekommt die Ki)
#print(features)
#print("----------------------------")
#Labels ->0-25 bzw. A-Z (soll di Ki zuordnen)
#print(labels)

#Split data
train_features, test_features, train_labels, test_labels = train_test_split(
    features,
    labels,
    test_size=0.1,      # 10 % Testdaten
    random_state=42,    # reproduzierbar
    shuffle=True
)

#Model architecutre (https://adamharley.com/nn_vis/)
model = Sequential([
    # Input: 32x32 grayscale image
    Input(shape=(32, 32, 1)),
    # 1. Convolution Block
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),   # 32x32 -> 16x16
    # 2. Convolution Block
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),   # 16x16 -> 8x8
    # 3. Convolution Block
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),   # 8x8 -> 4x4 
    # Flatten
    Flatten(),                              #Macht Array 1D
    # Fully Connected Layer
    Dense(128, activation='relu'),          #Hiddenlayers
    # Output Layer (26 Klassen)
    Dense(26, activation='softmax')         #Outputlayers (A-Z)
])

#Compile model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#Train model
#print(type(train_features))
#print("--------------")
#print(type(train_labels))
model.fit(train_features, train_labels, epochs=50, batch_size=64)
#Test models accuracy
loss,accuracy=model.evaluate(test_features,test_labels)
print(f"Loss: {loss}, Accuracy: {accuracy}")
model.save("model.keras")
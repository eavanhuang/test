#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 17:59:03 2019

@author: pteam
"""

from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers import Dense,Dropout,Convolution2D,MaxPooling2D,Flatten
from keras.optimizers import SGD
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.layers.recurrent import SimpleRNN
input_size=28
time_steps=28
cell_size=50

(train_image,train_labels),(test_image,test_labels)=mnist.load_data()
train_image=train_image/255.0
test_image=test_image/255.0

test_labels=np_utils.to_categorical(test_labels,num_classes=10)
train_labels=np_utils.to_categorical(train_labels,num_classes=10)

model = Sequential()

model.add(SimpleRNN(
        units=cell_size,
        input_shape=(time_steps,input_size)
        ))

model.add(Dense(10,activation='relu'))
adam=Adam(lr=1e-4)

model.compile(
        optimizer= adam,
        loss='categorical_crossentropy',
        metrics=['accuracy']
        )
model.fit(train_image,train_labels,batch_size=64,epochs=10)


loss,accuracy = model.evaluate(test_image,test_labels)

print("\ntest loss",loss)
print("accuracy",accuracy)
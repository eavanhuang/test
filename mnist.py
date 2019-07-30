#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 14:37:22 2019

@author: pteam
"""
#from keras.datasets import boston_housing
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers import Dense,Dropout,Convolution2D,MaxPooling2D,Flatten
from keras.optimizers import SGD
from keras.regularizers import l2
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from keras.utils.vis_utils import plot_model

(train_image,train_labels),(test_image,test_labels)=mnist.load_data()
train_image=train_image.reshape(-1,28,28,1)/255.0
test_image=test_image.reshape(-1,28,28,1)/255.0

test_labels=np_utils.to_categorical(test_labels,num_classes=10)
train_labels=np_utils.to_categorical(train_labels,num_classes=10)

model = Sequential()
"""
        Dense(units=200,input_dim=784,activation='tanh',kernel_regularizer=l2(0.0003)),

        Dense(units=100,activation='tanh',kernel_regularizer=l2(0.0003)),

        Dense(units=10,activation='softmax',kernel_regularizer=l2(0.0003))
"""
       

model.add(Convolution2D(
        input_shape=(28,28,1),
        filters=32,
        kernel_size=5,
        strides=1,
        padding='same',
        activation='relu',
        ))

model.add(MaxPooling2D(
        pool_size=2,
        strides= 2,
        padding='same'
        ))
model.add(Convolution2D(64,5,strides=1,padding='same',activation='relu'))
model.add(MaxPooling2D(2,2,'same'))
model.add(Flatten())
model.add(Dense(1024,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10,activation='softmax'))

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


plot_model(model,to_file="model.png",show_shapes=True,show_layer_names='False',rankdir='T8')
plt.figure(figsize=(10,10))
img=plt.imread("model.png")
plt.axis("off")
plt.show()
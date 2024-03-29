# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from keras.datasets import boston_housing
from keras import layers
from keras import models

(train_data,train_targets),(test_data,test_targets)=boston_housing.load_data()

mean=train_data.mean(axis=0)
train_data-=mean
std=train_data.std(axis=0)
train_data/=std

test_data-=mean
test_data/=std

def build_model():
    model=models.Sequential()
    model.add(layers.Dense(64,activation='relu',input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64,activation='relu'))
    model.add(layers.Dense(1))

    
    model.compile(optimizer='rmsprop',loss='mse',metrics=['mae'])
    return model

#K-fold
import numpy as np
import matplotlib.pyplot as plt
k=4
num_val_samples=len(train_data)//k
num_epochs=500
all_mae_socore=[]

for i in range(k):
    print('processing fold #',i)
    val_data=train_data[i*num_val_samples:(i+1)*num_val_samples]
    val_targets=train_targets[i*num_val_samples:(i+1)*num_val_samples]
    
    
    partial_train_data=np.concatenate([train_data[:i*num_val_samples],train_data[(i+1)*num_val_samples:]],axis=0)
    partial_train_targets=np.concatenate([train_targets[:i*num_val_samples],train_targets[(i+1)*num_val_samples:]],axis=0)
    
    model=build_model()
    history= model.fit(partial_train_data,partial_train_targets,batch_size=1, epochs= num_epochs,verbose=0)
   # print(history)
    #val_mae,val_mse=model.evaluate(val_data,val_targets,verbose=0)

    mae_history=history.history['val_mean_absolute_error']
    all_mae_socore.append(mae_history)
    average_mae_history=[
            np.mean([x[i] for x in all_mae_socore]) for i in range(num_epochs)]


    plt.plot(range(1,len(average_mae_history)+1),average_mae_history)
    plt.xlabel('EPOCHS')
    plt.ylabel('validation Mae')
    plt.show()

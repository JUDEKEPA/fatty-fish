# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 16:55:21 2020

@author: lenovo
"""


import matplotlib.pyplot as plt
import numpy as np
from numpy import newaxis
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
from keras.layers import LSTM, Dense, Input
import os
import pandas as pd
from matplotlib import pyplot


data = pd.read_excel('E:\F Disk\Project\深度学习股票预测\模型测\数据.xlsx')  #数据文件导入


time_step = 20  #预测的时间步
train_begin = 0  #预测例子行数的开始
train_end = 2000  #预测例子行数的结束


x_Unnormalized = data.iloc[:,1: 13].values  #所有x的变量导入
y_Unnormalized_undone = data.iloc[:,8:9].values  #所有y的变量导入


x_Normalized = (x_Unnormalized-np.mean(x_Unnormalized,axis=0))/np.std(x_Unnormalized,axis=0)  #x标准化
y_Normalized = (y_Unnormalized_undone-np.mean(y_Unnormalized_undone,axis=0))/np.std(y_Unnormalized_undone,axis=0) #y标准化



#===================================
#对训练集数据处理
x_train = []
y_train = []
for i in range(20, 2000):   #数据排列按照时间从前到现在
    x_train.append(x_Normalized[i - time_step:i])   #训练集从时间步长的前面开始
    y_train.append(y_Normalized[i, 0])  #预测的为之后的一天
x_train, y_train = np.array(x_train), np.array(y_train)   #将两个列表修改为数组


#===================================
#对测试集数据处理 代码内容与训练集相同
x_test = []
y_test = []
for i in range(2020, 2400):
    x_test.append(x_Normalized[i - 20:i])
    y_test.append(y_Normalized[i, 0])
x_test, y_test = np.array(x_test), np.array(y_test)


print(len(data))
mean = np.mean(y_Unnormalized_undone,axis=0) #Label均值
std = np.std(y_Unnormalized_undone,axis=0)  #Label方差


print(x_train.shape)
print(y_train.shape)


'''
#x_train = np.reshape(x_Normalized, (x_Normalized.shape[0], x_Normalized.shape[1], 1))
x_Done = []
for i in range(len(x_Unnormalized)-time_step+1):
    if i == 0:
        continue
    x = x_Normalized[i:i+time_step,:]
    x_Done.append(x.tolist())
y_Done = y_Normalized[begin: end,:]
'''

'''
train_begin = 0
train_end = 2000
test_begin = 2000
test_end = 2400


train_x_Normalized = x_Done[: train_end]
train_y_Normalized = y_Done[: train_end]


test_x_Normalized = x_Done[test_begin: test_end]
test_y_Normalized = y_Done[test_begin: test_end]


train_x_Normalized_array = np.asarray(train_x_Normalized)
train_y_Normalized_array = np.asarray(train_y_Normalized)


test_x_Normalized_array = np.asarray(test_x_Normalized)
test_y_Normalized_array = np.asarray(test_y_Normalized)
'''


#======================================
#神经网络建立
#HIDDEN_DIM = 512

LAYER_NUM = 10
model = Sequential()
model.build((None, 20, 12))
model.add(LSTM(20, activation='linear', input_shape=(None, 20, 12), return_sequences = True))
model.add(LSTM(100, return_sequences = False))
model.add(Dense(1))
model.add(Activation('linear'))
model.compile(loss="mse", optimizer="rmsprop") #rmsprop
model.summary()
BATCH_SIZE = 10
epoch = 1
history=model.fit(x_train, y_train ,epochs=300, batch_size=10, validation_data=(x_test, y_test), verbose=1, shuffle=False)

'''
model=Sequential()
model.build((None, 20, 12))
model.add(LSTM(50, activation='tanh', input_shape=(None, 20, 12), return_sequences=True))
model.add(LSTM(100, return_sequences=False, activation='linear'))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
model.summary()
history=model.fit(x_train, y_train ,epochs=30, batch_size=48, validation_data=(x_test, y_test), verbose=1, shuffle=False)
'''
y1_pred = model.predict(x_test)
#y_pred=y_pred*std[11]+mean[11]
y2_pred = model.predict(x_train)
#y_test=y_test*std[11]+mean[11]
print('MSE Train loss:', model.evaluate(x_train, y_train, batch_size = BATCH_SIZE))
print('MSE Test loss:', model.evaluate(x_test, y_test, batch_size = BATCH_SIZE))

plt.plot(y_test, label='test')
plt.plot(y1_pred, label='pred')
plt.legend()
plt.show()

plt.plot(y_train, label='test')
plt.plot(y2_pred, label='pred')
plt.legend()
plt.show()

plt.subplot(2,1,1)
plt.plot(history.history['loss'],label='train')
plt.plot(history.history['val_loss'],label='test')
plt.title('loss and val_loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend()
plt.show()







'''
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

# start with first frame
curr_frame = x_test[0]
 
# start with zeros
# curr_frame = np.zeros((100,1))
 
predicted = []
SEQ_LENGTH = 100
for i in range(len(x_test)):
    predicted.append(model.predict(curr_frame[newaxis, :, :])[0, 0])
    curr_frame = curr_frame[1:]
    curr_frame = np.insert(curr_frame, [SEQ_LENGTH - 1], predicted[-1], axis=0)
predicted1 = model.predict(x_test)
predicted1 = np.reshape(predicted1, (predicted1.size,))
 
plt.figure(1)
plt.subplot(211)
plt.plot(predicted)
plt.plot(y_test)
plt.subplot(212)
plt.plot(predicted1)
plt.plot(y_test)
plt.show()
'''
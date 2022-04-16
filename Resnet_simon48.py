# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 20:31:06 2020

@author: 18744
"""

import simon48 as sm
import pickle
import numpy as np


from keras import models
from keras import layers
from keras import optimizers
from keras.models import Model
from keras.layers import Dense, Conv1D, Input, Reshape, Permute, Add, Flatten, BatchNormalization, Activation
from keras.callbacks import ModelCheckpoint
from keras.regularizers import l2

from keras.models import Model
from keras.utils import plot_model

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"#选定进行计算的显卡

import tensorflow as tf
config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth=True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)#设定显卡针对数据量自动分配显存

def create_model():
    num_blocks=int(sm.block_size()/sm.WORD_SIZE())
    num_filters=sm.block_size()
    num_outputs=1
    d1=sm.block_size()*2
    d2=sm.block_size()*2
    word_size=sm.WORD_SIZE()
    ks=3
    reg_param=0.0001
    inp = Input(shape=(num_blocks * word_size * 2,))
    rs = Reshape((2 * num_blocks, word_size))(inp)
    perm = Permute((2,1))(rs)
    #add a single residual layer that will expand the data to num_filters channels
    #this is a bit-sliced layer
    conv0 = Conv1D(num_filters, kernel_size=1, padding='same', kernel_regularizer=l2(reg_param))(perm)
    conv0 = BatchNormalization()(conv0)
    conv0 = Activation('relu')(conv0)
    
    #add residual blocks
    shortcut = conv0
    for i in range(5):
        conv1 = Conv1D(num_filters, kernel_size=ks, padding='same', kernel_regularizer=l2(reg_param))(shortcut)
        conv1 = BatchNormalization()(conv1)
        conv1 = Activation('relu')(conv1)
        conv2 = Conv1D(num_filters, kernel_size=ks, padding='same',kernel_regularizer=l2(reg_param))(conv1)
        conv2 = BatchNormalization()(conv2)
        conv2 = Activation('relu')(conv2)
        shortcut = Add()([shortcut, conv2])
    #add prediction head
    flat1 = Flatten()(shortcut)
    dense1 = Dense(d1,kernel_regularizer=l2(reg_param))(flat1)
    dense1 = BatchNormalization()(dense1)
    dense1 = Activation('relu')(dense1)
    
    dense2 = Dense(d2, kernel_regularizer=l2(reg_param))(dense1)
    dense2 = BatchNormalization()(dense2)
    dense2 = Activation('relu')(dense2)
    
    out = Dense(num_outputs, activation='sigmoid', kernel_regularizer=l2(reg_param))(dense1)
    model = Model(inputs=inp, outputs=out)
    return(model)

def train_model(num,Round):
    diff=sm.DIFF()
    net_name=str(diff)
    x_round=Round
    data_x,data_y=sm.make_train_data(num,x_round)
    data_x=data_x.astype(np.uint8)#转换格式
    data_y=data_y.astype(np.uint8)

    x_val,y_val=sm.make_train_data(int(num/100),x_round)
    x_val=x_val.astype(np.uint8)#转换格式
    y_val=y_val.astype(np.uint8)

    #开始训练
    seed=199847
    np.random.seed(seed)
    model=create_model()
    model.compile(optimizer='adam',loss='mse',metrics=['acc'])
    filepath_net=net_name+'weight'+str(Round)+'.h5'
    checkpoint=ModelCheckpoint(filepath=filepath_net,monitor='val_acc',verbose=1,save_best_only=True,mode='max')
    callback_list=[checkpoint]
    history=model.fit(data_x,data_y,validation_data=(x_val,y_val),epochs=100,batch_size=10000,verbose=1,callbacks=callback_list)
    model_json=model.to_json()
    with open(net_name+'model'+str(Round)+'.json','w') as file:
                file.write(model_json)
    with open(net_name+str(Round)+'.txt','wb') as file:        
        pickle.dump(history.history,file)
#train_model(10**8,10)

"""
Created on Thu Apr 22 11:17:43 2020

@author: deeplearning

The code is related to 'Practical Attacks of Round-Reduced SIMON Based on Deep Learning', which is submitted to The Computer Journal.
The code is also related to 'Improve Neural Distinguisher for Cryptanalysis' (https://eprint.iacr.org/2021/1017).
If you want to use the code, please refer to 'Practical Attacks of Round-Reduced SIMON Based on Deep Learning' or 'Improve Neural Distinguisher for Cryptanalysis'.
"""
#This code is used to train neural distinguishers of SIMON48/96.
import simon_48 as sm
import pickle
import numpy as np

from keras.models import Model
from keras.layers import Dense, Conv1D, Input, Reshape, Permute, Add, Flatten, BatchNormalization, Activation
from keras.callbacks import ModelCheckpoint
from keras.regularizers import l2

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf
config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth=True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)

def create_model():
    num_blocks=int(sm.block_size()/sm.WORD_SIZE())
    num_filters=sm.block_size()
    num_outputs=1
    d=sm.block_size()*2
    word_size=sm.WORD_SIZE()
    ks=3
    reg_param=0.0001
    inp = Input(shape=(num_blocks * word_size * 2,))
    rs = Reshape((2 * num_blocks, word_size))(inp)
    perm = Permute((2,1))(rs)

    conv0 = Conv1D(num_filters, kernel_size=1, padding='same', kernel_regularizer=l2(reg_param))(perm)
    conv0 = BatchNormalization()(conv0)
    conv0 = Activation('relu')(conv0)
    
    shortcut = conv0
    for i in range(5):
        conv1 = Conv1D(num_filters, kernel_size=ks, padding='same', kernel_regularizer=l2(reg_param))(shortcut)
        conv1 = BatchNormalization()(conv1)
        conv1 = Activation('relu')(conv1)
        conv2 = Conv1D(num_filters, kernel_size=ks, padding='same',kernel_regularizer=l2(reg_param))(conv1)
        conv2 = BatchNormalization()(conv2)
        conv2 = Activation('relu')(conv2)
        shortcut = Add()([shortcut, conv2])

    flat1 = Flatten()(shortcut)
    dense1 = Dense(d,kernel_regularizer=l2(reg_param))(flat1)
    dense1 = BatchNormalization()(dense1)
    dense1 = Activation('relu')(dense1)
    
    dense2 = Dense(d, kernel_regularizer=l2(reg_param))(dense1)
    dense2 = BatchNormalization()(dense2)
    dense2 = Activation('relu')(dense2)
    
    out = Dense(num_outputs, activation='sigmoid', kernel_regularizer=l2(reg_param))(dense1)
    model = Model(inputs=inp, outputs=out)
    return(model)

def train_model(num,Round,diff):
    net_name=str(diff)
    x_round=Round
    data_x,data_y=sm.make_train_data(num,x_round,diff)
    data_x=data_x.astype(np.uint8)#转换格式
    data_y=data_y.astype(np.uint8)

    x_val,y_val=sm.make_train_data(int(num/100),x_round,diff)
    x_val=x_val.astype(np.uint8)#转换格式
    y_val=y_val.astype(np.uint8)

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
train_model(10**7,10,(0x0,0x100000))

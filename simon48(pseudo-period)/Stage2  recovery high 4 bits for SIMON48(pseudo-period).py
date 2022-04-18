# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 09:13:57 2021

@author: deeplearning

The code is related to 'Practical Attacks of Round-Reduced SIMON Based on Deep Learning', which is submitted to The Computer Journal.
The code is also related to 'Improve Neural Distinguisher for Cryptanalysis' (https://eprint.iacr.org/2021/1017).
If you want to use the code, please refer to 'Practical Attacks of Round-Reduced SIMON Based on Deep Learning' or 'Improve Neural Distinguisher for Cryptanalysis'.
"""

import numpy as np
from os import urandom
import simon48 as sm
from random import randint
from time import time


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"#

from keras.models import model_from_json
import tensorflow as tf
config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth=True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)#


json_file = open('(0, 1048576)model10.json','r')
json_model = json_file.read()
net = model_from_json(json_model)
net.load_weights('(0, 1048576)weight10.h5')


def sequence(x):
    buffer=np.arange(pow(2,8),dtype=np.uint32)
    return buffer^x
    

def make_train_data(n, nr, diff):#
    keys = np.frombuffer(urandom(16),dtype=np.uint32).reshape(4,-1)
    keys = keys % pow(2,24)
    plain0l = np.frombuffer(urandom(4*n),dtype=np.uint32)%pow(2,24)
    plain0r = np.frombuffer(urandom(4*n),dtype=np.uint32)%pow(2,24)
    plain1l = plain0l ^ diff[0]
    plain1r = plain0r ^ diff[1]

    ks = sm.expand_key(keys, nr)
    ctdata0l, ctdata0r = sm.encrypt((plain0l, plain0r), ks)
    ctdata1l, ctdata1r = sm.encrypt((plain1l, plain1r), ks)

    return ks[-1],ctdata0l, ctdata0r,ctdata1l, ctdata1r

def hw(v):
    res = np.zeros(v.shape,dtype=np.uint8)
    for i in range(20):
        res = res + ((v >> i) & 1)
    return res


def generate_data(wrong_bit,num):
    start=time()
    Round=12
    diff=(0x400000,0x100001)
    wrong = np.array(range(2**20), dtype=np.uint32)
    wrong = wrong[hw(wrong) == wrong_bit]
    
    num_cipertest_pairs=3000
    
    subkeys = np.zeros(num,dtype=np.uint32)
    sk,C0l,C0r,C1l,C1r=make_train_data(num_cipertest_pairs, Round+1, diff)
    subkeys[0]=sk[0]
    for i in range(1,num):
        sk,c0l,c0r,c1l,c1r=make_train_data(num_cipertest_pairs, Round+1, diff)
        subkeys[i]=sk[0]
        C0l=np.concatenate([C0l,c0l], axis=0)
        C0r=np.concatenate([C0r,c0r], axis=0)
        C1l=np.concatenate([C1l,c1l], axis=0)
        C1r=np.concatenate([C1r,c1r], axis=0)
    
    w=np.array([wrong[randint(0,len(wrong)-1)] for i in range(num)],dtype=np.uint32)
    
    sk20=subkeys&0x0fffff
    wrong_sk20=sk20^w
    
    score=[]
    for delta in range(2**4):
        start1=time()
        sk_Candidate=(delta<<20)^wrong_sk20
        sk_Candidate = np.repeat(sk_Candidate, num_cipertest_pairs)
        p0l,p0r=sm.dec_one_round((C0l, C0r), sk_Candidate)
        p1l,p1r=sm.dec_one_round((C1l, C1r), sk_Candidate)
        X = sm.convert_to_binary([p0l,p0r,p1l,p1r])
        Z = net.predict(X)
        
        Z = Z/(1-Z)
        Z = np.log2(Z)
        
        Z = Z.reshape(num,-1)
        
        score.append(np.sum(Z, axis=1))
        end1=time()
        print(delta,end1-start1)
    score=np.array(score,dtype=np.float64)
    score=np.transpose(score)
    flag=np.amax(score, axis=1)
    guess_4bit=np.array([np.where(score[i]==flag[i])[0][0] for i in range(num)],dtype=np.uint32)
    guess_key=wrong_sk20^(guess_4bit<<20)
    
    guess_key=guess_key^subkeys
    
    np.save('Result_wrongbit_' + str(wrong_bit) + '.npy', guess_key) 
    end=time()
    print('Wrong Bits:',wrong_bit)
    print('Number of Test:',num)
    print('Cipertest pairs in one test:',num_cipertest_pairs)
    print('Computing Time',end-start)
    
    
num=500# Test 500 times
for wrong_bit in range(6):     
    generate_data(wrong_bit,num)

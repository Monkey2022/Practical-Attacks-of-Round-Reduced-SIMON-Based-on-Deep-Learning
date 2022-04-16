# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 09:04:50 2020

@author: 18744
"""
#Simon48/96
import numpy as np
from os import urandom

def block_size():
    return(48)

def WORD_SIZE():
    return(24)

def key_words():
    return(4)

def all_round():
    return(36)
 
def DIFF():#数据差分
    return(0x0,0x100000)

#Z0 = [1,1,1,1,1,0,1,0,0,0,1,0,0,1,0,1,0,1,1,0,0,0,0,1,1,1,0,0,1,1,0]
Z1=[1,0,0,0,1,1,1,0,1,1,1,1,1,0,0,1,0,0,1,1,0,0,0,0,1,0,1,1,0,1,0]


# bit位循环左移
def left_round(value, shiftBits):
	t1 = (value >> (WORD_SIZE() - shiftBits)) ^ (value << shiftBits)
	t2 = ((2 ** WORD_SIZE()) - 1)
	return t1 & t2
# bit位循环右移
def right_round(value, shiftBits):
	t1 = (value << (WORD_SIZE() - shiftBits)) ^ (value >> shiftBits)
	t2 = ((2 ** WORD_SIZE()) - 1)
	return t1 & t2

def enc_one_round(p,k):#加密一轮
    x,y=p[0],p[1]
    tmp=x
    x=y^((left_round(x,1))&(left_round(x,8)))^(left_round(x,2))^k
    y=tmp
    return (x,y)

def dec_one_round(c, k):#解密一轮
    x,y=c[0],c[1]
    tmp=y
    y=x^k^(left_round(y,2))^((left_round(y,1))&(left_round(y,8)))
    x=tmp
    return (x,y)

def expand_key(k,t):#simon的密钥扩展
    subkey= [0 for i in range(t)]
    c=pow(2,WORD_SIZE())-4
    m=key_words()
    for i in range(0,m):
        subkey[i]=k[m-1-i]
    for i in range(m,t):
        tmp=right_round(subkey[i-1], 3)
        if (m==4):
            tmp=tmp^subkey[i-3]
        tmp=tmp^(right_round(tmp,1))
        subkey[i]=c^Z1[(i-m)%31]^tmp^subkey[i-m]
    return(subkey)
    
def encrypt(p,ks):
    x,y= p[0],p[1]
    for k in ks:
        x,y=enc_one_round((x,y), k)
    return(x,y)

def decrypt(c, ks):
    x, y = c[0], c[1]
    
    for k in reversed(ks):
        x, y = dec_one_round((x,y), k)
    return(x,y)

def check_testvector():#用于验证算法的正确性
    key= (0x1a1918, 0x121110, 0x0a0908, 0x020100)
    pt = (0x726963,0x20646e)
    ks = expand_key(key,all_round())
    ct = encrypt(pt, ks)
    p  = decrypt(ct, ks)
    if ((ct == (0x6e06a5, 0xacf156))and(p == (0x726963,0x20646e))):
        print("Testvector verified.")
        return(1)
    else:
        print("Testvector not verified.")
        return(0)

       
def convert_to_binary(arr):#将十六进制的数转为二进制数，即将[0xc69b,0xe9bb,0xe9bb,0x7e01]转为[1,1,0,0,....]型的16*4的数组
    X = np.zeros((4 * WORD_SIZE(),len(arr[0])),dtype=np.uint8)
    for i in range(4 * WORD_SIZE()):
        index = i // WORD_SIZE()
        offset = WORD_SIZE() - (i % WORD_SIZE()) - 1
        X[i] = (arr[index] >> offset) & 1
    X = X.transpose()
    return(X)
    #具体方法为：将长度为n的数组[[0xc69b,0xe9bb,0xe9bb,0x7e01]*n]转化为[[1,1,1,0,0,1,...,1,1]*n]，其中n为明文组个数

def make_train_data(n,nr):
    diff=DIFF()
    Y = np.frombuffer(urandom(n), dtype=np.uint8)
    Y = Y & 1
    keys = np.frombuffer(urandom(16*n),dtype=np.uint32).reshape(4,-1)%(2**24)
    plain0l = np.frombuffer(urandom(4*n),dtype=np.uint32)%(2**24)
    plain0r = np.frombuffer(urandom(4*n),dtype=np.uint32)%(2**24)
    plain1l = plain0l ^ diff[0]
    plain1r = plain0r ^ diff[1]
    num_rand_samples = np.sum(Y==0)
    plain1l[Y==0] = np.frombuffer(urandom(4*num_rand_samples),dtype=np.uint32)%(2**24)
    plain1r[Y==0] = np.frombuffer(urandom(4*num_rand_samples),dtype=np.uint32)%(2**24)
    ks = expand_key(keys, nr)
    ctdata0l, ctdata0r = encrypt((plain0l, plain0r), ks)
    ctdata1l, ctdata1r = encrypt((plain1l, plain1r), ks)
    X = convert_to_binary([ctdata0l, ctdata0r, ctdata1l, ctdata1r])
    return X,Y

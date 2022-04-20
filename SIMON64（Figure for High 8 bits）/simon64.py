# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 09:13:57 2021

@author: deeplearning

The code is related to 'Practical Attacks of Round-Reduced SIMON Based on Deep Learning', which is submitted to The Computer Journal.
The code is also related to 'Improve Neural Distinguisher for Cryptanalysis' (https://eprint.iacr.org/2021/1017).
If you want to use the code, please refer to 'Practical Attacks of Round-Reduced SIMON Based on Deep Learning' or 'Improve Neural Distinguisher for Cryptanalysis'.
"""
#Simon64/128
import numpy as np
from os import urandom

def block_size():
    return(64)

def WORD_SIZE():
    return(32)

def key_words():
    return(4)

def const_seq():
    return(8)

def all_round():
    return(44)

def DIFF():
    return (0,16)

Z0 = [1,1,0,1,1,0,1,1,1,0,1,0,1,1,0,0,0,1,1,0,0,1,0,1,1,1,1,0,0,0,0,0,0,1,0,0,1,0,0,0,1,0,1,0,0,1,1,1,0,0,1,1,0,1,0,0,0,0,1,1,1,1]

#
def left_round(value, shiftBits):
	t1 = (value >> (WORD_SIZE() - shiftBits)) ^ (value << shiftBits)
	t2 = ((2 ** WORD_SIZE()) - 1)
	return t1 & t2
#
def right_round(value, shiftBits):
	t1 = (value << (WORD_SIZE() - shiftBits)) ^ (value >> shiftBits)
	t2 = ((2 ** WORD_SIZE()) - 1)
	return t1 & t2

def enc_one_round(p,k):#
    x,y=p[0],p[1]
    tmp=x
    x=y^((left_round(x,1))&(left_round(x,8)))^(left_round(x,2))^k
    y=tmp
    return (x,y)

def dec_one_round(c, k):#
    x,y=c[0],c[1]
    tmp=y
    y=x^k^(left_round(y,2))^((left_round(y,1))&(left_round(y,8)))
    x=tmp
    return (x,y)

def expand_key(k,t):#
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
        subkey[i]=c^Z0[(i-m)%62]^tmp^subkey[i-m]
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

def check_testvector():#
    key= (0x1b1a1918, 0x13121110, 0x0b0a0908, 0x03020100)
    pt = (0x656b696c,0x20646e75)
    ks = expand_key(key,all_round())
    ct = encrypt(pt, ks)
    p  = decrypt(ct, ks)
    if ((ct == (0x44c8fc20,0xb9dfa07a))and(p == (0x656b696c,0x20646e75))):
        #print("Testvector verified.")
        return(1)
    else:
        #print("Testvector not verified.")
        return(0)
      
def convert_to_binary(arr):
    X = np.zeros((4 * WORD_SIZE(),len(arr[0])),dtype=np.uint8)
    for i in range(4 * WORD_SIZE()):
        index = i // WORD_SIZE()
        offset = WORD_SIZE() - (i % WORD_SIZE()) - 1
        X[i] = (arr[index] >> offset) & 1
    X = X.transpose()
    return(X)


def make_train_data(n,nr):
    diff=DIFF()
    Y = np.frombuffer(urandom(n), dtype=np.uint8)
    Y = Y & 1
    keys = np.frombuffer(urandom(16*n),dtype=np.uint32).reshape(4,-1)
    plain0l = np.frombuffer(urandom(4*n),dtype=np.uint32)
    plain0r = np.frombuffer(urandom(4*n),dtype=np.uint32)
    plain1l = plain0l ^ diff[0]
    plain1r = plain0r ^ diff[1]
    num_rand_samples = np.sum(Y==0)
    plain1l[Y==0] = np.frombuffer(urandom(4*num_rand_samples),dtype=np.uint32)
    plain1r[Y==0] = np.frombuffer(urandom(4*num_rand_samples),dtype=np.uint32)
    ks = expand_key(keys, nr)
    ctdata0l, ctdata0r = encrypt((plain0l, plain0r), ks)
    ctdata1l, ctdata1r = encrypt((plain1l, plain1r), ks)
    X = convert_to_binary([ctdata0l, ctdata0r, ctdata1l, ctdata1r])
    return X,Y

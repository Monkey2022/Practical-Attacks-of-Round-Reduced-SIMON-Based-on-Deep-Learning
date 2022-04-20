# -*- coding: utf-8 -*-
"""
Created on Sun Aug 16 16:20:56 2020

@author: deeplearning
"""

import simon64 as sm
import numpy as np

from keras.models import model_from_json
from os import urandom
from time import time
from math import sqrt,log2

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"#
import tensorflow as tf
config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth=True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)#


Wrong_key_size=2**24

WORD_SIZE = sm.WORD_SIZE()
diff=sm.DIFF()

#
#load distinguishers
json_file10 = open(str(diff)+'model10.json', 'r')
json_model10 = json_file10.read()
net10 = model_from_json(json_model10)
net10.load_weights(str(diff)+'weight10.h5')

json_file11 = open(str(diff)+'model11.json', 'r')
json_model11 = json_file11.read()
net11 = model_from_json(json_model11)
net11.load_weights(str(diff)+'weight11.h5')

m10 = np.load('data_wrong_key_11r_mean.npy')#
s10 = np.load('data_wrong_key_11r_std.npy')#
s10 = 1.0/s10#

m11= np.load('data_wrong_key_12r_mean.npy')
s11 = np.load('data_wrong_key_12r_std.npy')
s11 = 1.0/s11

def convert_to_binary(arr):#
    X = np.zeros((4 * WORD_SIZE(),len(arr[0])),dtype=np.uint8)
    for i in range(4 * WORD_SIZE()):
        index = i // WORD_SIZE()
        offset = WORD_SIZE() - (i % WORD_SIZE()) - 1
        X[i] = (arr[index] >> offset) & 1
    X = X.transpose()
    return(X)

    
low_weight = np.loadtxt('buffer1.txt',dtype=np.uint32)#hw(t)<=2

def make_structure(pt0, pt1,neutral_bit):
    
    diff=sm.DIFF()
    neutral_bits=neutral_bit
    p0 = np.copy(pt0) 
    p1 = np.copy(pt1)
    p0 = p0.reshape(-1,1)  #
    p1 = p1.reshape(-1,1)
    for i in range(0,2**(len(neutral_bits))):
        bin_i=np.zeros(len(neutral_bits),dtype=int)
        buffer=bin(i)
        buffer=buffer[2:]
        buffer= list(map(int,buffer))
        length=len(buffer)
        bin_i[(len(neutral_bits)-length):len(neutral_bits)]=buffer[0:length]
        #print(bin_i)
        for j in range(0,len(neutral_bits)):
            if(neutral_bits[j]<32):
                if(bin_i[j]==1):
                    d=1<<(WORD_SIZE-1-neutral_bits[j])
                    p0[:,-1]=np.copy(p0[:,-1]|d)
                else:
                    d=1<<(WORD_SIZE-1-neutral_bits[j])
                    d0=0xffffff^d
                    p0[:,-1]=np.copy(p0[:,-1]&d0)
            else:
                if(bin_i[j]==1):
                    d=1<<(neutral_bits[j]-(WORD_SIZE-1))
                    p1[:,-1]=np.copy(p1[:,-1]|d)
                else:
                    d=1<<(neutral_bits[j]-(WORD_SIZE-1))
                    d0=0xffffff^d
                    p1[:,-1]=np.copy(p1[:,-1]&d0)
        p0 = np.concatenate((p0,p0[:,0].reshape(-1,1)), axis=1)    #
        p1 = np.concatenate((p1,p1[:,0].reshape(-1,1)), axis=1)
    p0=np.copy(np.delete(p0,-1,axis=1))#
    p1=np.copy(np.delete(p1,-1,axis=1))
    p0b = p0^diff[0]
    p1b = p1^diff[1]
    return(p0, p1, p0b, p1b)


def gen_key(nr):#
    key = np.frombuffer(urandom(16), dtype=np.uint32)
    ks = sm.expand_key(key, nr)
    return(ks)

def gen_plain(n):#
    pt0 = np.frombuffer(urandom(4*n), dtype=np.uint32)
    pt1 = np.frombuffer(urandom(4*n), dtype=np.uint32)
    return(pt0, pt1)

def gen_challenge(n,nr,neutral_bit):
    pt0, pt1 = gen_plain(n)
    pt0a, pt1a, pt0b, pt1b = make_structure(pt0,pt1,neutral_bit)
    pt0a, pt1a = sm.dec_one_round((pt0a, pt1a), 0)
    pt0b, pt1b = sm.dec_one_round((pt0b, pt1b), 0)
    #
    key = gen_key(nr)
    ct0a, ct1a = sm.encrypt((pt0a, pt1a), key)
    ct0b, ct1b = sm.encrypt((pt0b, pt1b), key)
    return([ct0a, ct1a, ct0b, ct1b], key)


def verifier_search(cts, best_guess,neutral_bit,net):#net=net_help
    #print(best_guess)
    use_n=2**len(neutral_bit)
    ck1 = best_guess[0] ^ low_weight
    ck2 = best_guess[1] ^ low_weight
    n = len(ck1)
    ck1 = np.repeat(ck1, n)#
    keys1 = np.copy(ck1) 
    ck2 = np.tile(ck2, n) #
    keys2 = np.copy(ck2)
    ck1 = np.repeat(ck1, use_n)
    ck2 = np.repeat(ck2, use_n)
    ct0a = np.tile(cts[0][0:use_n], n*n)     
    ct1a = np.tile(cts[1][0:use_n], n*n)
    ct0b = np.tile(cts[2][0:use_n], n*n)
    ct1b = np.tile(cts[3][0:use_n], n*n)
    pt0a, pt1a = sm.dec_one_round((ct0a, ct1a), ck1)
    pt0b, pt1b = sm.dec_one_round((ct0b, ct1b), ck1)
    pt0a, pt1a = sm.dec_one_round((pt0a, pt1a), ck2)
    pt0b, pt1b = sm.dec_one_round((pt0b, pt1b), ck2)
    X = sm.convert_to_binary([pt0a, pt1a, pt0b, pt1b])
    Z = net.predict(X, batch_size=10000)
    Z = Z / (1 - Z)
    Z = np.log2(Z)
    Z = Z.reshape(-1, use_n)#
    v = np.mean(Z, axis=1) * len(cts[0])
    m = np.argmax(v)
    val = v[m]
    key1 = keys1[m]
    key2 = keys2[m]
    return(key1, key2, val)

def bayesian_rank_kr(cand, emp_mean,m,s):
    n = len(cand)
    tmp_br = np.arange(Wrong_key_size, dtype=np.uint32)
    tmp_br = np.repeat(tmp_br, n).reshape(-1, n)
    
    tmp = tmp_br ^ cand
    v = (emp_mean - m[tmp%Wrong_key_size]) * s[tmp%Wrong_key_size]
    v = v.reshape(-1, n)
    scores = np.linalg.norm(v, axis=1)
    return(scores)

def bayesian_key_recovery(cts, net, m, s, num_iter):
# 
    num_cand = 8#
    n = len(cts[0])
    keys = np.random.choice(Wrong_key_size, num_cand, replace=False)
    r = np.random.randint(0, pow(2,4), num_cand, dtype=np.uint32) #
    r = r << 20
    keys = keys ^ r
    scores = 0

    ct0a, ct1a, ct0b, ct1b = np.tile(cts[0], num_cand), np.tile(cts[1], num_cand), np.tile(cts[2], num_cand), np.tile(cts[3], num_cand)
    
    scores = np.zeros(Wrong_key_size)

    all_keys = np.zeros(num_cand * num_iter, dtype=np.uint32)#
    all_v = np.zeros(num_cand * num_iter)#
    for i in range(num_iter):
        k = np.repeat(keys, n)
        c0a, c1a = sm.dec_one_round((ct0a, ct1a), k)#
        c0b, c1b = sm.dec_one_round((ct0b, ct1b), k)#
        X = sm.convert_to_binary([c0a, c1a, c0b, c1b])
        Z = net.predict(X, batch_size=10000)
        Z = Z.reshape(num_cand, -1)#
        means = np.mean(Z, axis=1)
        Z = Z/(1-Z)
        Z = np.log2(Z)
        v =np.sum(Z, axis=1)
        all_v[i * num_cand:(i+1)*num_cand] = v
        all_keys[i * num_cand:(i+1)*num_cand] = np.copy(keys)
        
        scores = bayesian_rank_kr(keys, means, m=m, s=s)
        
        tmp = np.argpartition(scores, num_cand)  #
        
        keys = tmp[0:num_cand]
        r = np.random.randint(0, pow(2,8), num_cand, dtype=np.uint32) #
        r = r << 24
        keys = keys ^ r
    return(all_keys, scores, all_v)

    
def test_bayes(cts, neutral_bit,it, cutoff1, cutoff2, net, net_help, m_main, m_help, s_main, s_help):
    n = len(cts[0])
    verify_breadth=len(cts[0][0])
    alpha = sqrt(n)#
    best_val = -100.0
    best_key = (0, 0)
    best_pod = 0 
    keys = np.random.choice(2**WORD_SIZE, 16, replace=False)
    eps = 0.001
    local_best = np.full(n, -10)#
    num_visits = np.full(n, eps)#

    for j in range(it):
        #print(j)
        priority = local_best + alpha * np.sqrt(log2(j+1) / num_visits)
        i = np.argmax(priority)#
        num_visits[i] = num_visits[i] + 1
        if (best_val > cutoff2):
            improvement = (verify_breadth > 0)
            while improvement:
                k1, k2, val = verifier_search([cts[0][best_pod], cts[1][best_pod], cts[2][best_pod], cts[3][best_pod]], best_key, neutral_bit,net=net_help)
                improvement = (val > best_val)
                if (improvement):
                    best_key = (k1, k2)
                    best_val = val
            return(best_key, j)
        keys, scores, v = bayesian_key_recovery([cts[0][i], cts[1][i], cts[2][i], cts[3][i]], net=net, m=m_main, s=s_main,num_iter=5)
        vtmp = np.max(v)
        if (vtmp > local_best[i]): 
            local_best[i] = vtmp
        #print('*********vtmp**********\n',vtmp,'\n\n\n\n\n')
        if (vtmp > cutoff1):
            l2 = [i for i in range(len(keys)) if v[i] > cutoff1]
            for i2 in l2:
                c0a, c1a = sm.dec_one_round((cts[0][i], cts[1][i]), keys[i2])#
                c0b, c1b = sm.dec_one_round((cts[2][i], cts[3][i]), keys[i2])         
                keys2, scores2, v2 = bayesian_key_recovery([c0a, c1a, c0b, c1b], net=net_help, m=m_help, s=s_help,num_iter=5 )
                vtmp2 = np.max(v2)
                #print('*********vtmp2**********',vtmp2,'\n')
                if (vtmp2 > best_val):
                    best_val = vtmp2
                    best_key = (keys[i2], keys2[np.argmax(v2)])
                    best_pod=i
    improvement = (verify_breadth > 0)
    while improvement:
        k1, k2, val = verifier_search([cts[0][best_pod], cts[1][best_pod], cts[2][best_pod], cts[3][best_pod]], best_key,neutral_bit, net=net_help);
        improvement = (val > best_val);
        if (improvement):
            best_key = (k1, k2) 
            best_val = val

    return(best_key, it)

def enc_time():
    n=pow(2,10)
    pt0 = np.frombuffer(urandom(4*n), dtype=np.uint32)
    pt1 = np.frombuffer(urandom(4*n), dtype=np.uint32)
    pt0=(pt0.reshape(-1,1)).flatten()
    pt1=(pt1.reshape(-1,1)).flatten()

    key = np.frombuffer(urandom(16), dtype=np.uint32).reshape(4,-1)
    ks = sm.expand_key(key, 13)

    t0=time()
    for i in range(n):
        ct0a, ct1a = sm.encrypt((pt0, pt1), ks)
    t1=time()
    return ((t1 - t0)/n)


def test(n,neutral_bit, nr=13, num_structures=pow(2,9), it=500, cutoff1=10.0, cutoff2=10.0, keyschedule='real', net=net11, net_help=net10, m_main=m11, s_main=s11,  m_help=m10, s_help=s10):
    if (sm.check_testvector()==0):
        print("Error about simon")
        return(0)
    arr1 = np.zeros(n, dtype=np.uint32) #
    data = 0
    for i in range(n): 
        print(i)
        t1=time()
        ct, key = gen_challenge(num_structures, nr,neutral_bit)
        print('gen_challenge done')
        guess, num_used = test_bayes(ct, neutral_bit,it=it, cutoff1=cutoff1, cutoff2=cutoff2, net=net, net_help=net_help, m_main=m_main, s_main=s_main, m_help=m_help, s_help=s_help)
        num_used = min(num_structures, num_used)
        data = data + 2 * (2 ** len(neutral_bit)) * num_used
        arr1[i] = guess[0] ^ key[nr-1]  
        t2=time()
        print(i,hex(arr1[i]))
        print(t2-t1)
    print(arr1)
    print("Data blocks used (average, log2): ", log2(data) - log2(n))
    return(arr1)

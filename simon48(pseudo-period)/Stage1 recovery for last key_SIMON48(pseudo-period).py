# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 09:13:57 2021

@author: deeplearning

The code is related to 'Practical Attacks of Round-Reduced SIMON Based on Deep Learning', which is submitted to The Computer Journal.
The code is also related to 'Improve Neural Distinguisher for Cryptanalysis' (https://eprint.iacr.org/2021/1017).
If you want to use the code, please refer to 'Practical Attacks of Round-Reduced SIMON Based on Deep Learning' or 'Improve Neural Distinguisher for Cryptanalysis'.
"""
#Before running this Python file, please run 'wrong_key_rank.py' to 
#generate 'data_wrong_key_10r_mean', 'data_wrong_key_10r_std.npy', 
#'data_wrong_key_11r_mean.npy' and 'data_wrong_key_11r_std.npy'.

import simon48 as sm
import numpy as np

from keras.models import model_from_json
from os import urandom
from time import time,localtime
from math import sqrt,log2

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"#SELECT gpu

import tensorflow as tf
config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth=True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)

WORD_SIZE = sm.WORD_SIZE()
diff=sm.DIFF()

Wrong_key_size=2**20

#可修改成为
#load distinguishers
json_file9 = open(str(diff)+'model9.json', 'r')
json_model9 = json_file9.read()
net9 = model_from_json(json_model9)
net9.load_weights(str(diff)+'weight9.h5')

json_file10 = open(str(diff)+'model10.json', 'r')
json_model10 = json_file10.read()
net10 = model_from_json(json_model10)
net10.load_weights(str(diff)+'weight10.h5')

m10 = np.load('data_wrong_key_11r_mean.npy')
s10 = np.load('data_wrong_key_11r_std.npy')
s10 = 1.0/s10#

m9= np.load('data_wrong_key_10r_mean.npy')
s9 = np.load('data_wrong_key_10r_std.npy')
s9 = 1.0/s9

m10=m10[:Wrong_key_size]
s10=s10[:Wrong_key_size]
m9=m10[:Wrong_key_size]
s9=s10[:Wrong_key_size]

def convert_to_binary(arr):
    X = np.zeros((4 * WORD_SIZE(),len(arr[0])),dtype=np.uint8)
    for i in range(4 * WORD_SIZE()):
        index = i // WORD_SIZE()
        offset = WORD_SIZE() - (i % WORD_SIZE()) - 1
        X[i] = (arr[index] >> offset) & 1
    X = X.transpose()
    return(X)

    
def hw(v):
    res = np.zeros(v.shape,dtype=np.uint8)
    for i in range(WORD_SIZE):
        res = res + ((v >> i) & 1)
    return res

low_weight = np.array(range(2**WORD_SIZE), dtype=np.uint32)
low_weight = low_weight[hw(low_weight) <= 2]


def make_structure(pt0, pt1,neutral_bit):

    Diff=(0x400000,0x100001)
    neutral_bits=neutral_bit
    p0 = np.copy(pt0) 
    p1 = np.copy(pt1)
    p0 = p0.reshape(-1,1)  
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
            if(neutral_bits[j]<24):
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
        p0 = np.concatenate((p0,p0[:,0].reshape(-1,1)), axis=1)    
        p1 = np.concatenate((p1,p1[:,0].reshape(-1,1)), axis=1)
    p0=np.copy(np.delete(p0,-1,axis=1))
    p1=np.copy(np.delete(p1,-1,axis=1))
    p0b = p0^Diff[0]
    p1b = p1^Diff[1]
    return(p0, p1, p0b, p1b)

    
#generate a key, return expanded key
def gen_key(nr):
    key = np.frombuffer(urandom(16),dtype=np.uint32).reshape(4,-1)%(2**24)
    ks = sm.expand_key(key, nr)
    return(ks)

def gen_plain(n):
    pt0 = np.frombuffer(urandom(4*n),dtype=np.uint32)%(2**24)
    pt1 = np.frombuffer(urandom(4*n),dtype=np.uint32)%(2**24)
    return(pt0, pt1)

def gen_challenge(n,nr,neutral_bit):
    pt0, pt1 = gen_plain(n)
    pt0a, pt1a, pt0b, pt1b = make_structure(pt0,pt1,neutral_bit)
    pt0a, pt1a = sm.dec_one_round((pt0a, pt1a), 0)
    pt0b, pt1b = sm.dec_one_round((pt0b, pt1b), 0)
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
    ck1 = np.repeat(ck1, n)
    keys1 = np.copy(ck1) 
    ck2 = np.tile(ck2, n) 
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
    Z = Z.reshape(-1, use_n)
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
#
    num_cand = 64#
    n = len(cts[0])
    
    keys = np.random.choice(2**20, num_cand, replace=False) 
    r = np.random.randint(0, pow(2,4), num_cand, dtype=np.uint32) #
    r = r << 20
    keys = keys ^ r
    scores = 0

    ct0a, ct1a, ct0b, ct1b = np.tile(cts[0], num_cand), np.tile(cts[1], num_cand), np.tile(cts[2], num_cand), np.tile(cts[3], num_cand)
    scores = np.zeros(2**(WORD_SIZE-2))

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
        r = np.random.randint(0, pow(2,4), num_cand, dtype=np.uint32) #
        r = r << 20
        keys = keys ^ r
    return(all_keys, scores, all_v)

    
def test_bayes(cts, neutral_bit,it, cutoff1, cutoff2, net, net_help, m_main, m_help, s_main, s_help):
    n = len(cts[0])
    verify_breadth=len(cts[0][0])
    alpha = sqrt(n)#
    best_val = -100.0
    best_key = (0, 0)
    best_pod = 0 
    keys = np.random.choice(2**WORD_SIZE, 32, replace=False)
    eps = 0.001
    local_best = np.full(n, -10)#
    num_visits = np.full(n, eps)#

    for j in range(it):
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
   
        if (vtmp > cutoff1):
            l2 = [i for i in range(len(keys)) if v[i] > cutoff1]
            for i2 in l2:
                c0a, c1a = sm.dec_one_round((cts[0][i], cts[1][i]), keys[i2])#解密1轮
                c0b, c1b = sm.dec_one_round((cts[2][i], cts[3][i]), keys[i2])         
                keys2, scores2, v2 = bayesian_key_recovery([c0a, c1a, c0b, c1b], net=net_help, m=m_help, s=s_help,num_iter=5 )
                vtmp2 = np.max(v2);
                if (vtmp2 > best_val):
                    best_val = vtmp2
                    best_key = (keys[i2], keys2[np.argmax(v2)])
                    best_pod=i
    improvement = (verify_breadth > 0)
    while improvement:
        k1, k2, val = verifier_search([cts[0][best_pod], cts[1][best_pod], cts[2][best_pod], cts[3][best_pod]], best_key, neutral_bit, net=net_help)
        improvement = (val > best_val)
        if (improvement):
            best_key = (k1, k2) 
            best_val = val

    return(best_key, it)

def enc_time():
    n=pow(2,10)
       
    pt0 = np.frombuffer(urandom(4*n), dtype=np.uint32)%(2**24)
    pt1 = np.frombuffer(urandom(4*n), dtype=np.uint32)%(2**24)
    pt0=(pt0.reshape(-1,1)).flatten()
    pt1=(pt1.reshape(-1,1)).flatten()

    key = np.frombuffer(urandom(16),dtype=np.uint32).reshape(4,-1)%(2**24)
    ks = sm.expand_key(key, 14)

    t0=time()
    for i in range(n):
        ct0a, ct1a = sm.encrypt((pt0, pt1), ks)
    t1=time()
    return ((t1 - t0)/n)

def test(n, nr=14, num_structures=pow(2,15), it=100, cutoff1=10, cutoff2=50, keyschedule='real', net=net10, net_help=net9, m_main=m10, s_main=s10,  m_help=m9, s_help=s9):
    #print("Checking SIMON32/64 enc/dec")
    if (sm.check_testvector()==0):
        print("Error about simon")
        return(0)
    arr1 = np.zeros(n, dtype=np.uint32) #
    data = 0
    neutral_bit=[44,47,21,39,3,28]
    print("neutral_bit:",neutral_bit)
    
    t0 = time()
    for i in range(n):       
        print("Test:", i,end=' ')
        t2=time()
        ct, key = gen_challenge(num_structures, nr,neutral_bit)
        #print("gen challenge done")
        guess, num_used = test_bayes(ct, neutral_bit,it=it, cutoff1=cutoff1, cutoff2=cutoff2, net=net, net_help=net_help, m_main=m_main, s_main=s_main, m_help=m_help, s_help=s_help)
        #print("test bayes done")
        num_used = min(num_structures, num_used)
        data = data + 2 * (2 ** len(neutral_bit)) * num_used
        arr1[i] = guess[0] ^ key[nr-1]     
        print("Difference between real key and key guess: ", hex(arr1[i]&0x0fffff))
        t3=time()
        print(t3-t2)
        #print('\n')
    t1 = time()
    print(t1-t0)
    #print("Done.")
    
    d1 = [hex(x) for x in arr1]   
    low_weight3 = np.array(range(2**WORD_SIZE), dtype=np.uint32)
    low_weight3 = low_weight3[hw(low_weight3) <= 5]
    a=[i for i in arr1 if i in low_weight3]
    print("neutral_bit:",neutral_bit)
    print("it:",it,"  cutoff1:",cutoff1,"  cutoff2:",cutoff2,sep=' ')
    print("Differences between guessed and last key:", d1)
    print("Wall time per attack (average in seconds):", (t1 - t0)/n)
    print("Wall  enc time per attack (round 11 of simon48, log2):", log2(((t1 - t0)/n)/enc_time()))
    print("Data blocks used (average, log2): ", log2(data) - log2(n))
    print("Probibality for last key: ",((len(a)/n)))
    
    return(arr1,neutral_bit)


n=30
print(localtime())
arr1,neutral_bit= test(n)
low_weight3 = np.array(range(2**WORD_SIZE), dtype=np.uint32)
low_weight3 = low_weight3[hw(low_weight3) <= 5]
a=[i for i in arr1 if i in low_weight3]
print("Probibality for last key: ",((len(a)/n)))
print(localtime())

# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 09:13:57 2021

@author: deeplearning

The code is related to 'Practical Attacks of Round-Reduced SIMON Based on Deep Learning', which is submitted to The Computer Journal.
The code is also related to 'Improve Neural Distinguisher for Cryptanalysis' (https://eprint.iacr.org/2021/1017).
If you want to use the code, please refer to 'Practical Attacks of Round-Reduced SIMON Based on Deep Learning' or 'Improve Neural Distinguisher for Cryptanalysis'.
"""
import simon64 as sm
import numpy as np
import time


from os import urandom
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from keras.models import model_from_json

WORD_SIZE = sm.WORD_SIZE()

def key_single(ct0a, ct1a, ct0b, ct1b, keys, net):
    pt0a, pt1a = sm.dec_one_round((ct0a, ct1a), keys)
    pt0b, pt1b = sm.dec_one_round((ct0b, ct1b), keys)
    X = sm.convert_to_binary([pt0a, pt1a, pt0b, pt1b])
    Z = net.predict(X, batch_size=10000)
    return(Z)

def make_testset(nr,number):
    diff=sm.DIFF()
    pt0a = np.frombuffer(urandom(4*number),dtype=np.uint32)
    pt1a = np.frombuffer(urandom(4*number), dtype=np.uint32)
    pt0b=pt0a ^ diff[0]
    pt1b=pt1a ^ diff[1]
    keys = np.frombuffer(urandom(16*number),dtype=np.uint32).reshape(4,-1)
    ks = sm.expand_key(keys, nr)
    ct0a, ct1a = sm.encrypt((pt0a, pt1a), ks)
    ct0b, ct1b = sm.encrypt((pt0b, pt1b), ks)
    return(ct0a, ct1a, ct0b, ct1b, ks[-1])
    
def wrong_key_response(net_structure, net_weight,nr, number):
    #load distinguishers
    json_file = open(net_structure + '.json', 'r')
    json_model = json_file.read()
    print(net_weight)
    net = model_from_json(json_model)
    
    net.load_weights(net_weight + '.h5')
    
    wrong_key_mean = np.zeros(2 ** 24, dtype=np.float64)
    wrong_key_std = np.zeros(2 ** 24, dtype=np.float64)

    if(sm.check_testvector()==1):
        print('the ' + str(nr) + ' round:\n')
        t1=time.time()
        for delta in range(2 ** 24):
            if(delta%(2**16)==0):#????????????
                print(delta/2**16)
                t2=time.time()
                print(t2-t1)
                t1=time.time()
            ct0a, ct1a, ct0b, ct1b, last_subkey = make_testset(nr,number)
            last_subkey=last_subkey^ delta
            wrong_response= key_single(ct0a, ct1a, ct0b, ct1b, last_subkey, net)
            wrong_key_mean[delta]= np.mean(wrong_response)#
            wrong_key_std[delta]= np.std(wrong_response)#
        #print(wrong_key_mean, wrong_key_std)
        #print('right!\n')
        np.save('data_wrong_key_' + str(nr) + 'r_mean.npy', wrong_key_mean)
        np.save('data_wrong_key_' + str(nr) + 'r_std.npy', wrong_key_std)
        print('done!\n')
    else:
        print('error')
        

#
def get_rank(a,b,num):#
    diff=sm.DIFF()
    
    for i in range(a,b+1):
        net_structure=str(diff)+'model'+str(i)
        net_weight=str(diff)+'weight'+str(i)
        wrong_key_response(net_structure, net_weight,i+1, num)
      
start = time.time()
get_rank(10,11,3000)
end = time.time()
print("Execution Time: ", end - start)

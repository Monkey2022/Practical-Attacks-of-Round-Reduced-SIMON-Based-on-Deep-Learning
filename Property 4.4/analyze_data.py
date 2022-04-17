# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 11:17:43 2020

@author: deeplearning

The code is related to 'Practical Attacks of Round-Reduced SIMON Based on Deep Learning', which is submitted to The Computer Journal.
The code is also related to 'Improve Neural Distinguisher for Cryptanalysis' (https://eprint.iacr.org/2021/1017).
If you want to use the code, please refer to 'Practical Attacks of Round-Reduced SIMON Based on Deep Learning' or 'Improve Neural Distinguisher for Cryptanalysis'.
"""
#Before running this Python file, please run 'wrong_key_rank.py' to generate 'H_GPUx_Mean.npy' or 'H_GPUx_Std.npy'
import os
import numpy as np
from math import log2
path=os.getcwd()#read path

data=os.listdir(path)#read file name

flag=1
subkey_size=24

while(flag):
    flag=0
    for i in data:
        if(i[-4:]!='.npy'):
            flag=1
            data.remove(i)

data_mean=[i for i in data if i[-8:]=='Mean.npy']
data_std=[i for i in data if i[-7:]=='Std.npy']

Mean_difference=[]
Mean_difference_max=[]
for i in data_mean:
    real_mean=np.load(i)
    buffer=[]
    for k in range(1,subkey_size+1):
        partial_mean =np.tile(real_mean[:pow(2,k)], pow(2,subkey_size-k)) #Repeat real_mean[:pow(2,k)] pow(2,24-k)times
        difference_value=real_mean-partial_mean
        difference_value=np.maximum(difference_value,-difference_value)
        buffer.append(sum(difference_value))
    buffer=[log2(i) for i in buffer if i>0 ]
    Mean_difference.append(buffer)
np.save('Mean_difference.npy',np.array(Mean_difference))

Std_difference=[]
Std_difference_max=[]
for i in data_std:
    real_std=np.load(i)
    buffer=[]
    for k in range(1,subkey_size+1):
        partial_std =np.tile(real_std[:pow(2,k)], pow(2,subkey_size-k)) #Repeat real_std[:pow(2,k)] pow(2,24-k)times
        difference_value=real_std-partial_std
        difference_value=np.maximum(difference_value,-difference_value)
        buffer.append(sum(difference_value))
    buffer=[log2(i) for i in buffer if i>0 ]
    Std_difference.append(buffer) 

np.save('Std_difference.npy',np.array(Std_difference))




 
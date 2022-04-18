# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 09:13:57 2021

@author: deeplearning

The code is related to 'Practical Attacks of Round-Reduced SIMON Based on Deep Learning', which is submitted to The Computer Journal.
The code is also related to 'Improve Neural Distinguisher for Cryptanalysis' (https://eprint.iacr.org/2021/1017).
If you want to use the code, please refer to 'Practical Attacks of Round-Reduced SIMON Based on Deep Learning' or 'Improve Neural Distinguisher for Cryptanalysis'.
"""
#Before running this Python file, please run 'Stage2  recovery high 4 bits for SIMON48(pseudo-period).py'  


import numpy as np


for i in range(6):
    
    f1=np.load('Result_wrongbit_'+str(i)+'.npy')
    l1=len(f1)
    f=np.zeros(l1,dtype=np.uint32)
    f[:l1]=np.copy(f1)
    f=f>>20
    k=0
    for j in f:
        k1=0
        j1=bin(j)[2:]
        for u in j1:
            if(u=='1'):
                k1=k1+1
        if(k1<3):
            k=k+1
    print(i,k/(l1))
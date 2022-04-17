# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 11:17:43 2020

@author: deeplearning

The code is related to 'Practical Attacks of Round-Reduced SIMON Based on Deep Learning', which is submitted to The Computer Journal.
The code is also related to 'Improve Neural Distinguisher for Cryptanalysis' (https://eprint.iacr.org/2021/1017).
If you want to use the code, please refer to 'Practical Attacks of Round-Reduced SIMON Based on Deep Learning' or 'Improve Neural Distinguisher for Cryptanalysis'.
"""
#This code is used to find optimal probability for a given number of rounds.
from z3 import *
import numpy as np

block_size=64#block size
word_size=int(block_size/2)

global f1,f2,f3,f4,f5
f1=0
f2=0
f3=0
f4=0
f5=0
for i in range(int(word_size/4)):
    f1=(f1<<4)+0x5
    f2=(f2<<4)+0x3
    f5=(f5<<4)+0xf
    if(i%2==0):
        f3=(f3<<4)+0
        f4=(f4<<4)+0
    else:
        f3=(f3<<4)+0xf
        f4=(f4<<4)+0x1

def hw(a):
    num = (a & f1) + ((a >> 1) & f1)
    num = (num & f2) + ((num >> 2) & f2)
    num = (num & f3) + ((num >> 4) & f3)
    num = ((num * f4) & ((1 << (word_size)) - 1)) >> (word_size-8)
    return num

def SIMON_SAT_diff(Round,Probability,parameter):
    global X,Y,V,D,P,W,Z,G
    X=[BitVec('x%d' % i,word_size) for i in range(Round+1)]
    Y=[BitVec('y%d' % i,word_size) for i in range(Round+1)]
    V=[BitVec('v%d' % i,word_size) for i in range(Round)]
    D=[BitVec('d%d' % i,word_size) for i in range(Round)]
    P=[Int('p%d' % i) for i in range(Round)]
    W=[BitVec('w%d' % i,word_size) for i in range(Round)]
    Z=[BitVec('z%d' % i,word_size) for i in range(Round)]
    G=[BitVec('g%d' % i,word_size) for i in range(Round)]

    s=Solver()

    s.add(Probability==Sum(P))
    for i in range(Round):
        
        s.add(V[i]==RotateLeft(X[i],parameter[0])|RotateLeft(X[i],parameter[1]))
        s.add(D[i]==RotateLeft(X[i],parameter[1])&(~RotateLeft(X[i],parameter[0]))&RotateLeft(X[i],2*parameter[0]-parameter[1]))   
        s.add(Y[i+1]==X[i])
        s.add(Z[i]&V[i]==0)
        s.add((Z[i]^RotateLeft(Z[i],parameter[0]-parameter[1]))&D[i]==0)
        s.add(X[i+1]==Y[i]^Z[i]^RotateLeft(X[i],parameter[2]))
        s.add(W[i]==hw(V[i]^D[i]))

        s.add(G[i]==X[i+1]^Y[i]^RotateLeft(X[i],parameter[2]))   
        s.add(
            If(And(X[i]==f5,hw(G[i])%2==0)==True,
                P[i]==word_size-1,
                If(And(X[i]!=f5,G[i]&(~V[i])==0,(G[i]^RotateLeft(G[i],parameter[0]-parameter[1]))&D[i]==0)==True,
                    P[i]==BV2Int(W[i]),
                    P[i]==100 )       
                )
        )  
        s.add(P[i]<100)
    return s

def find_Probability(Round):
    diff=[]
    parameter=[8,1,2]
    for Probability in range(1,block_size):
        s=SIMON_SAT_diff(Round,Probability,parameter)
        flag=s.check()
        if(flag==sat):
            print('block size=',block_size,sep='',end='\n')
            print('Round=',Round,sep='',end='\n')
            print('Probability=',Probability,sep='',end='\n')
            break
    return Probability
max_Probability=find_path(11)
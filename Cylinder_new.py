# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 17:01:35 2025

@author: Anunnya
"""
import numpy as np
import math
import matplotlib.pyplot as plt
"""HOLLOW CYLINDER"""
e0=8.854 * 10**(-12)
def getL(M,N,h,r0):
    MN=M*N
    l=np.ndarray(shape=(MN,MN))
    a=(2*np.pi*r0)/M
    b=h/N
    for m in range(MN):
        for n in range(MN):
            if(m!=n):
                temp1=(np.sin((m-n)*(np.pi/M))*2*r0)**2
                temp2=(((m-n)*h)/N)**2
                l[m][n]=(((2*np.pi*r0*h)/MN)/(temp1+temp2)**0.5)/(4*np.pi*e0)
            else:
                temp1=a*math.log((b+math.pow(math.pow(a,2)+math.pow(b,2),0.5))/(-b+math.pow(math.pow(a,2)+math.pow(b,2),0.5)))
                temp2=b*math.log((a+math.pow(math.pow(a,2)+math.pow(b,2),0.5))/(-a+math.pow(math.pow(a,2)+math.pow(b,2),0.5)))
                l[m][n]=(temp1+temp2)/(4*np.pi*e0)
    return l

def alpha(L,V):
    LI=np.linalg.inv(L)
    al=np.matmul(LI,V)
    return al

def charge(al,M,N,h,r0):
    sum=0
    sn=(2*np.pi*r0*h)/(M*N)
    for i in range(M*N):
        sum+=al[i][0]*sn
    return sum

L=getL(11,9,1,0.5)
V=np.ones((11*9,1))
al=alpha(L,V)
Q=charge(al,11,9,1,0.5)
print(Q*10**12)
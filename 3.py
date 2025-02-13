# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 15:37:01 2025

@author: anunn
"""

import numpy as np
import matplotlib.pyplot as plt

def charge(d):
    a=1
    n=2
    m=n**2
    b=a/(2*n)
    tt=np.ndarray(shape=(m,m))
    tb=np.ndarray(shape=(m,m))
    bt=np.ndarray(shape=(m,m))
    bb=np.ndarray(shape=(m,m))
    e=8.854*10**(-12) 
    coord_x=np.ndarray(shape=(n,n))                                                #x coordinate
    x=[]
    coord_y=np.ndarray(shape=(n,n))                                                #y coordinate
    y=[]
    for i in range(n):
        for j in range(n):
            coord_y[i][j]=(1/n)*j + (1/(n*2))
            y.append(coord_y[i][j])
        

    for i in range(n):
        for j in range(n):
            coord_x[i][j]= 1-(((1/n) * i) +(1/(n*2)))
            x.append(coord_x[i][j])

    x.reverse()
    distance=np.ndarray(shape=(m,m))                                               #distance matrix
    for i in range(m):
        for j in range(m):
            if i!=j:
                distance[i][j]=((x[i]- x[j])**2 + (y[i] - y[j])**2)**0.5
                
    for i in range(m):
        for j in range(m):
            if i!=j:
                tt[i][j]=(a**2)/(4*m*np.pi*e*distance[i][j])
            else:
                tt[i][j]=((a/n)*0.8814)/(np.pi * e)
            bb[i][j]=tt[i][j]
            
            
    for i in range(m):
        for j in range(m):
            if(i==j):
                temp1=(0.282*2*b)/e
                temp2=(1+((np.pi*d**2)/(4*b**2)))**0.5
                temp3=(d*np.pi**0.5)/(2*b)
                tb[i][j]=temp1*(temp2-temp3)
            else:
                temp1=b**2
                temp2=((x[i]-x[j])**2 + (y[i]-y[j])**2 +d**2)**0.5
                temp3=np.pi*e
                tb[i][j]=temp1/(temp2*temp3)
            bt[i][j]=tb[i][j]
    l=np.subtract(tt,tb)
    l=np.linalg.inv(l)
    s=0
    for i in range(m):
        for j in range(m):
            s+=l[i][j]
    C=4*s*b**2
    return C

d=0.1
a=1
A=a**2
X=[]
Y=[]
e=8.854*10**(-12)
while(d<=2):
    C=charge(d)
    print(C)
    X.append(d/(2*a))
    Y.append((C*d)/(e*A))
    d+=0.1

plt.plot(X,Y)
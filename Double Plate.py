# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 15:05:58 2025

@author: anunn
"""

import numpy as np
import matplotlib.pyplot as plt

def coordinates(a, n):
    coord_x = np.ndarray(shape=(n, n))                                                 
    coord_y = np.ndarray(shape=(n, n))
    x = []
    y = []
    
    for i in range(n):
        for j in range(n):
            coord_y[i][j] = (1/n) * j + (1/(n*2))
            y.append(coord_y[i][j])

    for i in range(n):
        for j in range(n):
            coord_x[i][j] = 1 - (((1/n) * i) + (1/(n*2)))
            x.append(coord_x[i][j])

    x.reverse()
    return x, y

def single_plate(a, n, v):
    M = n**2
    b = a / (2 * n)
    e = 8.854 * 10**(-12)
    X, Y = coordinates(a, n)
    L = np.ndarray(shape=(M, M))

    for i in range(M):
        for j in range(M):
            if(i == j):
                L[i][j] = (2 * b * 0.8814) / (np.pi * e)
            else:
                L[i][j] = ((b**2) / (np.pi * e * ((X[i] - X[j])**2 + (Y[i] - Y[j])**2)**0.5))/(np.pi*4*e)
                
    return L

def double_plate(a, n, d, v):
    N = n**2
    e = 8.854 * 10**(-12)
    b = a / (2 * n)
    """submatrices"""
    l = np.ndarray(shape=(2 * N, 2 * N))
    ltt = np.ndarray(shape=(N, N))
    ltb = np.ndarray(shape=(N, N))
    lbt = np.ndarray(shape=(N, N))
    lbb = np.ndarray(shape=(N, N))
    ltt = single_plate(a, n, v)
    lbb = single_plate(a, n, v)
    X, Y = coordinates(a, n)
    
    for i in range(N):
        for j in range(N):
            if(i == j):
                """
                temp1=(0.282*2*b)/e
                temp2=(1+((np.pi/4)*(d**2/b**2)))**0.5
                temp3=d*np.pi**0.5/(2*b)"""
                ltb[i][j] = (((0.282 * 2 * b) * ((1 + ((np.pi * d**2) / (4 * b**2)))**0.5 - ((d * np.pi**0.5) / (2 * b)))) / e)
                lbt[i][j] = ltb[i][j]  
            else:
                ltb[i][j] = (((a**2)/N)/((X[i]-X[j])**2+(Y[i]-Y[j])**2+d**2)**0.5)
                lbt[i][j] = ltb[i][j]  # Correctly assigning

    """Assigning the tt,tb,bt,bb values to l matrix"""
    for i in range(N):
        for j in range(N):
            l[i][j]=ltt[i][j]
        for k in range(N,2*N):
            l[i][k]=ltb[i][k-N]
    for i in range(N,2*N):
        for j in range(N):
            l[i][j]=lbt[i-N][j]
        for k in range(N,2*N):
            l[i][k]=lbb[i-N][k-N]
    print(l)
    li = np.linalg.inv(l)
    
    phi=np.ndarray(shape=(2*N,1))
    #print(phi)
    for i in range(N):
        phi[i][0]=-v
    for i in range(N,2*N):
        phi[i][0]=v
    alpha=np.matmul(li,phi)
    #print(alpha)
    Q=0
    
    for i in range(N):
        Q+=alpha[i]*(a/N)
        
    return float(Q)
X=[]
Y=[]
A=1
e = 8.854 * 10**(-12)
#d=np.linspace(0.00001, 1,100)
d=0.001
C0=e*A/d
C=double_plate(1, 1,0.001, 1)
print(C)
print(C0)
print(C/C0)
"""
for i in range(1,10):
    #C0=(e*A)/i
    X.append(i)
    Y.append(double_plate(1, i,0.000001, 1))
print(X,Y)
plt.plot(X,Y)"""

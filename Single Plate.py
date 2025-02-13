# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 12:46:48 2025

@author: anunn
"""
import numpy as np
import matplotlib.pyplot as plt

""" function to find the coordinates"""
def coordinates(a,n):
    coord_x=np.ndarray(shape=(n,n))                                                
    x=[]
    coord_y=np.ndarray(shape=(n,n))
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
    return x,y

"""function to find the charge of a single palte"""
def single_plate(a,n,v):
    #creating the coordinate matrix
    N=n**2
    b=a/(2*n)
    e=8.854*10**(-12)
    X,Y=coordinates(a,n)
    L=np.ndarray(shape=(N,N))

    for i in range(N):
        for j in range(N):
            if(i==j):
                L[i][j]=(2*b*0.8814)/(np.pi*e)
            else:
                L[i][j]=(b**2)/(np.pi*e*((X[i]-X[j])**2+(Y[i]-Y[j])**2)**0.5)
                
    LI=np.linalg.inv(L)
    V=np.ndarray(shape=(N,1))
    V.fill(v)
    alpha=np.matmul(LI,V)
    Q=0
    
    for i in range(N):
        Q+=alpha[i]*(1/N)
        
    return Q
"""
print("No. of subreas",end=' ')
print("lmn")
print("1",end=' ')
print(single_plate(1,1,1))
print("9",end=' ')
print(single_plate(1,3,1))
print("16",end=' ')
print(single_plate(1,4,1))
print("36",end=' ')
print(single_plate(1,6,1))
print("100",end=' ')
print(single_plate(1,10,1))
"""
X=[]
Y=[]
for i in range(21):
    if(i!=0):
        Y.append(single_plate(1,i,1))
        X.append(i)
    
plt.plot(X,Y)
plt.scatter(X,Y)
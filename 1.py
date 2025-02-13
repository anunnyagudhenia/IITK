# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np

def mulMat(m1, m2):
    r1 = len(m1)
    c1 = len(m1[0])
    r2 = len(m2)
    c2 = len(m2[0])

    if c1 != r2:
        print("Invalid Input")
        return None

    # Initialize the result matrix with zeros
    res = [[0] * c2 for _ in range(r1)]

    # Perform matrix multiplication
    for i in range(r1):
        for j in range(c2):
            for k in range(c1):
                res[i][j] += m1[i][k] * m2[k][j]

    return res


a=1                                                                            #length
n=2
m=n**2                                                                         #no. of subareas
e=8.854*10**(-12)                                                              #value of epsilon not  
v=1

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
##print(x)
##print(y)
d=4*np.pi*e

distance=np.ndarray(shape=(m,m))                                               #distance matrix
for i in range(m):
    for j in range(m):
        if i!=j:
            distance[i][j]=((x[i]- x[j])**2 + (y[i] - y[j])**2)**0.5

L=np.ndarray(shape=(m,m))
for i in range(m):
    for j in range(m):
        if i!=j:
            L[i][j]=(a**2)/(4*m*np.pi*e*distance[i][j])
        else:
            L[i][j]=((a/n)*0.8814)/(np.pi * e)
#print(L)
               
V=np.ndarray(shape=(m,1))
LI=np.linalg.inv(L)
V.fill(1)
alpha=mulMat(LI, V)
Q=0
for i in range(m):
    Q=Q+(alpha[i][0]*(1/m))
print(Q)

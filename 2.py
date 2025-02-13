# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 12:53:17 2025

@author: anunn"""
"""
import numpy as np
import matplotlib.pyplot as plt
e=8.854*10**(-12)
a=1
A=a**2
def charge(n):
    a=1                                                                            #length
    m=n**2                                                                         #no. of subareas
    e=8.854*10**(-12)
    b=a/(2*n)                                                        #value of epsilon not  
    v=1
    d=1
    ltt=np.ndarray(shape=(m,m))
    ltb=np.ndarray(shape=(m,m))
    lbt=np.ndarray(shape=(m,m))
    lbb=np.ndarray(shape=(m,m))
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
    d=4*np.pi*e

    distance=np.ndarray(shape=(m,m))                                               #distance matrix
    for i in range(m):
        for j in range(m):
            if i!=j:
                distance[i][j]=((x[i]- x[j])**2 + (y[i] - y[j])**2)**0.5

    for i in range(m):
        for j in range(m):
            if i!=j:
                ltt[i][j]=(a**2)/(4*m*np.pi*e*distance[i][j])
                lbb[i][j]=ltt[i][j]
            else:
                ltt[i][j]=((a/n)*0.8814)/(np.pi * e)
                lbb[i][j]=ltt[i][j]

    for i in range(m):
        for j in range(m):
            if(i!=j):
                ltb[i][j]=b**2/(np.pi*e*((x[i]-x[j])**2 +(y[i]-y[j])**2+d**2)**0.5)
                lbt[i][j]=ltb[i][j]
            else:
                ltb[i][j]=(0.282/e)*2*b*((1+((np.pi*d**2)/(4*b**2)))**0.5-((d*np.pi**0.5)/(2*b)))
                lbt[i][j]=ltb[i][j]
    print(ltt)
    l=np.ndarray(shape=(2*n,2*n))
    for i in range(n):
        for j in range(n):
            l[i][j]=ltt[i][j]
    k=0
    for i in range(n):
        for j in range(n,2*n):
            l[i][j]=ltb[i][k]
            k+=1
        k=0
    k=0
    for i in range(n,2*n):
        for j in range(n):
            l[i][j]=lbt[k][j]
        k+=1
    k=0
    m=0
    for i in range(n,2*n):
        for j in range(n,2*n):
            l[i][j]=lbb[k][m]
            m+=1
        m=0
        k+=1
    alpha=np.ndarray(shape=(2*n,1))
    gt=np.ndarray(shape=(n,1))
    gb=np.ndarray(shape=(n,1))
    gt.fill(v)
    gb.fill(-v)
    LI=np.subtract(ltt,ltb)
    LI=np.linalg.inv(LI)
    C=0
    for i in range(n):
        for j in range(n):
            C+=LI[i][j]
    C=C*4*b**2
    return C

d=0
X=[]
Y=[]
n=1
C=charge(36)
print(C)
print((C*2)/(e))

while(n<=36):
    C=charge(n)
    Y.append((C)/(e*A))
    X.append(d/(2*a))
    n+=1
plt.plot(X,Y)"""
import numpy as np
l=np.ndarray(shape=(2,2))
l[0][0]=1
l[0][1]=2
l[1][0]=3
l[1][1]=4
l=np.linalg.inv(l)
print(l)



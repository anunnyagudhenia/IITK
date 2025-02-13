# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 17:37:03 2025

@author: anunn
"""
import numpy as np
import math
e0=8.854 * 10**(-12)
h=1
M=1
N=1
r1=1
r2=1
def r(n):
    return r2+((r1-r2)*z(n)/h)
def z(n):
    return ((n-1)*h/N)+(h/(2*N))



L=np.ndarray(size=(M*N,M*N))

for i in range(M*N):
    for j in range(M*N):
        if(i!=j):
            temp1=4*np.pi*e0
            temp2=((r(i)/h)**2)+((r(j)/h)**2)-((2*r(i)*r(j)*np.cos(i-j)*2*np.pi)/(h*h*M))+(((i-j)/N)**2)
            S=temp1*(temp2)**0.5
            L[i][j]=((2*np.pi*r(j))/(M*N*h))/S
        else:
            temp1=1/(2*np.pi*e0)
            temp2=math.log((((2*np.pi*r(j)/h))*((1+(2*np.pi*r(j)/h)**2)**0.5)),np.e)
            temp3=math.log(((h/(2*np.pi*r(j)))*(1+(h/(2*np.pi*r(j)))**2)**0.5),math.e)/N
            temp4=temp1*(temp2*temp3)
            L[i][j]=temp4

LI=np.linalg(L)
sum=0
for i in range(M*N):
    for j in range(M*N):
        sum+=LI[i][j]*(2*np.pi*r(j)*h)/(M*N)
Q=V*sum

sum=0
for i in range(M*N):
    for j in range(M*N):
        sum+=((LI[i][j]*(2*np.pi*r(j)*h)/(M*N))/h)*(2*np.pi*r(j))/(M*N*h)
Ch=sum


        

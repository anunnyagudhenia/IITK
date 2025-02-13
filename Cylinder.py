# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 15:53:25 2025

@author: anunn
"""
import numpy as np
import math
import matplotlib.pyplot as plt
"""HOLLOW CYLINDER"""
e0=8.854 * 10**(-12)
def getL(M,N,h,r0):
    MN=M*N
    L=np.ndarray(shape=(MN,MN))
    for m in range(MN):
        for n in range(MN):
            if(m!=n):
                P=4*np.pi*e0*(((4*(r0/h)**2)*(np.pi*(np.sin(m-n))**2)/M)+(((m-n)/N)**2))**0.5
                l=((2*np.pi*r0)/(M*N*h))/P
                L[m][n]=h*l
            else:
                temp1=((2*np.pi*r0)/(M*h))*(math.log(((2*np.pi*r0)/h)*((1+((2*np.pi*r0)/h)**2)**0.5)))
                temp2=math.log((h/(2*np.pi*r0))*(1+(h/(2*np.pi*r0))**2)**0.5)/N
                l=(temp1+temp2)/(2*np.pi*e0)
                L[m][n]=h*l
    return L

M=11
N=9
h=2
r0=1
L=getL(M,N,h,r0)
LI=np.linalg.inv(L)

sum=0
for m in range(M*N):
    for n in range(M*N):
        sum+=((LI[m][n]*h)*((2*np.pi*r0)/(M*N*h)))
        
Ch=sum
print(Ch)

M=9
N=11
h=4
r0=1
L=getL(M,N,h,r0)
LI=np.linalg.inv(L)

sum=0
for m in range(M*N):
    for n in range(M*N):
        sum+=((LI[m][n]/h)*((2*np.pi*r0)/(M*N*h)))
        
Ch=sum
print(Ch)

"""
X=[2,4,12,40,120]
Y=[]
L1=getL(11,9,2,1)
L2=getL(9,11,4,1)
L3=getL(9,13,12,1)
L4=getL(10,30,40,1)
L5=getL(10,40,120,1)

LI1=np.linalg.inv(L1)
LI2=np.linalg.inv(L2)
LI3=np.linalg.inv(L3)
LI4=np.linalg.inv(L4)
LI5=np.linalg.inv(L5)

sum=0
for i in range(11*9):
    for j in range(11*9):
        sum+=(LI1[i][j]/2)*((2*np.pi)/(11*9*2))
Y.append(sum)
sum=0
for i in range(11*9):
    for j in range(11*9):
        sum+=(LI2[i][j]/4)*((2*np.pi)/(11*9*4))
Y.append(sum)
sum=0
for i in range(9*13):
    for j in range(9*13):
        sum+=(LI3[i][j]/12)*((2*np.pi)/(9*13*12))
Y.append(sum)
sum=0
for i in range(10*30):
    for j in range(10*30):
        sum+=(LI4[i][j]/40)*((2*np.pi)/(10*30*40))
Y.append(sum)
sum=0
for i in range(10*40):
    for j in range(10*40):
        sum+=(LI5[i][j]/120)*((2*np.pi)/(10*40*120))
Y.append(sum)

print(Y)
plt.plot(X,Y)
plt.scatter(X,Y)


M=int(input("Enter the value of M\n"))
N=int(input("Enter the value of N\n"))
h=int(input("Enter the height of the cylinder\n"))
r0=int(input("Enter the radius of the cylinder\n"))
V=int(input("Enter the value of V\n"))
L=getL(M,N,h,r0)
LI=np.linalg.inv(L)

sum=0

for i in range(M*N):
    for j in range(M*N):
        sum+=LI[i][j]*2*r0*np.pi*h/(M*N)
Q=V*sum
print("Q= ")
print(Q)

C=0
for i in range(M*N):
    for j in range(M*N):
        C+=(LI[i][j]*2*r0*np.pi/(M*N*h))*h
print("Capacitance per unit length= ")
print(C)

Csh=(0.708+0.615*(h/(2*r0))**0.76)*(r0/h)*10**(-10)
print("Smythe exp= ")
print(Csh)

print("Considering Top and bottom plate")
print("Capcitance= ")
print(Q+(16*e0*r0))"""









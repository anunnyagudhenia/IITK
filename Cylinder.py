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
    l=np.ndarray(shape=(MN,MN))
    a=(2*np.pi*r0)/M
    b=h/N
    for m in range(MN):
        for n in range(MN):
            if(m!=n):
                
                P=4*np.pi*e0*(((4*(r0/h)**2)*(np.pi*(np.sin(m-n))**2)/M)+(((m-n)/N)**2))**0.5
                l[m][n]=((2*np.pi*r0)/(M*N*h))/P
            else:
                temp1=a*math.log((b+math.pow(math.pow(a,2)+math.pow(b,2),0.5))/(-b+math.pow(math.pow(a,2)+math.pow(b,2),0.5)))
                temp2=b*math.log((a+math.pow(math.pow(a,2)+math.pow(b,2),0.5))/(-a+math.pow(math.pow(a,2)+math.pow(b,2),0.5)))
                l[m][n]=(temp1+temp2)/(4*np.pi*e0)
                """temp1=((2*np.pi*r0)/(M*h))*(math.log(((2*np.pi*r0)/h)*((1+((2*np.pi*r0)/h)**2)**0.5)))
                temp2=math.log((h/(2*np.pi*r0))*(1+(h/(2*np.pi*r0))**2)**0.5)/N
                l[m][n]=(temp1+temp2)/(2*np.pi*e0)"""
    return l
"""
M=11
N=9
h=2
r0=1
L=getL(M,N,h,r0)
LI=np.linalg.inv(L)

sum=0
for m in range(M*N):
    for n in range(M*N):
        sum+=((LI[m][n])*((2*np.pi*r0)/(M*N*h)))
        
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
        sum+=((LI[m][n])*((2*np.pi*r0)/(M*N*h)))
        
Ch=sum
print(Ch)

"""
X=[2,4,12,40,120]
Y=[]
l1=getL(11,9,2,1)
l2=getL(9,11,4,1)
l3=getL(9,13,12,1)
l4=getL(10,30,40,1)
l5=getL(10,40,120,1)
L1=l1*2
L2=l2*4
L3=l3*12
L4=l4*40
L5=l5*120
li1=np.linalg.inv(l1)
li2=np.linalg.inv(l2)
li3=np.linalg.inv(l3)
li4=np.linalg.inv(l4)
li5=np.linalg.inv(l5)

sum=0
for i in range(11*9):
    for j in range(11*9):
        sum+=(li1[i][j])*((2*np.pi)/(11*9*2))
Y.append(sum*10**(12))
sum=0
for i in range(11*9):
    for j in range(11*9):
        sum+=(li2[i][j])*((2*np.pi)/(9*11*4))
Y.append(sum*10**(12))
sum=0
for i in range(9*13):
    for j in range(9*13):
        sum+=(li3[i][j])*((2*np.pi)/(9*13*12))
Y.append(sum*10**(12))
sum=0
for i in range(10*30):
    for j in range(10*30):
        sum+=(li4[i][j])*((2*np.pi)/(10*30*40))
Y.append(sum*10**(12))
sum=0
for i in range(10*40):
    for j in range(10*40):
        sum+=(li5[i][j])*((2*np.pi)/(10*40*120))
Y.append(sum*10**(12))

print(Y)
plt.plot(X,Y)
plt.scatter(X,Y)

"""
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








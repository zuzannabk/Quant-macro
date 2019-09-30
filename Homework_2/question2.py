# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 02:05:32 2019

@author: Zuzanka
"""


import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import cm

alpha = 0.5
sigma = 0.25
nods = 20
N = [3, 9, 15]

def f(k, h):
    y = ((1-alpha)*k**((sigma-1)/sigma)+alpha*h**((sigma-1)/sigma))**(sigma/(sigma-1))
    return y

z = np.arange(1, nods+1)
z = np.cos((2*z-1)*math.pi/2/nods)
k = z*5 +5
h = z*5 +5

real_val=[]
for i in range(len(h)):
    real_val.append(f(k, h[i]))
real_val = np.matrix(real_val)

def polycheb(n, x): #basis functions #n-degreee of polinomials, x - grid
    psi =[]
    psi.append(np.ones(len(x)))
    psi.append(x)
    for i in range(1, n):
        psi_1 = 2*x*psi[i]-psi[i-1]
        psi.append(psi_1)
    y = np.matrix(psi[n])
    return y

def fun_theta(z, real_val, n):  
     theta = np.empty([n+1, n+1])
     for i in range(n+1):
         for j in range(n+1):
             theta[i,j] = (np.sum(np.array(real_val)*np.array((np.dot(polycheb(i, z).T, polycheb(j, z)))))/ np.array((polycheb(i,z)*polycheb(i,z).T)*(polycheb(j,z)*polycheb(j,z).T)))
     return theta

def chebyshev(k, h, theta, n):
    f = []
    k_arg = (2*k/10-1)
    h_arg = (2*h/10-1)
    for i in range(n):
        for j in range(n):
            f.append(np.array(theta[i,j]*np.array((np.dot(polycheb(i,k_arg).T, polycheb(j, h_arg))))))
    f_tylda = sum(f)
    return f_tylda


fig = plt.figure()
ax = fig.add_subplot(projection='3d')
X, Y = np.meshgrid(k, h)
surf = ax.plot_surface(k, h, real_val, alpha = 0.5, cmap=cm.bwr, linewidth=0, antialiased=False)
fig.colorbar(surf, shrink=1, aspect=10)
plt.title('Real value')
plt.show()
plt.clf()

for n in N:
    theta = fun_theta(z, real_val, n)
    fit = chebyshev(k, h, theta, n)
    error = real_val - fit
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    X, Y = np.meshgrid(k, h)
    surf = ax.plot_surface(k, h, fit, alpha = 0.5, cmap=cm.bwr, linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=1, aspect=10)
    plt.title('Chebyshev approximation of degree ' +str(n))
    plt.show()
    plt.clf()
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    surf = ax.plot_surface(k, h, error, alpha = 0.7, cmap=cm.bwr, linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=1, aspect=10)
    plt.title('Chebyshev error of degree ' +str(n))
    plt.show()
    plt.clf()
    
    
    
# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import math
import matplotlib.pyplot as plt

def fun_a (x):
    return x**0.321

def derivative_a (N, x):
    par=1
    for n in np.arange(0, N, 1):
        par=par*(0.321-n)
    f_x = par * x**(0.321-N)
    return f_x

def fun_ramp(x):
    y=(x+np.absolute(x))/2
    return y

def derivative_ramp(N, x):
    if N==0:
        return fun_ramp(x)
    elif np.logical_or(N<0, x==0 ):
        print('no derivative')
    elif np.logical_or(x<0, N>1):
        return 0
    else:
        return 1

def taylor (N, x, x_bar, derivative):
    taylor = 0
    for n in np.arange(0, N+1, 1):
        taylor += derivative(n, x_bar) / math.factorial(n) * (x - x_bar) ** n
    return taylor

N=np.array([1, 2, 5, 20]) #degrees of aprox

############ point 1################
x_bar = 1 # point of aprox
x_grid = np.linspace(0, 4, 200) #x_grid    

real_val = fun_a(x_grid)
plt.plot(x_grid, real_val, label="Real value", alpha=0.8)

for i in N:
    y = taylor(i, x_grid, x_bar, derivative_a)
    plt.plot(x_grid, y, alpha = 0.8, label="Approximation of degree " + str (i))

plt.title('Taylor approximation of x**0.321')
plt.ylim([0,4])
plt.legend()
plt.show()
plt.clf()

#################point 2######################
x_bar = 2 # point of aprox
x_grid = np.linspace(-2, 6, 200) #x_grid   

real_val = fun_ramp(x_grid)
plt.plot(x_grid, real_val, label="Real value", alpha=0.8)

for i in N:
    y = taylor(i, x_grid, x_bar, derivative_ramp)
    plt.plot(x_grid, y, alpha = 0.8, label="Approximation of degree " + str (i))

plt.title('Taylor approximation of Ramp function')
plt.ylim([-1,5])
plt.legend()
plt.show()
plt.clf()






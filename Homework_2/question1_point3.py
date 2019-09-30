# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 00:23:03 2019

@author: Zuzanka
"""

import numpy as np
import math
import matplotlib.pyplot as plt
import numpy.polynomial.polynomial as poly
import numpy.polynomial.chebyshev as cheby


def fun_b(x):
    y=np.exp(1/x)
    return y

def fun_runge(x):
    y=1/(1+25*x**2)
    return y

def fun_ramp(x):
    y=(x+np.absolute(x))/2
    return y

N=np.array([3, 5, 10])

#####Normal Polynomials#####
exercises = ['evenly spaced interpolation nodes', 'Chebyshev interpolation nodes']
for ex in exercises:
##exp###
    x_grid = np.linspace(-1, 1, 501)
    real_val = fun_b(x_grid)
    plt.plot(x_grid, real_val, label="Real value", alpha=0.8)
    
    if ex == 'Chebyshev interpolation nodes':
        x_grid = np.arange(500)
        x_grid = np.cos((2*x_grid-1)*math.pi/2/500)
    else:
        x_grid = np.linspace(-1, 1, 500)
        
    real_val = fun_b(x_grid)
    
    error = []
    for i in N:
        fit_b = poly.polyfit(x_grid, real_val, i)
        fit_b = poly.polyval(x_grid, fit_b)
        error.append(real_val-fit_b)
        plt.plot(x_grid, fit_b, alpha = 0.8, label="Polynomial of order " + str(i))

    plt.ylim([0,1000])
    plt.title('Approximation of exp(1/x) - '+ex)
    plt.legend()
    plt.show()
    plt.clf()

    for i in range(0, 3):
        plt.plot(x_grid, error[i], label="Polynomial of order " + str(N[i]))
    plt.ylim([0,1000])
    plt.title('Error of approximation of exp(1/x) - ' +ex)
    plt.legend()
    plt.show()
    plt.clf()

###runge###
    x_grid = np.linspace(-1, 1, 501)
    real_val = fun_runge(x_grid)
    plt.plot(x_grid, real_val, label="Real value", alpha=0.8)

    if ex == 'Chebyshev interpolation nodes':
        x_grid = np.arange(500)
        x_grid = np.cos((2*x_grid-1)*math.pi/2/500)
    else:
        x_grid = np.linspace(-1, 1, 500)
        
    real_val = fun_runge(x_grid)
    error = []

    for i in N:
        fit_runge = poly.polyfit(x_grid, real_val, i)
        fit_runge = poly.polyval(x_grid, fit_runge)
        error.append(real_val-fit_runge)
        plt.plot(x_grid, fit_runge, alpha = 0.8, label="Polynomial of order " + str(i))


    plt.title('Approximation of runge function - ' +ex)
    plt.legend()
    plt.show()
    plt.clf()

    for i in range(0,3):
        plt.plot(x_grid, error[i], label="Polynomial of order " + str(N[i]))
    plt.title('Error of approximation of runge function - ' +ex)
    plt.legend()
    plt.show()
    plt.clf()
    
###ramp###
    x_grid = np.linspace(-1, 1, 501)
    real_val = fun_ramp(x_grid)
    plt.plot(x_grid, real_val, label="Real value", alpha=0.8)

    if ex == 'Chebyshev interpolation nodes':
        x_grid = np.arange(500)
        x_grid = np.cos((2*x_grid-1)*math.pi/2/500)
    else:
        x_grid = np.linspace(-1, 1, 500)
        
    real_val = fun_ramp(x_grid)
    error = []

    for i in N:
        fit_ramp = poly.polyfit(x_grid, real_val, i)
        fit_ramp = poly.polyval(x_grid, fit_ramp)
        error.append(real_val-fit_ramp)
        plt.plot(x_grid, fit_ramp, alpha = 0.8, label="Monomial of order " + str(i))

    plt.title('Approximation of ramp function - ' +ex)
    plt.legend()
    plt.show()
    plt.clf()

    for i in range(0,3):
        plt.plot(x_grid, error[i], label="Monomial of order " + str(N[i]))
    plt.title('Error of approximation of ramp function - ' +ex)
    plt.legend()
    plt.show()
    plt.clf()

########################Chebyshev polynomial####################
def polycheb(n, x): #basis functions #n-degreee of polinomials, x - grid
    psi =[]
    psi.append(np.ones(len(x)))
    psi.append(x)
    for i in range(1, n):
        psi_1 = 2*x*psi[i]-psi[i-1]
        psi.append(psi_1)
    return psi[n]

def fun_theta(z, real_val, n):  
     theta = np.empty(n+1)
     for i in range(n+1):
         theta[i] = (np.sum(real_val*polycheb(i, x_grid)))/ np.sum(polycheb(i, x_grid)*polycheb(i, x_grid))
     return theta

def chebyshev(x, theta, n):
    f = 0
    x_arg = (2*(x+1)/2-1)
    for i in range(n):
        f += theta[i]*polycheb(i,x_arg)
    return f

#########exp############
x_grid = np.linspace(-1, 1, 501)
real_val = fun_b(x_grid)
plt.plot(x_grid, real_val, label="Real value", alpha=0.8)

x_grid = np.arange(500)
x_grid = np.cos((2*x_grid-1)*math.pi/2/500)
    
real_val = fun_b(x_grid)

error = []
for n in N:
    theta = fun_theta(x_grid, real_val, n)
    fit = chebyshev(x_grid, theta, n)
    error.append(real_val - fit)
    plt.plot(x_grid, fit, label = 'Chebyshev poly of deg '+str(n))

plt.ylim([0,1000])
plt.title('Chebychev approximation of exp(1/x) - Chebyshev nodes')
plt.legend()
plt.show()
plt.clf()

for i in range(0, 3):
    plt.plot(x_grid, error[i], label="Polynomial of order " + str(N[i]))
plt.ylim([0,1000])    
plt.title('Chebychev error of approximation of exp(1/x)')
plt.legend()
plt.show()
plt.clf()

########runge###########
x_grid = np.linspace(-1, 1, 501)
real_val = fun_runge(x_grid)
plt.plot(x_grid, real_val, label="Real value", alpha=0.8)

x_grid = np.arange(500)
x_grid = np.cos((2*x_grid-1)*math.pi/2/500)
    
real_val = fun_runge(x_grid)

error = []
for n in N:
    theta = fun_theta(x_grid, real_val, n)
    fit = chebyshev(x_grid, theta, n)
    error.append(real_val - fit)
    plt.plot(x_grid, fit, label = 'Chebyshev poly of deg '+str(n))

plt.title('Chebychev approximation of runge function - Chebyshev nodes')
plt.legend()
plt.show()
plt.clf()

for i in range(0, 3):
    plt.plot(x_grid, error[i], label="Polynomial of order " + str(N[i]))    
plt.title('Chebychev error of approximation of runge function')
plt.legend()
plt.show()
plt.clf()

#####ramp#########
x_grid = np.linspace(-1, 1, 501)
real_val = fun_ramp(x_grid)
plt.plot(x_grid, real_val, label="Real value", alpha=0.8)

x_grid = np.arange(500)
x_grid = np.cos((2*x_grid-1)*math.pi/2/500)
    
real_val = fun_ramp(x_grid)

error=[]
for n in N:
    theta = fun_theta(x_grid, real_val, n)
    fit = chebyshev(x_grid, theta, n)
    error.append(real_val - fit)
    plt.plot(x_grid, fit, label = 'Chebyshev poly of deg '+str(n))

plt.title('Chebychev approximation of ramp function - Chebyshev nodes')
plt.legend()
plt.show()
plt.clf()

for i in range(0,3):
    plt.plot(x_grid, error[i], label="Polynomial of order " + str(N[i]))    
plt.title('Chebychev error of approximation of ramp function')
plt.legend()
plt.show()
plt.clf()




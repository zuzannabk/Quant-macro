# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 22:01:58 2019

@author: Zuzanka
"""

#import sympy as sy
import numpy as np
#import matplotlib as mpl
from scipy.optimize import fsolve

#Parameters:
name=['A', 'B']
kappa = 5
v    = 1
sigma = 0.8
eta_ll = [0.5, 2.5]
eta_hh = [5.5, 3.5]
zeta  = 1
theta = 0.6
cap_up = 2
lamm = [0.95, 0.84]
phi   = 0.2

cap_ll = [1, 1]
cap_hh = [1,1]

# Solving our set of equations:

#0 - r, 1-w, 2->hl, 3->hH, 4->cL, 5->cH,
def f(x, eta_l, eta_h, lam, cap_l, cap_h):
    # From firms problem:
    f1 = (1-theta)*(zeta)*pow(cap_up,-theta)*pow(x[2]*eta_l+x[3]*eta_h, theta)-x[0]
    f2 = (theta)*(zeta)*pow(cap_up,1-theta)*pow(x[2]*eta_l+x[3]*eta_h, theta-1)-x[1]
    # From HHs problem:
    f3 = (lam)*(1-phi)*pow(x[1]*(eta_h),1-phi)*pow(x[5],-sigma)-kappa*pow(x[3],(phi) + (1/v))
    f4 = (lam)*(1-phi)*pow(x[1]*(eta_l),1-phi)*pow(x[4],-sigma)-kappa*pow(x[2],(phi) + (1/v))
    # Budget constraints
    f5 =  (lam)*pow(x[1]*x[3]*eta_h,1-phi)+x[0]*pow(cap_h, eta_h)-x[5]
    f6 =  (lam)*pow(x[1]*x[2]*eta_l,1-phi)+x[0]*pow(cap_l, eta_l)-x[4]
    return[f1,f2,f3,f4,f5,f6]
equil = np.empty([len(name), 6], dtype=object)
for i in range(len(name)):
    f_i = lambda x : f(x, eta_ll[i], eta_hh[i], lamm[i], cap_ll[i], cap_hh[i])
    equil[i] = (fsolve(f_i, [1,1,1,1,1,1]))

#Equilibrium
    print(str(name[i]),'-Rate of return:', round(equil[i,0],2))
    print(str(name[i]),'-Wages:', round(equil[i,1],2))  
    print(str(name[i]),'-Labor supply (low type)', round(equil[i,2],2))
    print(str(name[i]),'-Labor supply (high type):', round(equil[i,3],2))
    print(str(name[i]),'-Consumption (low types):', round(equil[i,4],2))
    print(str(name[i]),'-Consumption (high type):', round(equil[i,5],2))

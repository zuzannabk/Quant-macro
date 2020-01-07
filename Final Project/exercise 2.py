# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 23:02:33 2020

@author: Zuzanka
"""

import numpy as np
import math
import matplotlib.pyplot as plt
import quantecon as qe
from scipy import optimize

alpha = 0.3
beta  = 0.99**40
tau   = 0.0
lambd = 0.5
g     = 0.0 
T  = 1000

zetamean = 1.0
lnzetamean=0.0
rhomean  = 1.0
lnrhomean=0.0
etamean  = 1.0
lnetamean=0.0

std_lnzeta = 0.13
std_lnrho  = 0.50
std_lneta  = 0.95


np.random.seed(seed=123)

probhighz=0.5
lnzlow=0.97
lnzhigh=1.03

zeta = np.random.uniform(0,1, size=T)
zeta[0]=lnzhigh
for i in range(1,T):
    if zeta[i]>0.05: 
        zeta[i]=zeta[i-1]
    else: 
        if zeta[i-1]==lnzlow:
            zeta[i]=lnzhigh
        else: zeta[i]=lnzlow
zeta=np.log(zeta)

eta=qe.quad.qnwnorm(11, 1, std_lneta**2)
eta_disc=np.exp(eta[0])
np.mean(eta)
np.std(eta)

phih =probhighz*eta[1]/(1+(1-alpha)*(lambd*eta_disc[0]+tau*(1+lambd*(1-eta_disc[0])))/(alpha*math.exp(lnzhigh)*(1+lambd)))
phil =(1-probhighz)*eta[1]/(1+(1-alpha)*(lambd*eta_disc[0]+tau*(1+lambd*(1-eta_disc[0])))/(alpha*math.exp(lnzlow)*(1+lambd)))
phi=sum(phih)+sum(phil)

s=beta*phi/(1+beta*phi)

lnk_ss=math.log((1-alpha)*s*(1-tau)/(1-lambd))/(1-alpha)

lnk = [lnk_ss]

for i in range (1,T):
    lnk.append(math.log((1-alpha)/(1-lambd))+math.log(s)+math.log(1-tau)+zeta[i]+alpha*lnk[i-1])

lnkmean=np.mean(lnk)

plt.plot(range(0,T), lnk, label='capital', alpha=0.7)
plt.hlines(lnkmean, 0 ,T, ls='--', label="mean",  color="red")
plt.hlines(lnk_ss, 0 ,T, label='steady state', color="green", alpha=0.5)
plt.title('Capital path over time - exercise 2 T=1000')
plt.xlabel('Periods')
plt.ylabel('Capital stock')
plt.legend()
plt.show()
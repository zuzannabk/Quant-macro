# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 20:22:57 2020

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
T  = 50000

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

#####################1.2 continous ######################

zeta  = np.random.lognormal(mean=lnzetamean, sigma=std_lnzeta, size=T)
rho   = np.random.lognormal(mean=lnrhomean, sigma=std_lnrho, size=T)
eta   = np.random.lognormal(mean=lnetamean, sigma=std_lneta, size=T)

phi = 1/(1+(1-alpha)*(lambd*etamean+tau*(1+lambd*(1-etamean)))/(alpha*zetamean*(1+lambd)))
s=beta*phi/(1+beta*phi)

lnk_ss=math.log((1-alpha)*s*(1-tau)/(1-lambd))/(1-alpha)

lnk = [lnk_ss]

for i in range (1,T):
    lnk.append(math.log((1-alpha)/(1-lambd))+math.log(s)+math.log(1-tau)+math.log(zeta[i])+alpha*lnk[i-1])

lnkmean=np.mean(lnk)

plt.scatter(range(0,T), lnk, label='capital', alpha=0.7, s=0.5)
plt.hlines(lnkmean, 0 ,T, ls='--', label="mean",  color="red")
plt.hlines(lnk_ss, 0 ,T, label='steady state', color="green", alpha=0.5)
plt.title('Capital path over time- exercise 1.2 continuous case')
plt.xlabel('Periods')
plt.ylabel('Capital stock')
plt.legend()
plt.show()

#------------------------1.2 discrete--------------------------------------------------------

probhighz=0.5
lnzlow=-std_lnzeta*math.sqrt(probhighz/(1-probhighz))
lnzhigh=std_lnzeta*math.sqrt((1-probhighz)/probhighz)

probhighr=0.5
lnrlow=-std_lnrho*math.sqrt(probhighr/(1-probhighr))
lnrhigh=std_lnrho*math.sqrt((1-probhighr)/probhighr)

zeta = np.random.binomial(n=1, p=probhighz, size=T)
zeta=zeta*(-lnzlow+lnzhigh)
zeta=zeta+lnzlow
zeta_disc=np.exp([lnzlow,lnzhigh])

rho = np.random.binomial(n=1, p=probhighr, size=T)
rho=rho*(-lnrlow+lnrhigh)
rho=rho+lnrlow
rho_disc=[lnrlow,lnrhigh]

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

plt.scatter(range(0,T), lnk, label='capital', alpha=0.7, s=0.1)
plt.hlines(lnkmean, 0 ,T, ls='--', label="mean",  color="red")
plt.hlines(lnk_ss, 0 ,T, label='steady state', color="green", alpha=0.5)
plt.title('Capital path over time - exercise 1.2 discrete case with probability=0.125')
plt.xlabel('Periods')
plt.ylabel('Capital stock')
plt.legend()
plt.show()

#--------------------------------1.3-------------------------------------------

# Phi from the HL paper
phi = (1)/(1+((1-alpha)*(lambd*etamean))/(alpha*(1+lambd)*rhomean))

# Saving rate:
sav_rate = (beta*phi)/(1 + (beta*phi))

# Computing psi zero
psi_0 = np.log(sav_rate) + np.log(1-tau) + np.log((1-alpha)/(1-lambd)) + np.log(zetamean)

# Computing psi one
psi_1 = alpha

#-----b) Carry out the algorithm to convergence--------------------------------
#--------i) In iteration m, solve the household problem------------------------

#Defining capital path
rho1=1
eta1=1
#print('Steady state k:', k_ss, 'and log(k):', ln_k_ss) 
# Setting the grid
k_grid= np.linspace(0.5*np.exp(lnk_ss),1.5*np.exp(lnk_ss), 5) 

def capital_path(s, k1):
    psi_0=np.log(s) + np.log(1-tau) + np.log((1-alpha)/(1-lambd)) + j
    ln_cap = psi_0 + psi_1*np.log(k1)
    return np.exp(ln_cap)

# Wages (from Equation 3b from the HL paper)
def wage(k1, zeta1):
    w = (1-alpha)*(1+g)*(k1**alpha)*zeta1
    return w

# Interest rate (from Equation 3a)
def rate(k1, zeta1, rho1):
    r = alpha*pow(k1, alpha-1)*zeta1*rho1-1
    return r

# Assets
def assets(k1, s, zeta1):
    a = (s)*wage(k1, zeta1)
    return a

# Equation 2a
def c_young(k1, s, zeta1):
    c1 = (1-s)*wage(k1, zeta1)
    return c1

# Equation 2b
def c_old(k1, zeta1, zeta2, rho1, s):
    k2=capital_path(s, k1)
    c2 = assets(k1, s, zeta1)*rate(k2, zeta1, rho1)+ lambd*eta1*wage(k2, zeta2)*(1-tau)+(1+lambd)*tau*wage(k2, zeta2)
    return c2
    
s_matrix=np.zeros([5,2])
for kk, k in enumerate(k_grid):
    for jj, j in enumerate(zeta_disc):
        def Euler(s):
            
            c1=c_young(k, s, j)
            c2=c_old(k, j, 1, rho1, s)
            r=rate(k,j, 1)
            wynik=beta*c1*(1-r)/c2-1
            return wynik
        s_matrix[kk,jj]=optimize.brentq(Euler,0.00001,1)
psi0_matrix=np.log(s_matrix) + np.log(1-tau) + np.log((1-alpha)/(1-lambd)) + j


# Inititial values:

lncap=[lnk_ss]
betahat=beta/(1+beta)
u=[]

for i in range (0,T-1):
    j=np.argmin(np.abs(lncap[i]-k_grid))
    if np.exp(zeta[i])==zeta_disc[0]:
        k=0
    else: k=1
    if i>0:
        u.append((1-betahat)*c_young(np.exp(lncap[i-1]),s,zeta[i])+betahat*c_old(np.exp(lncap[i-1]), zeta[i-1], zeta[i], rho1, s))
    s=s_matrix[j][k]
    lncap.append(math.log((1-alpha)/(1-lambd))+math.log(s)+math.log(1-tau)+zeta[i]+alpha*lncap[i])

lnkmean=np.mean(lncap)
plt.scatter(range(0,T), lncap, label='capital', alpha=0.7, s=0.5)
plt.hlines(lnkmean, 0 ,T, ls='--', label="mean",  color="red")
#plt.hlines(lnk_ss, 0 ,T, label='steady state', color="green", alpha=0.5)
plt.title('Capital path over time- exercise 1.3')
plt.xlabel('Periods')
plt.ylabel('Capital stock')
plt.legend()
plt.show()









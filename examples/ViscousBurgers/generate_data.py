import numpy as np
import random as rn

# The below is necessary for starting Numpy generated random numbers
# in a well-defined initial state.
np.random.seed(1)
# The below is necessary for starting core Python generated random numbers
# in a well-defined state.
rn.seed(1)

# Importing the necessary lib

from numpy import linalg
import os
import math
from scipy.integrate import odeint, fixed_quad
from scipy.io import savemat
import matplotlib.pyplot as plt

# Storage directory

def RandU0(xx,fourier_coef,fourier_N):
    myU0 = 0.0
    for i in range(fourier_N):
        myU0 = myU0 + fourier_coef[i]*np.sin( (i+1) * xx )
    return myU0

def Randphi0(xx,fourier_coef,fourier_N,nu):
    myU0 = 0.0
    for i in range(fourier_N):
        myU0 = myU0 + fourier_coef[i]*(1.0-np.cos( (i+1.0) * xx ))/(i+1.0);
    myU0 = np.exp( - myU0 / (2.0*nu) )
    return myU0


def RandUt(xx,phiexpan_coef,N_exact_trucated,nu,dt):
    tempu_up=0.0
    tempu_down=0.0
    for j in np.arange(1,N_exact_trucated+1):
        tempu_up = tempu_up + j*phiexpan_coef[j]*np.sin( j*xx )*np.exp( -nu*j*j*dt )
        tempu_down = tempu_down + phiexpan_coef[j]*np.cos( j*xx )*np.exp( -nu*j*j*dt )
    temp = 2*nu*tempu_up/( phiexpan_coef[0] + tempu_down )
    return temp

def RandUtsin(x,phiexpan_coef,N_exact_trucated,nu,dt,k):
    uu=RandUt(x,phiexpan_coef,N_exact_trucated,nu,dt)*np.sin(k*x)
    return uu


def Randphi0cos( x,fourier_coef,fourier_N,nu,k ):
    uu=Randphi0(x,fourier_coef,fourier_N,nu)*np.cos(k*x)
    return uu
    
fourier_N = 10; # Fourier expansion N

dt=0.05
nu = 0.1
n_train=1000 #50*n_param
n_test = 100
xi_L = 0
xi_R = 2*np.pi
N_exact_trucated=160

# training data

K  = 128
Nc = fourier_N

x  = np.reshape(np.linspace(xi_L,xi_R,K+1)[:-1],(K,1))

bn = np.random.uniform(-1,1,size=(n_train,Nc))
scaling = np.asarray([1.0/(1+i) for i in range(Nc)]).reshape((1,Nc))
bn = bn * scaling
sinxx = np.sin(x.dot(1.0/scaling).T)# (Nc,K)
inputs_train_nodal = bn.dot(sinxx) #(N,K)
inputs_train  = bn #(n_train,d)
# randomly generateing sample data within x_domain
# generating outputs_train

T=40
outputs_train = np.zeros((n_train,fourier_N,T+1))
outputs_train_nodal = np.zeros((n_train,K,T+1))
outputs_train[...,0] = inputs_train
outputs_train_nodal[...,0] = inputs_train_nodal

for t in np.arange(1,T+1):
    for i in range(n_train):
        if np.remainder(i,100) == 0:
            print(i)
        fourier_coef = outputs_train[i,:, t-1]
        
        phiexpan_coef = np.zeros(N_exact_trucated+1)
        
        for j in np.arange(0,N_exact_trucated+1):
            (temp_val,temperr) = fixed_quad(Randphi0cos, xi_L, xi_R, args=(fourier_coef,fourier_N,nu,j),n=100)
            phiexpan_coef[j] = temp_val / np.pi
        phiexpan_coef[0] = phiexpan_coef[0]/2.0
        
        for j in np.arange(1,fourier_N+1):
            (temp_val,temperr) = fixed_quad(RandUtsin, xi_L, xi_R, args=(phiexpan_coef,N_exact_trucated,nu,dt,j),n=100)#, tol=1.49e-6, rtol=1.49e-6, maxiter=300)
            outputs_train[i,j-1] = temp_val / np.pi
    
    outputs_train_nodal[...,t] = outputs_train[:,:,t].dot(sinxx) #(N,K)
#savemat("data_modal.mat", mdict={"X":inputs_train, "Y":outputs_train})
savemat("ViscousBurgers_train.mat", mdict={"coordinates":x, "trajectories":outputs_train_nodal[:,:,np.newaxis,:]})
########################################
bn = np.random.uniform(-1,1,size=(n_test,Nc))
bn = bn * scaling
inputs_test_nodal = bn.dot(sinxx) #(N,K)
inputs_test  = bn #(n_train,d)
# randomly generateing sample data within x_domain
# generating outputs_train
T = 200
outputs_test = np.zeros((n_test, fourier_N, T+1))
outputs_test_nodal = np.zeros((n_test, K, T+1))
outputs_test[...,0] = inputs_test
outputs_test_nodal[...,0] = inputs_test_nodal
print(n_test)

for t in np.arange(1,T+1):
    for i in range(n_test):
        if np.remainder(i,100) == 0:
            print(i)
        fourier_coef = outputs_test[i,:,t-1]
        
        phiexpan_coef = np.zeros(N_exact_trucated+1)
        
        for j in np.arange(0,N_exact_trucated+1):
            (temp_val,temperr) = fixed_quad(Randphi0cos, xi_L, xi_R, args=(fourier_coef,fourier_N,nu,j),n=100)
            phiexpan_coef[j] = temp_val / np.pi
        phiexpan_coef[0] = phiexpan_coef[0]/2.0
        
        for j in np.arange(1,fourier_N+1):
            (temp_val,temperr) = fixed_quad(RandUtsin, xi_L, xi_R, args=(phiexpan_coef,N_exact_trucated,nu,dt,j),n=100)#, tol=1.49e-6, rtol=1.49e-6, maxiter=300)
            outputs_test[i,j-1] = temp_val / np.pi
    
    outputs_test_nodal[...,t] = outputs_test[:,:,t].dot(sinxx) #(N,K)
#savemat("data_modal_test.mat", mdict={"X":inputs_test, "Y":outputs_test[...,1:]})
savemat("ViscousBurgers_test.mat", mdict={"trajectories":outputs_test_nodal})

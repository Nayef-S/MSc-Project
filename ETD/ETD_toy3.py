#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 18:36:19 2021

@author: esuwws
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from ETD import ETD1, ETD2


#initialisation
rng = np.random.default_rng(1)

L = 2* np.array([[-1,-1],[1,-1]]) #-1 * np.asarray([[1,0],[0,1]])   #[[1,1],[1,0]]
#* rng.integers(10,size = (2,2))  #* np.eye(2)

R = L.T

X_0 = rng.standard_normal(size = L.shape)

r = np.identity(L.shape[0])

B = np.identity(L.shape[0]) #2 * L

Q =  np.identity(L.shape[0])

N =  -1* X_0 @ B @ r @ B.T @ X_0 + Q

X_true = linalg.solve_continuous_are(L,B,Q,r) #-np.identity(L.shape[0])


dt = 0.001

t = 0
    
t_final = 100

n_steps = round(t_final/dt)

error_array = np.zeros(n_steps)

m1_L_1, m1_R_1, m2_1 = ETD1(L,R,N,dt)

for j in range(n_steps):
    
    X_1 = m1_L_1 @ X_0 @ m1_R_1 + m2_1 @ N
    
    N =  -1* X_0 @ B @ r @ B.T @ X_0 + Q
       
    X_0 = X_1.copy()  
    
    t += dt 
        
    error_array[j] = np.linalg.norm(X_true-X_1) 
    
plt.plot(error_array)
plt.yscale('log')
plt.xscale('log')

print(X_1 @ L + R @ X_1 - X_1 @ B @ r @ B.T @ X_1 + Q)
print(X_true @ L + R @ X_true - X_true @ B @ r @ B.T @ X_true + Q)
print(X_1)



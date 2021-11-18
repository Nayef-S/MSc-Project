#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 12:05:21 2021

@author: nshkeir
"""
import numpy as np
import matplotlib.pyplot as plt
from ETD import ETD1, ETD2




#initialisation
rng = np.random.default_rng(1)


L = np.array([[-2,1],[-1,-3]]) #-1 * np.asarray([[1,0],[0,1]])   #[[1,1],[1,0]]
#* rng.integers(10,size = (2,2))  #* np.eye(2)


a = np.asarray([[7,2],[2,7]])

N = a

dt_array = [0.00001, 0.00005, 0.0001, 0.0005, 0.001 ,0.005 , 0.01, 0.05, 0.1]

error_array = np.zeros(len(dt_array))

time_array = np.zeros(len(dt_array))

A_true = linalg.solve_sylvester(L,L,-N)

for i in range(len(dt_array)):
    
    A_0 = rng.standard_normal(size = L.shape)
    
    print(A_0)
    
    dt = dt_array[i]

    t = 0
    
    t_final = 10
    
    n_steps = round(t_final/dt)
    
    m1, m2 = ETD1(L,N,dt)
    
    for j in range(n_steps):
        
        A = m1 @ A_0 @ m1 + m2 @ N #+ m3 @ N # integration by 1 timestep
        
        A_0 = A.copy()
        
        t += dt 
        
    error_array[i] = np.linalg.norm(A_true-A) # store error
        
    time_array[i] = dt_array[i]  # store time step
    
    
plt.plot(time_array,error_array)
plt.plot(time_array, time_array)
plt.yscale('log')
plt.xscale('log')
plt.ylabel(r'$|| a - A^{(n)}||_{2}$')
plt.xlabel(r'$\Delta t$')


#%%
import numpy as np
import matplotlib.pyplot as plt
from ETD import ETD1 , ETD2


#initialisation
rng = np.random.default_rng(1)


L = -1 * np.asarray([[1,0],[0,1]])   #[[1,1],[1,0]]
#* rng.integers(10,size = (2,2))  #* np.eye(2)


a = np.asarray([[7,2],[2,1]])

N = 2*a

A_0 = rng.standard_normal(size = L.shape)


dt = 0.001

n_steps = round(100/dt)

error_array = np.zeros(n_steps)
time_array = np.zeros(n_steps)

m1, m2 = ETD1(L,N,dt)

for j in range(n_steps):
    
    A = m1 @ A_0 @ m1 + m2 @ N # integration by 1 timestep
    
    A_0 = A.copy()
    
    error_array[j] = np.linalg.norm(a-A_0) # store error
    
    # store time step
    
plt.plot(error_array)
plt.yscale('log')




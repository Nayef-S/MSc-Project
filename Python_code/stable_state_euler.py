#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#%%

import math
import numpy as np
from tqdm import tqdm
from numpy.matlib import repmat
from scipy import linalg
from Gamma_lyap import Gamma_U
import matplotlib.pylab as plt

Nx = 64
Ny = 128
delta = 1.0
L = 2*math.pi

alp = 0.01
nu = 1e-6
beta = 4.5
p=2
dt = 1e-2
T = 1

dy = L*delta/Ny
yy = np.linspace(0,L*delta - dy,num=Ny)
sigma = np.load('sigma.npy')


# D^2 operator in y

D2y = np.diag(np.linspace(1, 1, Ny-1), +1) - 2*np.diag(np.linspace(1, 1, Ny)) + np.diag(np.linspace(1, 1, Ny-1), -1)

#periodic BCs

D2y[0, Ny-1] = 1
D2y[-1, 0] = 1

D2y = D2y/(dy**2)
I = np.eye(Ny)

U = np.zeros((128,int(T/dt)+1))

U[:,0] = 0.2*np.sin(2*yy/delta)

count = 0
i = 0

# k  = 13

# Gamma_k  = Gamma_U(U[:,count],k,D2y,I)

# sigma_real = np.real(np.fft.ifft(sigma[:,k])).reshape(Ny,1)

# Chi_k = 2 * (sigma_real @ sigma_real.conj().T ) / sigma_real.shape[0]

# C_k = linalg.solve_continuous_lyapunov(Gamma_k, Chi_k)

# A = np.diag((-2*k*np.linalg.inv(D2y - k**2 * I)) @ np.imag(C_k))


#%%

while i < T:

    C = np.zeros((Ny))
    
    for k in range(0,13):
    
        sigma_real = np.real(np.fft.ifft(sigma[:,k])).reshape(Ny,1)
        
        Chi_k = 2*sigma_real @ sigma_real.conj().T #/ sigma_real.shape[0]**2
        
        Gamma_k  = Gamma_U(U[:,count],k,D2y,I)
        
        C_k = linalg.solve_continuous_lyapunov(Gamma_k, Chi_k)
        
        A = np.diag((-2*k*np.linalg.inv(D2y - k**2 * I)) @ np.imag(C_k))
        
        C += A
        
    U[:,count+1] = U[:,count] + dt*( C - U[:,count] - nu*(((-D2y)**p) @ U[:,count])) # forward euler
    
    i += dt
    count += 1
    
    print('\n % {} complete '.format((i/T)*100))
    
    

fig, ((ax1), (ax2)) = plt.subplots(nrows=2, ncols =1 )

ax1.imshow(U,aspect = count/200)

ax2.plot(yy,U[:,-1],yy,U[:,0])

    
    



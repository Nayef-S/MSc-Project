#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#%%

import math
import numpy as np
from tqdm import tqdm
from numpy.matlib import repmat
from scipy import linalg
from Gamma_lyap import Gamma_U,Gamma_U_p
import matplotlib.pylab as plt

Nx = 64
Ny = 128
delta = 1.0
L = 2*math.pi

alp = 0.01
nu = 1e-5
beta = 4.5
p=2
dt = 40
T = 5000

dy = L*delta/Ny
yy = np.linspace(0,L*delta - dy,num=Ny)

dkx = 2*math.pi/L
kkx1 = np.arange(0,Nx/2+1)
kkx2 = np.arange(-Nx/2+1,0)
kkx = np.concatenate((kkx1, kkx2), axis=None)*dkx
dky = 2*math.pi/L/delta
kky1= np.arange(0,Ny/2+1)
kky2 = np.arange(-Ny/2+1,0)
kky = np.concatenate((kky1, kky2), axis=None)*dky
kx,ky = np.meshgrid(kkx,kky,indexing = 'xy')
k2 = kx**2 + ky**2

sig = 200

Corr = (np.logical_and(k2>=(10**2), k2<=(12**2)).astype(int))
Corr[:,0]  = 0
D = Corr / k2
D[0,0] = 0
Corr = Corr / (np.sum(D) * dkx*dky) / (2*(math.pi**2) * delta * L**2) *Nx*Ny



I = np.eye(Ny)

U = np.zeros((128,int(T/dt)+2))

U[:,0] = 0.02*np.sin(2*yy/delta) + 1e-3 * np.random.normal(0, 1, yy.shape[0])

count = 0
i = 0

# k  = 13

# Gamma_k  = Gamma_U(U[:,count],k,D2y,I)

# sigma_real = np.real(np.fft.ifft(sigma[:,k])).reshape(Ny,1)

# Chi_k = 2 * (sigma_real @ sigma_real.conj().T ) / sigma_real.shape[0]

# C_k = linalg.solve_continuous_lyapunov(Gamma_k, Chi_k)
    
# A = np.diag((-2*k*np.linalg.inv(D2y - k**2 * I)) @ np.imag(C_k))

m1 = np.exp((-nu*kky**(2*p) - alp)*dt)
m2 = (1.0 - np.exp((-nu*kky**(2*p) - alp)*dt)) / (nu*kky**(2*p) + alp)


#%%

while i <= T:

    C = np.zeros((Ny))
    
    Upp = np.real(np.fft.ifft((-kky**2) * np.fft.fft(U[:,count])))
    
    for k in range(1,13):
    
        sigma_k = sig  * np.fft.ifft(Corr[:,k])
        
        teosig = linalg.toeplitz(sigma_k)
        
        #Chi_k = 2*np.fft.ifft2(np.fft.fft2(teosig) * np.fft.fft2(teosig).conj().T) * 1/(Ny**2)#np.fft.ifft2(np.diag(2*sigma_real * sigma_real.conj().T)) #/ sigma_real.shape[0]**2
        
        Chi_k = 2 * teosig @ teosig  * 1/(Ny**2)
        
        Gamma_k  = Gamma_U_p(U[:,count],-Upp,k,p,kky)
        
        C_k = linalg.solve_continuous_lyapunov(Gamma_k, Chi_k)
        
        A = np.diag(-2*k*np.real(linalg.toeplitz(np.fft.ifft(1/(kky**2 + k**2)))) @ np.imag(C_k))
        
        C += alp*A/Nx
        
    #U[:,count+1] = U[:,count] + dt*( C - U[:,count] - (nu*( (-kky**2)**p) * U[:,count])) # forward euler 
    
    U[:,count+1] = np.real(np.fft.ifft(m1*np.fft.fft(U[:,count]) + m2 * np.fft.fft(C)))
    
    i += dt
    count += 1
    
    print('\n % {} complete '.format((i/T)*100))
    

fig, ((ax1), (ax2)) = plt.subplots(nrows=2, ncols =1 )

ax1.imshow(U,aspect = count/200)

ax2.plot(yy,U[:,-2],yy,U[:,0])

    
    



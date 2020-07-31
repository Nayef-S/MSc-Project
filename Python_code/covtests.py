#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 18:17:42 2020

@author: esuwws
"""

#%%

import math
import numpy as np
from tqdm import tqdm
from numpy.matlib import repmat
from scipy import linalg
import matplotlib.pylab as plt
from Gamma_lyap import Gamma_U,Gamma_U_p


Ny = 128
Nx = 64


sigma = np.load('sigma.npy')


                         
#%%
""" Covariance 1D  ( Ny = 1) """

realizations = 50000

sig = sigma[9,:]

sig2 = np.fft.ifft(sig)

coveta = np.zeros((Nx,realizations),dtype=np.complex)

eta_full = np.zeros((Nx,realizations),dtype=np.complex)


for i in range(0,realizations):

    eta_hat = np.fft.fft(np.random.randn(Nx))* sig * np.sqrt(1/Nx)   #np.sqrt(1/Nx) when fourier transform
     
    eta_full[:,i] = eta_hat
    
    coveta[:,i] = eta_hat *  eta_hat.conj().T


voc = np.cov(eta_full)

numcov = np.mean(coveta , axis = 1)

chihat = sig * sig.conj().T

#%%

""" Covariance 2D for one zonal wavenumber """

kx = 1

sig2 = sigma[:,kx]

chihat2 = np.diag(sig2 * sig2.conj().T)
 

coveta2 = np.zeros((Ny,Ny,10000),dtype=np.complex)


for i in range(0,10000):

    eta_hat2 = (np.random.default_rng().normal(0, 1, size=(Ny, Nx)) + 1j*np.random.default_rng().normal(0, 1, size=(Ny, Nx))) * sigma/np.sqrt(2)
    
    coveta2[:,:,i] = np.diag(eta_hat2[:,kx] * eta_hat2[:,kx].conj().T)

numcov2 = np.mean(coveta2 , axis = 2) 



#%%

A = np.zeros((Nx*Ny,100),dtype=np.complex)

for i in range(0,A.shape[1]):

    rndeta = (np.random.default_rng().normal(0, 1, size=(Ny, Nx)) + 1j*np.random.default_rng().normal(0, 1, size=(Ny, Nx))) * sigma/np.sqrt(2)
    
    veceta = rndeta.flatten('F')
    
    A[:,i] = veceta
    
    X = np.cov(A)
    
#%%
    
K = X.reshape((Ny,Nx,Ny,Nx))

k1 = K[1,1,:,:]

#Xtest = 0.5 * np.fft.ifft(sig) @ np.fft.ifft(sig).T  / np.fft.ifft(sig).shape[1]


#%%
L = 2*math.pi
alp = 0.01
nu = 1e-6
beta = 4.5
dt = 1e-2
T = 10

dy = L/Ny
yy = np.linspace(0,L-dy,num=Ny)

D2y = np.diag(np.linspace(1, 1, Ny-1), +1) - 2*np.diag(np.linspace(1, 1, Ny)) + np.diag(np.linspace(1, 1, Ny-1), -1)

#periodic BCs

D2y[0, Ny-1] = 1
D2y[-1, 0] = 1

D2y = D2y/(dy**2)
I = np.eye(Ny)

solnw = np.zeros((Ny,int(T/dt)+2))

#solnw[:,0] = 

w = np.zeros((Ny,Nx))

U = 0.2*np.sin(5*yy)

k = 1
 
Gamma_k  = Gamma_U_p(U,k,D2y,I,1)

eigen = min(np.real(linalg.eig(Gamma_k)[0]))

i = 0

t_tot = 0

while t_tot < T:
    
    eta_k = (np.random.default_rng().normal(0, 1, size=(Ny, Nx)) + 1j*np.random.default_rng().normal(0, 1, size=(Ny, Nx))) * sigma/np.sqrt(2)
    
    solnw[:,i+1] = solnw[:,i] + dt*(-Gamma_k @ solnw[:,i]) + np.fft.ifft(eta_k[:,k])
    
    i += 1
    
    t_tot += dt


omeg = np.zeros((Ny,Ny,solnw.shape[1]))

for j in range(int(solnw.shape[1]/2),solnw.shape[1]):

    omeg[:,:,j] = solnw[:,j].reshape(Ny,1) @ solnw[:,j].reshape(Ny,1).T  #np.fft.ifft2(np.diag(np.fft.fft(solnw[:,j]) * np.fft.fft(solnw[:,j]).conj().T))

TestChi = np.mean(omeg,axis  = 2)

# sigma_real = np.real(np.fft.ifft(sigma[:,k])).reshape(Ny,1)

# Chi_k = 2*sigma_real @ sigma_real.conj().T

sigma_k = sigma[:,k]

Chi_k = np.fft.ifft2(np.diag(2*sigma_k * sigma_k.conj().T))

C_k = linalg.solve_continuous_lyapunov(Gamma_k, Chi_k)


wt = np.fft.fft2(np.load('w.npy'))

test = np.fft.ifft(wt[:,k]).reshape(Ny,1) @ np.fft.ifft(wt[:,k]).reshape(Ny,1).conj().T


#%%
# # FT w

# ftw = np.fft.fft2(w)

# k0ftw = np.real(np.fft.ifft(ftw[:,0]).reshape(Ny,1))

# CovftW =  (k0ftw-np.mean(k0ftw)) @ (k0ftw-np.mean(k0ftw)).T / len(k0ftw)

# CorrftW = CovftW/(np.std(k0ftw)**2)

# # Real w

# k0w = w[:,0].reshape(Ny,1)

# CovW =  (k0w-np.mean(k0w)) @ (k0w-np.mean(k0w)).T / len(k0w)

# CorrW = CovW/(np.std(k0w)**2)


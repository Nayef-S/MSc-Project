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

teosig = linalg.toeplitz(np.fft.ifft(sig2))

chihat2 = np.fft.ifft2(np.fft.fft2(teosig) * np.fft.fft2(teosig).conj().T) * 1/(2*Ny*Nx)

#chihat2 = np.diag(sig2 * sig2.conj().T)
 

coveta2 = np.zeros((Ny,Ny,10000),dtype=np.complex)


for i in range(0,10000):

    eta_hat2 = (np.random.default_rng().normal(0, 1, size=(Ny, Nx)) + 1j*np.random.default_rng().normal(0, 1, size=(Ny, Nx))) * sigma/np.sqrt(2) 
    
    coveta2[:,:,i] = np.fft.ifft2(eta_hat2[:,kx].reshape(Ny,1) @ eta_hat2[:,kx].reshape(Ny,1).conj().T)

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
dt = 1e-3
T = 2000

dy = L/Ny
yy = np.linspace(0,L-dy,num=Ny)

D2y = np.diag(np.linspace(1, 1, Ny-1), +1) - 2*np.diag(np.linspace(1, 1, Ny)) + np.diag(np.linspace(1, 1, Ny-1), -1)

#periodic BCs

D2y[0, Ny-1] = 1
D2y[-1, 0] = 1

D2y = D2y/(dy**2)

I = np.eye(Ny)

solnw = np.zeros((Ny,int(T/dt)//1000),dtype=np.complex)

k = 1
w = np.zeros((Ny,Nx))
wk = w[:,k]

U = 0.2*np.sin(4*yy)

 
Gamma_k  = Gamma_U_p(U,k,D2y,I,1) 

ftGamma_k = np.fft.fft2(Gamma_k)

eigen = linalg.eig(Gamma_k)

i = 0

count = 0

t_tot = 0

while t_tot < T:
    
    eta_k = (np.random.default_rng().normal(0, 1, size=(Ny, Nx)) + 1j*np.random.default_rng().normal(0, 1, size=(Ny, Nx))) * sigma/np.sqrt(2)
    
    wk = wk + dt*(-Gamma_k @ wk) + np.real(np.fft.ifft(eta_k[:,k]))
    
    if (count % 1000) == 0 :
    
        solnw[:,i] = wk
        
        i += 1
    
    t_tot += dt
    
    count += 1
    
to = solnw.shape[1]
fro = solnw.shape[1] *  0 
stepp = 5

omeg = np.zeros((Ny,Ny,solnw.shape[1]+ 1),dtype=np.complex)
omeg2 = np.zeros((Ny,Ny,solnw.shape[1] + 1),dtype=np.complex)

ii = 0

for j in tqdm(range(0,solnw.shape[1])):

    omeg[:,:,ii] = solnw[:,j].reshape(Ny,1) @ solnw[:,j].reshape(Ny,1).conj().T #np.fft.ifft2(np.fft.fft(solnw[:,j]).reshape(Ny,1) @ np.fft.fft(solnw[:,j]).reshape(Ny,1).conj().T)#solnw[:,j].reshape(Ny,1) @ solnw[:,j].reshape(Ny,1).conj().T  #np.fft.ifft2(np.fft.fft(solnw[:,j]).reshape(Ny,1) @ np.fft.fft(solnw[:,j]).reshape(Ny,1).conj().T)
    
    #omeg2[:,:,ii] = np.fft.ifft2(np.fft.fft(solnw[:,j]).reshape(Ny,1) @ np.fft.fft(solnw[:,j]).reshape(Ny,1).conj().T)
    
    ii+=1


TestC_k = np.mean(omeg,axis  = 2)
TestC_k2 = np.mean(omeg2,axis  = 2) 

covfunc = np.cov(solnw[:,fro:-1],rowvar=1 )

# sigma_real = np.fft.ifft(sigma[:,k]).reshape(Ny,1)

# Chi_k = 2*sigma_real @ sigma_real.conj().T

#Chi_k = 2 * np.fft.ifft2(np.diag(sigma[:,k] * sigma[:,k].conj().T))

sigma_k = sigma[:,k]

teosig = linalg.toeplitz(np.fft.ifft(sigma_k))

Chi_k = 2 * np.fft.ifft2(np.fft.fft2(teosig) * np.fft.fft2(teosig).conj().T) 

Chi_k1 = 2 * teosig @ teosig * Ny

C_k = linalg.solve_continuous_lyapunov(Gamma_k, Chi_k)


wt = np.load('w.npy')

test = wt[:,k].reshape(Ny,1) @ wt[:,k].reshape(Ny,1).conj().T


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





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


Ny = 128
Nx = 64


w = np.load('w.npy')

wmean = np.mean(w)

w2mean = np.mean(w**2,axis = 0)

sigma = np.load('sigma.npy')


                         
#%%
""" 1D  ( Ny = 1) """

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

""" 2D for one zonal wavenumber """

kx = 1

sig2 = sigma[:,kx].reshape(Ny,1)

chihat2 = sig2 @ sig2.conj().T
 

coveta2 = np.zeros((Ny,Ny,10000),dtype=np.complex)


for i in range(0,10000):

    eta_hat2 = np.fft.fft2(np.random.randn(Ny, Nx)) * sig2 * 1/Nx
    
    coveta2[:,:,i] = eta_hat2[:,kx].reshape(Ny,1) @ eta_hat2[:,kx].reshape(Ny,1).conj().T

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




#  # FT w

# ftw = np.fft.fft2(w)

# k0ftw = np.real(np.fft.ifft(ftw[:,0]).reshape(Ny,1))

# CovftW =  (k0ftw-np.mean(k0ftw)) @ (k0ftw-np.mean(k0ftw)).T / len(k0ftw)

# CorrftW = CovftW/(np.std(k0ftw)**2)

# # Real w

# k0w = w[:,0].reshape(Ny,1)

# CovW =  (k0w-np.mean(k0w)) @ (k0w-np.mean(k0w)).T / len(k0w)

# CorrW = CovW/(np.std(k0w)**2)


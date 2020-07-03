#!/usr/bin/env python3
# -*- coding: utf-8 -*-


""" E check"""

import math
import numpy as np
from tqdm import tqdm
from numpy.matlib import repmat
from scipy import linalg



Nx = 64
Ny = 128
delta = 1.0
L = 2*math.pi

alp = 0.01
nu = 1e-6
beta = 4.5
p=2


dky = 2*math.pi/L/delta
kky1= np.arange(0,Ny/2+1)
kky2 = np.arange(-Ny/2+1,0)
kky = np.concatenate((kky1, kky2), axis=None)*dky
dy = L*delta/Ny
yy = np.linspace(0,L*delta - dy,num=Ny)

""" opening saved data """
sigma = np.load('sigma.npy')
U = np.load('U.npy')


""" Computing Gamma & C """

Ck = np.zeros((128,128),dtype=np.complex)

U_hat = np.fft.fft(U)


def Gamma_kyap(sigma,U_hat,kx):
    
    sigma_real = np.real(np.fft.ifft(sigma[:,kx]))
    
    Chi = 2*sigma_real.reshape(Ny,1) @ sigma_real.reshape(Ny,1).conj().T
     
    A = np.diag(1j * kx * U)
    
    invlap = 1/((kky**2) + (kx**2))
    
    invlap[invlap == np.inf ] = 0
    
    B = 1j*kx* np.fft.ifft2(np.diag(-kky**2 * U_hat - beta))
    
    C = np.fft.ifft2(np.diag(invlap))
    
    D = np.identity(Ny)*alp
    
    E = np.fft.ifft2(nu * (np.diag(kky**2) + np.identity(Ny)*(kx**2))**p)
    
    Gamma_k = A + B*C + D + E 
    
    Ck = linalg.solve_continuous_lyapunov(Gamma_k, Chi)
    
    return Ck

C_xpyp = []

"""Computing stationary energy"""

for kx in tqdm(range(0,Nx)):
    
    Ck = Gamma_kyap(sigma,U_hat,kx)

    diagC = [Ck[i][i] for i in range(min(len(Ck[0]),len(Ck)))] # diagonal of C
    
    C_xpyp.append(diagC)

E_st = 1 - (nu*0.5)*((np.mean(np.real(C_xpyp))*(L**2) *delta) + (L/alp)*(np.mean(np.real(np.fft.ifft(1j*kky*U_hat))**2) * delta)) #stationary energy

""" computing meta stable states - forward euler"""






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


""" opening saved data """
sigma = np.load('sigma.npy')
U = np.load('U.npy')


""" computing Gamma & C """
Ck = np.zeros((128,128),dtype=np.complex)
Chi = 2*sigma @ sigma.conj().T

for k in tqdm(range(1,Nx+1)):

    U_gamma = np.diag(U)
    
    U_ppgamma = np.diag(-kky**2 * U)
    
    A = np.fft.ifft2(1j * k * U_gamma)
    
    neglap = 1/(np.diag(kky**2) + np.identity(128)*(k**2))
    
    neglap[neglap == np.inf ] = 0
    
    B = np.fft.ifft2((1j*k*(np.identity(128)*beta + U_ppgamma)) * neglap)
    
    C = np.identity(128)*alp
    
    D = np.fft.ifft2(nu * (np.diag(kky**2) + np.identity(128)*(k**2))**p)
    
    Gamma_k = A + B + C + D
    
    Ck += linalg.solve_continuous_lyapunov(Gamma_k, Chi)



E_t0 = 1 - (nu/2)*((np.mean(np.real(Ck))*L**2 *delta) - (L/alp)*(np.mean(np.real((kky*np.fft.ifft(U)))**2) * delta))









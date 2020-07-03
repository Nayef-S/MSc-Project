#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import math
import numpy as np
from tqdm import tqdm
from numpy.matlib import repmat
from scipy import linalg
from Gamma_lyap import Gamma_lyap

Nx = 64
Ny = 128
delta = 1.0
L = 2*math.pi

alp = 0.01
nu = 1e-6
beta = 4.5
p=2
dt = 1e-3
T = 10

dy = L*delta/Ny
yy = np.linspace(0,L*delta - dy,num=Ny)
sigma = np.load('sigma.npy')

U = np.zeros((128,int(T/dt)))

U[:,0] = 0.2*np.sin(2*yy/delta)

A = np.zeros((Ny,1))

count = 0
i = 0

while i < T:

    for kx in range(0,int(Nx/2)):
    
        Ckx = Gamma_lyap(sigma,np.fft.fft(U[:,count]),kx)
        
        A = np.diag(-2*kx*(1/((yy**2) - (kx**2))) * np.imag(Ckx))
        
        A.setflags(write=1)
        
        A[np.isnan(A)] = 0
        
        A +=A
    
    U[:,count+1] = U[:,count] + dt*(A -  U[:,count]) # not sure about hyper-viscosity term
    
    i += dt
    count +=1
    
    print('\n % complete {}'.format((i/T)*100))
    
    
    
    
    
    
    

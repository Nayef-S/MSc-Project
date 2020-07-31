#!/usr/bin/env python3
# -*- coding: utf-8 -*-



import math
import numpy as np
from tqdm import tqdm
from scipy import linalg
from scipy import signal
import matplotlib.pylab as plt

Nx = 64
Ny = 128
delta = 1.0
L = 2*math.pi

alp = 0.01
nu = 1e-6
beta = 4.5
p=2


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

dy = L*delta/Ny


C = (np.logical_and(k2>=(10**2), k2<=(12**2)).astype(int))
C[:,0] = 0 
D = np.divide(C,k2)
D[0,0] = 0
C = (2* np.divide(C,np.mean(D)) * Nx *Ny )/(L**2)*delta
sigma = np.sqrt(C)

""" opening saved data """

sigma = np.load('sigma.npy')
U = np.load('U.npy').reshape(128,1)
UppQL = np.load('Upp.npy')
w = np.load('w.npy')


""" Computing Gamma & Ck """

C_k = np.zeros((128,128),dtype=np.complex)

# D^2 operator in y

D2y = np.diag(np.linspace(1, 1, Ny-1), +1) - 2*np.diag(np.linspace(1, 1, Ny)) + np.diag(np.linspace(1, 1, Ny-1), -1)

#periodic BCs

D2y[0, Ny-1] = 1
D2y[-1, 0] = 1

D2y = D2y/(dy**2)


Upp = D2y @ U # operator test

I = np.eye(Ny)

k = 1

def Gamma_U(U,k,D2y,I,p):
    
    lap = D2y - k**2 * I
    invlap = np.linalg.inv(lap)

    Gamma_k = 1j*k*U*I + 1j*k*((D2y @ U)*I - beta*I) @ invlap + alp*I + nu*(-D2y + k**2 * I)**p

    return Gamma_k



def Gamma_U_p(U,k,D2y,I,p):
    
    Ny = 128
    dky = 2*math.pi/(2*math.pi)
    kky1= np.arange(0,Ny/2+1)
    kky2 = np.arange(-Ny/2+1,0)
    kky = np.concatenate((kky1, kky2), axis=None)*dky
    
    lap = D2y - k**2 * I
    invlap = np.linalg.inv(lap)

    Gamma_k = 1j*k*U*I + 1j*k*((D2y @ U)*I - beta*I) @ invlap + alp*I + np.fft.ifft(nu*(-kky**(2*p) * I + -k**(2*p) *I) ,axis = 0)

    return Gamma_k



# def conv2(x, y, mode='full'):
#     return np.rot90(signal.convolve2d(np.rot90(x, 2), np.rot90(y, 2), mode=mode), 2)

if __name__ == "__main__":
    
    test = Gamma_U(U,1,D2y,I,1)
    
    test2 = Gamma_U_p(U,1,D2y,I,1)
    
    
    
    # for k in range(-int(Nx/2),int(Nx/2)):

    #     sigma_real = np.real(np.fft.ifft(sigma[:,k]))
        
    #     Chi_k = 2*sigma_real.reshape(Ny,1) @ sigma_real.reshape(Ny,1).conj().T
        
    #     Gamma_k  = Gamma_U(U,k,D2y,I)
        
    #     C_k = linalg.solve_continuous_lyapunov(Gamma_k, Chi_k)
    
    
    # test = np.real(C_k)
    # test2 = w**2

    # C_xpyp = np.zeros((Ny,Nx))
    
    # """Computing stationary energy"""
    
    # for kx in tqdm(range(-int(Nx/2),int(Nx/2))):
        
    #     Ck = Gamma_lyap(sigma,U_hat,kx)
        
    #     C_xpyp[:,kx] = np.fft.fft(Ck.diagonal()) # function of ky,ky'
    
    # E_st = 1 - (nu*0.5)*((np.mean(np.fft.ifft2(C_xpyp))*(L**2) *delta) + (L/alp)*(np.mean(np.real(np.fft.ifft(1j*kky*U_hat))**2) * delta)) #stationary energy
        
    
    # test = Gamma_lyap(sigma,U_hat,5)
    
    # Ck, Chi, E = Gamma_lyap(sigma,U_hat,1)
    
    # chik = np.zeros((Ny,Ny))
    
    # k = 1
            
    # for q in range(0,Nx):
        
    #     k_q = abs(1-q)
        
    #     chik = sigma[:,q].reshape(Ny,1) @ sigma[:,k_q].reshape(Ny,1).conj().T
        
    #     chik += chik
        
    # sigma0 = sigma[:,0]

    # testchi = conv2(sigma,sigma0[:,None].conj().T)
    
    # Cx = np.zeros((Ny,Nx))
    
    kkx2 = np.concatenate((kkx2, kkx1), axis=None)*dkx
    kky2 = np.concatenate((kky2, kky1), axis=None)*dky
    kx2,ky2 = np.meshgrid(kkx2,kky2,indexing = 'xy')
    
    Cx = np.fft.fftshift(C)
    
    plt.contourf(kx2,ky2,Cx)
    
    
    
    
    
    
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np
#numerical parameters
steps = 5e6
Nx = 64
Ny = 128
dt = 1e-3
E = []
Ec = []
Umodes = []
betas = []

#model parameters
L = 2*math.pi
alp = 0.01
nu = 1e-6
beta = 4.5
p=2
delta = 1.0

dx = L/Nx
xx = np.linspace(0, L-dx, num=Nx)
dy = L*delta/Ny
yy = np.linspace(0,L*delta - dy,num=Ny)
x, y = np.meshgrid(xx, yy, indexing='xy')

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
kp = k2**2
modk = np.sqrt(k2)
kmax = np.amax(modk)

U = 0.2*np.fft.fft(np.sin(2*yy/delta))
w = np.zeros((Ny,Nx))

M1 = np.exp((-nu*kp - alp)*dt)
M2 = (1.0 - M1)/(nu*kp + alp)
m1 = np.exp((-nu*kky**(2*p) - alp)*dt)
m2 = (1.0 - m1)/(nu*kky**(2*p) + alp)

if alp ==0 :
    if nu ==0:
        M2 = 0*M1 + dt
        m2 = 0*m1 + dt
    M2[0] = dt
    m2[0] = dt



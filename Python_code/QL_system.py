#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np
from tqdm import tqdm
from numpy.matlib import repmat
import matplotlib.pylab as plt
import matplotlib.animation as animation
from scipy import linalg
# import pyfftw
#numerical parameters
steps = int(1e6)
Nx = 64
Ny = 128
dt = 1e-3
E = []
E_c = []
betas = []

#model parameters
L = 2*math.pi
alp = 0.01
nu = 1e-6
beta = 4.5
p = 2
delta = 1.0
sig = 200

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
kp = k2**p
modk = np.sqrt(k2)
kmax = np.amax(modk)

U = 0.02*np.fft.fft(np.sin(2*yy/delta)) 
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

C = (np.logical_and(k2>=(10**2), k2<=(12**2)).astype(int))
C[:,0]  = 0
D = C / k2
D[0,0] = 0
C = C / (np.sum(D) * dkx*dky) / (2*(math.pi**2) * delta * L**2) *Nx*Ny
sigma = sig * C


v =  -1j*(kx/k2)*np.fft.fft2(w) 
v[0,0] = 0

ims = []
hovmoller = np.empty((128,steps))
Umodes = np.empty((5,steps))
counter = 0

fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(nrows=2, ncols = 3)
fig = plt.gcf()
fig.set_size_inches(25, 12)

for step in tqdm(range(0,steps)):
    
    U_bar =  np.real_if_close(repmat(np.fft.ifft(U),Nx,1).T) # check
    
    upp =  np.real(np.fft.ifft((-kky**2)*(U)))
    
    Upp_bar = repmat(upp,Nx,1).T
    
    wrest = -U_bar*np.real(np.fft.ifft2(1j*kx*np.fft.fft2(w))) - (beta - Upp_bar)*np.real(np.fft.ifft2(v))
    
    w = np.real(np.fft.ifft2(M1*np.fft.fft2(w) + M2*np.fft.fft2(wrest)))
    
    eta = (np.random.default_rng().normal(0, 1, size=(Ny, Nx)) + 1j*np.random.default_rng().normal(0, 1, size=(Ny, Nx))) * sigma/np.sqrt(2)
    
    w += np.sqrt(dt) * np.real(np.fft.ifft2(eta))
    
    v =  -1j*(kx/k2)*np.fft.fft2(w) 
    v[0,0] = 0
    
    
    avgvw = alp*np.mean(np.real(np.fft.ifft2(v))*w,axis = 1).T * L
    
    U = m1*U + np.multiply(m2,np.fft.fft(avgvw))
    U[0] = 0
    U[5:] = 0
    
    #de alaising
    U = (abs(kky)<=(2/3)*max(kky)).astype(int) * U
    w = np.fft.fft2(w)
    w = (modk<=(2/3)*kmax).astype(int) * w
    w = np.real(np.fft.ifft2(w))
    
    
    if (step % 1000) == 0 :
        a = max(U[1:6])
        b = np.argmax(U[1:6])
        highestmode = b
        print('\n Step = {} , dominating mode = {} '.format(step,highestmode))

    u = 1j *(ky/k2)*np.fft.fft2(w)
    u[0,0] = 0
    
    E_u = L*0.5 * np.mean(np.real(np.fft.ifft(U))**2) * delta
    E_w = alp*0.5 * np.real(np.mean(np.fft.ifft2(v)**2 + np.fft.ifft2(u)**2)) * (L**2) * delta
    E_now = E_u + E_w
    E_c.append(E_u)
    E.append(E_now)
    
    # hovmoller stream
    psi = U/(1j *kky)
    psi[0] = 0
    hovmoller[:,step] = np.real(np.fft.ifft(psi))

    # modes
    Umodes[:,step] = abs(U[1:6])
    
    
    
""" plots """


#Energy
ax1.plot(E,label = 'E')
ax1.plot(E_c,label = 'E_{U}')
ax1.legend()
ax1.set_xlabel('step')
ax1.set_ylabel('Energy')
        
    
#avg vw
ax2.plot(yy,avgvw)
ax2.set_xlabel('y')
ax2.set_ylabel('mean(uv)')

# omega

ws = ax3.contourf(xx,yy,w)
ax3.set_xlabel('y')
ax3.set_ylabel('y')

# U and dU/dy^2
ax4.plot(yy,np.real(np.fft.ifft(U)),yy , np.real(np.fft.ifft(-kky**2 * U)))
ax4.set_xlabel('y')
ax4.set_ylabel("U and U''")

cs = ax5.imshow(hovmoller, aspect = steps/200)

# modes
ax6.plot(Umodes.T)
ax6.legend(['k_y=1', 'k_y=2', 'k_y=3', 'k_y=4', 'k_y=5'])


  
fig.colorbar(ws,ax=ax3)   
fig.colorbar(cs,ax=ax5)

plt.savefig("soln.png", dpi=150)

""" saving data """

# np.save('Hovmoller.npy', hovmoller) 
# np.save('Umodes.npy', Umodes) 
# np.save('U.npy', np.real(np.fft.ifft(U))) 
# np.save('Upp.npy', np.real(np.fft.ifft(-kky**2 * U)))
# np.save('w.npy', w) 
# np.save('sigma.npy', sigma) 

# ani = animation.ArtistAnimation(fig, ims, interval=10, blit=True)

# ani.save('hovmoll.gif', writer='imagemagick')












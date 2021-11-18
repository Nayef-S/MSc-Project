#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 14:02:14 2021

@author: nshkeir

"""
import numpy as np
from scipy.linalg import expm
from numpy.linalg import matrix_power

def ETD1(L,N,dt):
    
    m1 = expm(dt*L) 
    
    m2_b =  0.5 * (m1 @ m1 - np.identity(L.shape[0]))
                 
    m2 = np.linalg.solve(L, m2_b)
    
        
    return m1, m2

def ETD2(L,N,dt):
    
    m1 = expm(dt*L) 
    
    m2 = 0.25* dt* matrix_power(L,-2)@ ((2*L*dt + np.identity(L.shape[0])@(m1@m1))  - 4*L*dt- np.identity(L.shape[0]))
    
    m3 = 0.25* dt* matrix_power(L,-2)@ (np.identity(L.shape[0]) + 2*L*dt - m1@m1)
    
    return m1 , m2 , m3

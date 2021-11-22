#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 14:02:14 2021

@author: nshkeir

"""
import numpy as np
from scipy.linalg import expm
from numpy.linalg import matrix_power
import mpmath as mp




def ETD1(L,R,N,dt):
    """
    Returns the operators for matrix ETD1

    Parameters
    ----------
    L : (M, M) array_like
        LHS operator acting on Q
        
    R : (M, M) array_like
       RHS operator acting on Q
       
    N : (M, N) array_like
        Non-linear operator acting on Q
        
    dt : float
        Time step 

    Returns
    -------
    m1_L : (M, M) array_like
        M1[L] acting on LHS of Q_n 
    m1_R : (M, M) array_like
        M1[R] acting on RHS of Q_n 
    m2 : TYPE
        M2[L+R] acting on the non-linear operator 

    """
    
    m1_L = expm(dt*L) 
    m1_R = expm(dt*R) 
    
    m2_b =  (m1_L @ m1_R - np.identity(L.shape[0]))
                 
    m2 = np.linalg.solve((L+R), m2_b)
    
    return m1_L, m1_R, m2

def ETD1_eps(L,R,N,dt,eps):
    """
    Returns the operators for matrix ETD1

    Parameters
    ----------
    L : (M, M) array_like
        LHS operator acting on Q
        
    R : (M, M) array_like
       RHS operator acting on Q
       
    N : (M, N) array_like
        Non-linear operator acting on Q
        
    dt : float
        Time step 

    Returns
    -------
    m1_L : (M, M) array_like
        M1[L] acting on LHS of Q_n 
    m1_R : (M, M) array_like
        M1[R] acting on RHS of Q_n 
    m2 : TYPE
        M2[L+R] acting on the non-linear operator 

    """
    mp.dps = eps
    
    L_dps = mp.matrix(L)
    R_dps = mp.matrix(R)
    
    m1_L = mp.expm(dt*L_dps) 
    m1_R = mp.expm(dt*R_dps) 
    
    m2_b =  (m1_L * m1_R - mp.eye(L.shape[0]))
                 
    m2 = (L_dps+R_dps)**-1 * m2_b
    
    return m1_L, m1_R, m2

def ETD2(L,R,N,dt):
    """
    

    Parameters
    ----------
    L : M, M) array_like
        LHS operator acting on Q
        
    R : (M, M) array_like
       RHS operator acting on Q
       
    N : (M, N) array_like
        Non-linear operator acting on Q
        
    dt : float
        Time step

    Returns
    -------
    m1_L : (M, M) array_like
        M1[L] acting on LHS of Q_n
    m1_R : (M, M) array_like
        M1[R] acting on RHS of Q_n
    m2 : TYPE
        DESCRIPTION.
    m3 : TYPE
        DESCRIPTION.

    """

    m1_L = expm(dt*L) 
    m1_R = expm(dt*R) 
    
    m_lr = m1_L @ m1_R
    
    m2 = (1/dt)* matrix_power((L+R),-2) @ (  (dt*(L+R)+np.identity(L.shape[0]) ) @ (m_lr) - 2*dt*(L+R) -   np.identity(L.shape[0]) )  
    
    m3 = (1/dt)* matrix_power((L+R),-2) @ ( np.identity(L.shape[0]) + dt*(L+R)  -    (m_lr)   )
    
    return m1_L , m1_R , m2 , m3  









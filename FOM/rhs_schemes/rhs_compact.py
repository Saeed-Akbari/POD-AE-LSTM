#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 21:49:29 2021

@author: suraj
"""

import numpy as np
from .thomas_algorithms import *
import matplotlib.pyplot as plt
from .compact_schemes_first_order_derivative import *
from .compact_schemes_second_order_derivative import *
from .rhs_conservative import *

def rhs_compact_scheme(nx,ny,dx,dy,re,pr,w,s,th,icompact):
    
    # viscous terms for vorticity transport equation
    
    uw = np.copy(w)
    us = np.copy(s)
    uth = np.copy(th)
    
    if icompact == 1:
        # convective terms streamfunciton
        sx = c4d_p(us,dx,dy,nx,ny,'X') # sx 
        sy = c4d(us,dx,dy,nx,ny,'Y') # sy    
        
        # convective terms vorticity
        wx = c4d_p(uw,dx,dy,nx,ny,'X') # wx
        wy = c4d(uw,dx,dy,nx,ny,'Y') # wy
        
        # convective terms temperature
        thx = c4d_p(uth,dx,dy,nx,ny,'X') # wx
        thy = c4d(uth,dx,dy,nx,ny,'Y') # wy
        
        # dissipative terms vorticity    
        wxx = c4dd_p(uw,dx,dy,nx,ny,'XX') # wxx
        wyy = c4dd(uw,dx,dy,nx,ny,'YY') # wyy
        
        # dissipative terms temperature    
        thxx = c4dd_p(uth,dx,dy,nx,ny,'XX') # theta_xx
        thyy = c4dd(uth,dx,dy,nx,ny,'YY') # theta_yy
    
    if icompact == 2:
        # convective terms streamfunciton
        sx = c6d_p(us,dx,dy,nx,ny,'X') # sx 
        sy = c6d_b5_d(us,dx,dy,nx,ny,'Y') # sy    
        
        # convective terms vorticity
        wx = c6d_p(uw,dx,dy,nx,ny,'X') # wx
        wy = c6d_b5_d(uw,dx,dy,nx,ny,'Y') # wy
        
        # convective terms temperature
        thx = c6d_p(uth,dx,dy,nx,ny,'X') # wx
        thy = c6d_b5_d(uth,dx,dy,nx,ny,'Y') # wy
        
        # dissipative terms vorticity    
        wxx = c6dd_p(uw,dx,dy,nx,ny,'XX') # wxx
        wyy = c6dd_b5_d(uw,dx,dy,nx,ny,'YY') # wyy
        
        # dissipative terms temperature    
        thxx = c6dd_p(uth,dx,dy,nx,ny,'XX') # theta_xx
        thyy = c6dd_b5_d(uth,dx,dy,nx,ny,'YY') # theta_yy
    
    fw = np.zeros((nx+1,ny+1))
    fth = np.zeros((nx+1,ny+1))
    
    jac = sy*wx - sx*wy
    fw[:,:] = -jac + (wxx + wyy)/re + thx
    
    jac = sy*thx - sx*thy
    fth[:,:] = -jac + (thxx + thyy)/(re*pr) 
    
    return fw, fth
    
    
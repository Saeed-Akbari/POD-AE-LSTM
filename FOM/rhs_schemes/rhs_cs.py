#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 18:19:04 2021

@author: suraj
"""
import nunpy as np

def rhs_cs(nx,ny,dx,dy,re,w,s):
    aa = 1.0/(dx*dx)
    bb = 1.0/(dy*dy)
    gg = 1.0/(4.0*dx*dy)
    hh = 1.0/3.0
    dd = 1.0/(2.0*dx)
    
    f = np.zeros((nx+1,ny+1))
    
    ii = np.arange(1,nx)
    jj = np.arange(1,ny)
    i,j = np.meshgrid(ii,jj, indexing='ij')
   
    j1 = gg*((w[i+1,j]-w[i-1,j])*(s[i,j+1]-s[i,j-1]) - \
                 (w[i,j+1]-w[i,j-1])*(s[i+1,j]-s[i-1,j]))
    
    jac = j1
        
    lap = aa*(w[i+1,j]-2.0*w[i,j]+w[i-1,j]) + bb*(w[i,j+1]-2.0*w[i,j]+w[i,j-1])
                                
    f[i,j] = -jac + lap/re 
        
    return f


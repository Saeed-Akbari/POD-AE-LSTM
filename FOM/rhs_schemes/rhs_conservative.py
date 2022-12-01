#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 18:19:04 2021

@author: suraj
"""
import numpy as np


def jacobian_second_order(nx,ny,dx,dy,re,w,s,i,j):
    gg = 1.0/(4.0*dx*dy)
        
    jac = gg*((w[i+1,j]-w[i-1,j])*(s[i,j+1]-s[i,j-1]) - \
                 (w[i,j+1]-w[i,j-1])*(s[i+1,j]-s[i-1,j]))
        
    return jac

def jacobian_fourth_order(nx,ny,dx,dy,re,w,s,i,j):
    gg = 1.0/(4.0*dx*dy)
    hh = 1.0/3.0
      
    j1 = gg*((w[i+1,j]-w[i-1,j])*(s[i,j+1]-s[i,j-1]) - \
                 (w[i,j+1]-w[i,j-1])*(s[i+1,j]-s[i-1,j]))

    j2 = gg*(w[i+1,j]*(s[i+1,j+1]-s[i+1,j-1]) - \
             w[i-1,j]*(s[i-1,j+1]-s[i-1,j-1]) - \
             w[i,j+1]*(s[i+1,j+1]-s[i-1,j+1]) + \
             w[i,j-1]*(s[i+1,j-1]-s[i-1,j-1]))

    j3 = gg*(w[i+1,j+1]*(s[i,j+1]-s[i+1,j]) - \
             w[i-1,j-1]*(s[i-1,j]-s[i,j-1]) - \
        	 w[i-1,j+1]*(s[i,j+1]-s[i-1,j]) + \
        	 w[i+1,j-1]*(s[i+1,j]-s[i,j-1]))

    jac = (j1+j2+j3)*hh
        
    return jac    

def laplacian_second_order(nx,ny,dx,dy,re,w,i,j):
    aa = 1.0/(dx*dx)
    bb = 1.0/(dy*dy)
    
    lap = aa*(w[i+1,j]-2.0*w[i,j]+w[i-1,j]) + bb*(w[i,j+1]-2.0*w[i,j]+w[i,j-1])
    
    return lap
    
#%% 
# compute rhs using arakawa scheme
# computed at all physical domain points (1:nx+1,1:ny+1; all boundary points included)
# no ghost points
def rhs_arakawa(nx,ny,dx,dy,re,pr,w,s,th):
    aa = 1.0/(dx*dx)
    bb = 1.0/(dy*dy)
    gg = 1.0/(4.0*dx*dy)
    hh = 1.0/3.0
    dd = 1.0/(2.0*dx)
    
    fw = np.zeros((nx+1,ny+1))
    fth = np.zeros((nx+1,ny+1))
    
    
    ii = np.arange(0,nx)
    jj = np.arange(1,ny)
    i,j = np.meshgrid(ii,jj, indexing='ij')
    
    wp = w[ii,:]
    sp = s[ii,:]
    thp = th[ii,:]
    
    j1 = gg*((wp[(i+1)%nx,j] - wp[i-1,j])*(sp[i,j+1] - sp[i,j-1]) - \
             (wp[i,j+1] - wp[i,j-1])*(sp[(i+1)%nx,j] - sp[i-1,j]))

    j2 = gg*(wp[(i+1)%nx,j]*(sp[(i+1)%nx,j+1] - sp[(i+1)%nx,j-1]) - \
             wp[i-1,j]*(sp[i-1,j+1] - sp[i-1,j-1]) - \
             wp[i,j+1]*(sp[(i+1)%nx,j+1] - sp[i-1,j+1]) + \
             wp[i,j-1]*(sp[(i+1)%nx,j-1] - sp[i-1,j-1]))

    j3 = gg*(wp[(i+1)%nx,j+1]*(sp[i,j+1] - sp[(i+1)%nx,j]) - \
             wp[i-1,j-1]*(sp[i-1,j] - sp[i,j-1]) - \
        	 wp[i-1,j+1]*(sp[i,j+1] - sp[i-1,j]) + \
        	 wp[(i+1)%nx,j-1]*(sp[(i+1)%nx,j] - sp[i,j-1]))

    jac = (j1+j2+j3)*hh
        
    lap = aa*(wp[(i+1)%nx,j] - 2.0*wp[i,j] + wp[i-1,j]) + bb*(wp[i,j+1] - 2.0*wp[i,j] + wp[i,j-1])
    
    source = (thp[(i+1)%nx,j] - thp[i-1,j])/(2.0*dx)
#    source = (thp[i,j+1] - thp[i,j-1])/(2.0*dy)
                                
    fw[i,j] = -jac + lap/re + source
    
    j1 = gg*((thp[(i+1)%nx,j] - thp[i-1,j])*(sp[i,j+1] - sp[i,j-1]) - \
             (thp[i,j+1] - thp[i,j-1])*(sp[(i+1)%nx,j] - sp[i-1,j]))

    j2 = gg*(thp[(i+1)%nx,j]*(sp[(i+1)%nx,j+1] - sp[(i+1)%nx,j-1]) - \
             thp[i-1,j]*(sp[i-1,j+1] - sp[i-1,j-1]) - \
             thp[i,j+1]*(sp[(i+1)%nx,j+1] - sp[i-1,j+1]) + \
             thp[i,j-1]*(sp[(i+1)%nx,j-1] - sp[i-1,j-1]))

    j3 = gg*(thp[(i+1)%nx,j+1]*(sp[i,j+1] - sp[(i+1)%nx,j]) - \
             thp[i-1,j-1]*(sp[i-1,j] - sp[i,j-1]) - \
        	 thp[i-1,j+1]*(sp[i,j+1] - sp[i-1,j]) + \
        	 thp[(i+1)%nx,j-1]*(sp[(i+1)%nx,j] - sp[i,j-1]))

    jac = (j1+j2+j3)*hh
        
    lap = aa*(thp[(i+1)%nx,j] - 2.0*thp[i,j] + thp[i-1,j]) + bb*(thp[i,j+1] - 2.0*thp[i,j] + thp[i,j-1])
                                   
    fth[i,j] = -jac + lap/(re*pr)
        
    return fw, fth

def rhs_cs(nx,ny,dx,dy,re,w,s):
    
    f = np.zeros((nx+1,ny+1))
    
    ii = np.arange(1,nx)
    jj = np.arange(1,ny)
    i,j = np.meshgrid(ii,jj, indexing='ij')
   
    jac = jacobian_second_order(nx,ny,dx,dy,re,w,s,i,j)
    lap = laplacian_second_order(nx,ny,dx,dy,re,w,i,j)    
                            
    f[i,j] = -jac + lap/re 
        
    return f

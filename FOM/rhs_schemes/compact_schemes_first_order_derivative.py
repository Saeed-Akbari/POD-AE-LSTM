#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 18:52:37 2021

@author: suraj
"""

import numpy as np
from .thomas_algorithms import *
import matplotlib.pyplot as plt
from scipy.sparse import csc_matrix
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import inv
from scipy.sparse.linalg import spsolve

def c4d(f,dx,dy,nx,ny,isign):
    
    if isign == 'X':
        u = np.copy(f)
        h = dx
    if isign == 'Y':
        u = np.copy(f.T)
        h = dy
        temp = nx
        nx = ny
        ny = temp
        
    a = np.zeros((nx+1,ny+1))
    b = np.zeros((nx+1,ny+1))
    c = np.zeros((nx+1,ny+1))
    r = np.zeros((nx+1,ny+1))
    
    i = 0
    a[i,:] = 0.0
    b[i,:] = 1.0
    c[i,:] = 2.0
    r[i,:] = (-5.0*u[i,:] + 4.0*u[i+1,:] + u[i+2,:])/(2.0*h)
    
    ii = np.arange(1,nx)
    
    a[ii,:] = 1.0/4.0
    b[ii,:] = 1.0
    c[ii,:] = 1.0/4.0
    r[ii,:] = 3.0*(u[ii+1,:] - u[ii-1,:])/(4.0*h)
    
    i = nx
    a[i,:] = 2.0
    b[i,:] = 1.0
    c[i,:] = 0.0
    r[i,:] = -1.0*(-5.0*u[i,:] + 4.0*u[i-1,:] + u[i-2,:])/(2.0*h)
    
    start = 0
    end = nx
    
#    A = csc_matrix((nx+1, nx+1))
#    ii = np.arange(1,nx)
#    A[ii,ii] = 1.0
#    A[ii,ii-1] = 1.0/4.0
#    A[ii,ii+1] = 1.0/4.0
#    
#    ii = 0
#    A[ii,ii] = 1.0
#    A[ii,ii+1] = 2.0
#    
#    ii = nx
#    A[ii,ii] = 1.0
#    A[ii,ii-1] = 2.0
#    
#    ud = spsolve(A, r) #inv(A) @ r
    
    ud = tdma(a,b,c,r,start,end)
    
    if isign == 'X':
        fd = np.copy(ud)
    if isign == 'Y':
        fd = np.copy(ud.T)
        
    return fd

def c4d_p(f,dx,dy,nx,ny,isign):
    
    if isign == 'X':
        u = np.copy(f)
        h = dx
    if isign == 'Y':
        u = np.copy(f.T)
        h = dy
        temp = nx
        nx = ny
        ny = temp
    
    a = np.zeros((nx,ny+1))
    b = np.zeros((nx,ny+1))
    c = np.zeros((nx,ny+1))
    r = np.zeros((nx,ny+1))

    ii = np.arange(0,nx)
    up = u[ii,:]
    a[ii,:] = 1.0/4.0
    b[ii,:] = 1.0
    c[ii,:] = 1.0/4.0
    r[ii,:] = 3.0*(up[(ii+1)%nx,:] - up[ii-1,:])/(4.0*h)
    
    start = 0
    end = nx
        
    alpha = np.zeros((1,ny+1))
    beta = np.zeros((1,ny+1))
    
    alpha[0,:] = 1.0/4.0
    beta[0,:] = 1.0/4.0
    
    x = ctdmsv(a,b,c,alpha,beta,r,0,nx-1,ny)
    
    ud = np.zeros((nx+1,ny+1))
    ud[0:nx,:] = x[0:nx,:]
    ud[nx,:] = ud[0,:]

    if isign == 'X':
        fd = np.copy(ud)
    if isign == 'Y':
        fd = np.copy(ud.T)
        
    return fd

def c4d_b4(f,dx,dy,nx,ny,isign):
    
    if isign == 'X':
        u = np.copy(f)
        h = dx
    if isign == 'Y':
        u = np.copy(f.T)
        h = dy
        temp = nx
        nx = ny
        ny = temp
    
    a = np.zeros((nx+1,ny+1))
    b = np.zeros((nx+1,ny+1))
    c = np.zeros((nx+1,ny+1))
    r = np.zeros((nx+1,ny+1))
    
    i = 0
    a[i,:] = 0.0
    b[i,:] = 1.0
    c[i,:] = 3.0
    r[i,:] = (-17.0*u[i,:] + 9.0*u[i+1,:] + 9.0*u[i+2,:] - u[i+3,:])/(6.0*h)
    
    ii = np.arange(1,nx)
    
    a[ii,:] = 1.0/4.0
    b[ii,:] = 1.0
    c[ii,:] = 1.0/4.0
    r[ii,:] = 3.0*(u[ii+1,:] - u[ii-1,:])/(4.0*h)
    
    i = nx
    a[i,:] = 3.0
    b[i,:] = 1.0
    c[i,:] = 0.0
    r[i,:] = -1.0*(-17.0*u[i,:] + 9.0*u[i-1,:] + 9.0*u[i-2,:] - u[i-3,:])/(6.0*h)
    
    start = 0
    end = nx
    ud = tdma(a,b,c,r,start,end)
    
    if isign == 'X':
        fd = np.copy(ud)
    if isign == 'Y':
        fd = np.copy(ud.T)
        
    return fd

def c6d_p(f,dx,dy,nx,ny,isign):
    
    if isign == 'X':
        u = np.copy(f)
        h = dx
    if isign == 'Y':
        u = np.copy(f.T)
        h = dy
        temp = nx
        nx = ny
        ny = temp
    
    a = np.zeros((nx,ny+1))
    b = np.zeros((nx,ny+1))
    c = np.zeros((nx,ny+1))
    r = np.zeros((nx,ny+1))

    ii = np.arange(0,nx)
    up = u[ii,:]
    a[ii,:] = 1.0/3.0
    b[ii,:] = 1.0
    c[ii,:] = 1.0/3.0
    r[ii,:] = (1.0/9.0)*(up[(ii+2)%nx,:] - up[ii-2,:])/(4.0*h) + (14.0/9.0)*(up[(ii+1)%nx,:] - up[ii-1,:])/(2.0*h)
    
    start = 0
    end = nx
        
    alpha = np.zeros((1,ny+1))
    beta = np.zeros((1,ny+1))
    
    alpha[0,:] = 1.0/3.0
    beta[0,:] = 1.0/3.0
    
    x = ctdmsv(a,b,c,alpha,beta,r,0,nx-1,ny)
    
    ud = np.zeros((nx+1,ny+1))
    ud[0:nx,:] = x[0:nx,:]
    ud[nx,:] = ud[0,:]
    
    if isign == 'X':
        fd = np.copy(ud)
    if isign == 'Y':
        fd = np.copy(ud.T)
        
    return fd

def c6d_b3_d(f,dx,dy,nx,ny,isign):
    
    if isign == 'X':
        u = np.copy(f)
        h = dx
    if isign == 'Y':
        u = np.copy(f.T)
        h = dy
        temp = nx
        nx = ny
        ny = temp
    
    a = np.zeros((nx+1,ny+1))
    b = np.zeros((nx+1,ny+1))
    c = np.zeros((nx+1,ny+1))
    r = np.zeros((nx+1,ny+1))
    
    i = 0
    a[i,:] = 0.0
    b[i,:] = 1.0
    c[i,:] = 2.0
    r[i,:] = (-5.0*u[i,:] + 4.0*u[i+1,:] + u[i+2,:])/(2.0*h)
    
    i = 1
    a[i,:] = 0.0
    b[i,:] = 1.0
    c[i,:] = 2.0
    r[i,:] = (-5.0*u[i,:] + 4.0*u[i+1,:] + u[i+2,:])/(2.0*h)
       
    
    ii = np.arange(2,nx-1)
    a[ii,:] = 1.0/3.0
    b[ii,:] = 1.0
    c[ii,:] = 1.0/3.0
    r[ii,:] = (1.0/9.0)*(u[(ii+2),:] - u[ii-2,:])/(4.0*h) + (14.0/9.0)*(u[(ii+1),:] - u[ii-1,:])/(2.0*h)
           
    i = nx-1
    a[i,:] = 2.0
    b[i,:] = 1.0
    c[i,:] = 0.0
    r[i,:] = -1.0*(-5.0*u[i,:] + 4.0*u[i-1,:] + u[i-2,:])/(2.0*h)
    
    i = nx
    a[i,:] = 2.0
    b[i,:] = 1.0
    c[i,:] = 0.0
    r[i,:] = -1.0*(-5.0*u[i,:] + 4.0*u[i-1,:] + u[i-2,:])/(2.0*h)
    
    start = 0
    end = nx

    ud = tdma(a,b,c,r,start,end)
    
    if isign == 'X':
        fd = np.copy(ud)
    if isign == 'Y':
        fd = np.copy(ud.T)
        
    return fd


def c6d_b5_d(f,dx,dy,nx,ny,isign):
    
    if isign == 'X':
        u = np.copy(f)
        h = dx
    if isign == 'Y':
        u = np.copy(f.T)
        h = dy
        temp = nx
        nx = ny
        ny = temp
    
    a = np.zeros((nx+1,ny+1))
    b = np.zeros((nx+1,ny+1))
    c = np.zeros((nx+1,ny+1))
    r = np.zeros((nx+1,ny+1))
    
    # 5th order
    i = 0
    a[i,:] = 0.0
    b[i,:] = 1.0
    c[i,:] = 4.0
    r[i,:] = (-37.0*u[i,:] + 8.0*u[i+1,:] + 36.0*u[i+2,:] - 8.0*u[i+3,:] + 1.0*u[i+4,:])/(12.0*h)

    # 5th order
    i = 1
    a[i,:] = 0.0
    b[i,:] = 1.0
    c[i,:] = 4.0
    r[i,:] = (-37.0*u[i,:] + 8.0*u[i+1,:] + 36.0*u[i+2,:] - 8.0*u[i+3,:] + 1.0*u[i+4,:])/(12.0*h)
       
    # 6th order
    ii = np.arange(2,nx-1)
    a[ii,:] = 1.0/3.0
    b[ii,:] = 1.0
    c[ii,:] = 1.0/3.0
    r[ii,:] = (1.0/9.0)*(u[(ii+2),:] - u[ii-2,:])/(4.0*h) + (14.0/9.0)*(u[(ii+1),:] - u[ii-1,:])/(2.0*h)

    # 5th order    
    i = nx-1
    a[i,:] = 4.0
    b[i,:] = 1.0
    c[i,:] = 0.0
    r[i,:] = -1.0*(-37.0*u[i,:] + 8.0*u[i-1,:] + 36.0*u[i-2,:] - 8.0*u[i-3,:] + 1.0*u[i-4,:])/(12.0*h)
    
    # 4th order    
    i = nx
    a[i,:] = 4.0
    b[i,:] = 1.0
    c[i,:] = 0.0
    r[i,:] = -1.0*(-37.0*u[i,:] + 8.0*u[i-1,:] + 36.0*u[i-2,:] - 8.0*u[i-3,:] + 1.0*u[i-4,:])/(12.0*h)

    start = 0
    end = nx
        
    ud = tdma(a,b,c,r,start,end)
    
    if isign == 'X':
        fd = np.copy(ud)
    if isign == 'Y':
        fd = np.copy(ud.T)
        
    return fd

if __name__ == "__main__":
    xl = -1.0
    xr = 1.0
    
    error = []
    
    print('#-----------------Dx-------------------#')
    for i in range(5):
        # dx = 0.05/(2**i)
        nx = 16*2**i #int((xr - xl)/dx)
        dx = (xr - xl)/nx
          
        ny = nx
        
        x = np.linspace(xl,xr,nx+1)
        y = np.linspace(xl,xr,ny+1)
        X,Y = np.meshgrid(x,y, indexing='ij')
        xx = (np.ones((ny+1,1))*x).T
        
        u = np.zeros((nx+1,ny+1))
        udx = np.zeros((nx+1,ny+1))
        udn = np.zeros((nx+1,ny+1))
        
        u[:,:] = np.sin(np.pi*X) + np.sin(2.0*np.pi*Y)
        udx[:,:] = (np.pi)*np.cos(np.pi*X) 
        
        udn = c6d_b5_d(u,dx,nx,ny,'X')
#        udn = c4d_b4(u,dx,nx,ny,'X')
        
        errL2 = np.linalg.norm(udx - udn)/np.sqrt(np.size(udn))
        
        error.append(errL2 )
        
        print('#----------------------------------------#')
        print('n = %d' % (nx))
        print('L2 error:  %5.3e' % errL2)
        if i>=1:
            rateL2 = np.log((errL2_0)/(errL2))/np.log(2.0);
            print('L2 order:  %5.3f' % rateL2)
        
        errL2_0 = errL2
    
    print('#-----------------Dy-------------------#')
    for i in range(5):
#        dx = 0.05/(2**i)
        nx = 16*2**i #int((xr - xl)/dx)
        dx = (xr - xl)/nx
          
        ny = nx
        
        x = np.linspace(xl,xr,nx+1)
        y = np.linspace(xl,xr,ny+1)
        X,Y = np.meshgrid(x,y, indexing='ij')
        xx = (np.ones((ny+1,1))*x).T
        
        u = np.zeros((nx+1,ny+1))
        udy = np.zeros((nx+1,ny+1))
        udn = np.zeros((nx+1,ny+1))
        
        u[:,:] = np.sin(np.pi*X) + np.sin(2.0*np.pi*Y)
        udy[:,:] = (2.0*np.pi)*np.cos(2.0*np.pi*Y) 
        
        udn = c6d_b5_d(u,dx,ny,nx,'Y')
#        udn = c4d_b4(u,dx,ny,nx,'Y')
                
        errL2 = np.linalg.norm(udy - udn)/np.sqrt(np.size(udn))
        
        print('#----------------------------------------#')
        print('n = %d' % (nx))
        print('L2 error:  %5.3e' % errL2)
        if i>=1:
            rateL2 = np.log(errL2_0/errL2)/np.log(2.0);
            print('L2 order:  %5.3f' % rateL2)
        
        errL2_0 = errL2
        
    fig, axs = plt.subplots(1,2,figsize=(14,5))
    cs = axs[0].contourf(X, Y, udy, 60,cmap='jet')
    #cax = fig.add_axes([1.05, 0.25, 0.05, 0.5])
    fig.colorbar(cs, ax=axs[0], orientation='vertical')
    
    cs = axs[1].contourf(X, Y, udn,60,cmap='jet')
    #cax = fig.add_axes([1.05, 0.25, 0.05, 0.5])
    fig.colorbar(cs, ax=axs[1], orientation='vertical')
    
    plt.show()
    fig.tight_layout()
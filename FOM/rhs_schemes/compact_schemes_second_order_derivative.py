#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 18:52:37 2021

@author: suraj
"""

import numpy as np
from .thomas_algorithms import *
import matplotlib.pyplot as plt

def c4dd(f,dx,dy,nx,ny,isign):
    
    if isign == 'XX':
        u = np.copy(f)
        h = dx
    if isign == 'YY':
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
    c[i,:] = 11.0
    r[i,:] = (13.0*u[i,:] - 27.0*u[i+1,:] + 15.0*u[i+2,:] - u[i+3,:])/(h**2)
    
    ii = np.arange(1,nx)
    
    a[ii,:] = 1.0/10.0
    b[ii,:] = 1.0
    c[ii,:] = 1.0/10.0
    r[ii,:] = (6.0/5.0)*(u[ii-1,:] - 2.0*u[ii,:] + u[ii+1,:])/(h*h)
    
    i = nx
    a[i,:] = 11.0
    b[i,:] = 1.0
    c[i,:] = 0.0
    r[i,:] = (13.0*u[i,:] - 27.0*u[i-1,:] + 15.0*u[i-2,:] - u[i-3,:])/(h**2)
    
    start = 0
    end = nx
    udd = tdma(a,b,c,r,start,end)
    
    if isign == 'XX':
        fdd = np.copy(udd)
    if isign == 'YY':
        fdd = np.copy(udd.T)
    
    return fdd

def c4dd_p(f,dx,dy,nx,ny,isign):
    
    if isign == 'XX':
        u = np.copy(f)
        h = dx
    if isign == 'YY':
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
    a[ii,:] = 1.0/10.0
    b[ii,:] = 1.0
    c[ii,:] = 1.0/10.0
    r[ii,:] = (6.0/5.0)*(up[ii-1,:] - 2.0*up[ii,:] + up[(ii+1)%nx,:])/(h*h)
    
    start = 0
    end = nx
        
    alpha = np.zeros((1,ny+1))
    beta = np.zeros((1,ny+1))
    
    alpha[0,:] = 1.0/10.0
    beta[0,:] = 1.0/10.0
    
    x = ctdmsv(a,b,c,alpha,beta,r,0,nx-1,ny)
    
    udd = np.zeros((nx+1,ny+1))
    udd[0:nx,:] = x[0:nx,:]
    udd[nx,:] = udd[0,:]
    
    if isign == 'XX':
        fdd = np.copy(udd)
    if isign == 'YY':
        fdd = np.copy(udd.T)
        
    return fdd

def c6dd_p(f,dx,dy,nx,ny,isign):
    
    if isign == 'XX':
        u = np.copy(f)
        h = dx
    if isign == 'YY':
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
    a[ii,:] = 2.0/11.0
    b[ii,:] = 1.0
    c[ii,:] = 2.0/11.0
    r[ii,:] = (3.0/11.0)*(up[ii-2,:] - 2.0*up[ii,:] + up[(ii+2)%nx,:])/(4.0*h*h) + \
              (12.0/11.0)*(up[ii-1,:] - 2.0*up[ii,:] + up[(ii+1)%nx,:])/(h*h)
    
    start = 0
    end = nx
        
    alpha = np.zeros((1,ny+1))
    beta = np.zeros((1,ny+1))
    
    alpha[0,:] = 2.0/11.0
    beta[0,:] = 2.0/11.0
    
    x = ctdmsv(a,b,c,alpha,beta,r,0,nx-1,ny)
    
    udd = np.zeros((nx+1,ny+1))
    udd[0:nx,:] = x[0:nx,:]
    udd[nx,:] = udd[0,:]
    
    if isign == 'XX':
        fdd = np.copy(udd)
    if isign == 'YY':
        fdd = np.copy(udd.T)
        
    return fdd

def c6dd_b3_d(f,dx,dy,nx,ny,isign):
    
    if isign == 'XX':
        u = np.copy(f)
        h = dx
    if isign == 'YY':
        u = np.copy(f.T)
        h = dy
        temp = nx
        nx = ny
        ny = temp
    
    a = np.zeros((nx+1,ny+1))
    b = np.zeros((nx+1,ny+1))
    c = np.zeros((nx+1,ny+1))
    r = np.zeros((nx+1,ny+1))
    
    # 3rd order
    i = 0
    a[i,:] = 0.0
    b[i,:] = 1.0
    c[i,:] = 11.0
    r[i,:] = (13.0*u[i,:] - 27.0*u[i+1,:] + 15.0*u[i+2,:] - u[i+3,:])/(h**2)
    
    # 4th order
    i = 1
    a[i,:] = 1.0/10.0
    b[i,:] = 1.0
    c[i,:] = 1.0/10.0
    r[i,:] = (6.0/5.0)*(u[i-1,:] - 2.0*u[i,:] + u[i+1,:])/(h*h)
    
    # 6th order
    ii = np.arange(2,nx-1)
    a[ii,:] = 2.0/11.0
    b[ii,:] = 1.0
    c[ii,:] = 2.0/11.0
    r[ii,:] = (3.0/11.0)*(u[ii-2,:] - 2.0*u[ii,:] + u[(ii+2),:])/(4.0*h*h) + \
              (12.0/11.0)*(u[ii-1,:] - 2.0*u[ii,:] + u[(ii+1),:])/(h*h)
    
    # 4th order
    i = nx-1
    a[i,:] = 1.0/10.0
    b[i,:] = 1.0
    c[i,:] = 1.0/10.0
    r[i,:] = (6.0/5.0)*(u[i-1,:] - 2.0*u[i,:] + u[i+1,:])/(h*h)
    
    # 3rd order
    i = nx
    a[i,:] = 11.0
    b[i,:] = 1.0
    c[i,:] = 0.0
    r[i,:] = (13.0*u[i,:] - 27.0*u[i-1,:] + 15.0*u[i-2,:] - u[i-3,:])/(h**2)
    
    start = 0
    end = nx
    udd = tdma(a,b,c,r,start,end)
    
    if isign == 'XX':
        fdd = np.copy(udd)
    if isign == 'YY':
        fdd = np.copy(udd.T)
    
    return fdd

def c6dd_b5_d(f,dx,dy,nx,ny,isign):
    
    if isign == 'XX':
        u = np.copy(f)
        h = dx
    if isign == 'YY':
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
    c[i,:] = 10.0
    # r[i,:] = (12.525*u[i,:] - 26.0*u[i+1,:] + 14.3*u[i+2,:] - 0.715*u[i+3,:] + \
    #           0.182*u[i+4,:] + 0.265*u[i+5,:])/(h**2)
    r[i,:] = ((145.0/12.0)*u[i,:] - (76.0/3.0)*u[i+1,:] + (29.0/2.0)*u[i+2,:] - \
              (4.0/3.0)*u[i+3,:] + (1.0/12.0)*u[i+4,:])/(h**2)
    
    # 5th order
    i = 1
    a[i,:] = 0.0
    b[i,:] = 1.0
    c[i,:] = 10.0
    # r[i,:] = (12.525*u[i,:] - 26.0*u[i+1,:] + 14.3*u[i+2,:] - 0.715*u[i+3,:] + \
    #           0.182*u[i+4,:] + 0.265*u[i+5,:])/(h**2)
    r[i,:] = ((145.0/12.0)*u[i,:] - (76.0/3.0)*u[i+1,:] + (29.0/2.0)*u[i+2,:] - \
              (4.0/3.0)*u[i+3,:] + (1.0/12.0)*u[i+4,:])/(h**2)
    
    # 6th order
    ii = np.arange(2,nx-1)
    a[ii,:] = 2.0/11.0
    b[ii,:] = 1.0
    c[ii,:] = 2.0/11.0
    r[ii,:] = (3.0/11.0)*(u[ii-2,:] - 2.0*u[ii,:] + u[(ii+2),:])/(4.0*h*h) + \
              (12.0/11.0)*(u[ii-1,:] - 2.0*u[ii,:] + u[(ii+1),:])/(h*h)
    
    # 5th order
    i = nx-1
    a[i,:] = 10.0
    b[i,:] = 1.0
    c[i,:] = 0.0
    # r[i,:] = (12.525*u[i,:] - 26.0*u[i-1,:] + 14.3*u[i-2,:] - 0.715*u[i-3,:] + \
    #           0.182*u[i-4,:] + 0.265*u[i-5,:])/(h**2)
    r[i,:] = ((145.0/12.0)*u[i,:] - (76.0/3.0)*u[i-1,:] + (29.0/2.0)*u[i-2,:] - \
              (4.0/3.0)*u[i-3,:] + (1.0/12.0)*u[i-4,:])/(h**2)
    
    # 5th order
    i = nx
    a[i,:] = 10.0
    b[i,:] = 1.0
    c[i,:] = 0.0
    # r[i,:] = (12.525*u[i,:] - 26.0*u[i-1,:] + 14.3*u[i-2,:] - 0.715*u[i-3,:] + \
    #           0.182*u[i-4,:] + 0.265*u[i-5,:])/(h**2)
    r[i,:] = ((145.0/12.0)*u[i,:] - (76.0/3.0)*u[i-1,:] + (29.0/2.0)*u[i-2,:] - \
              (4.0/3.0)*u[i-3,:] + (1.0/12.0)*u[i-4,:])/(h**2)
    
    start = 0
    end = nx
    udd = tdma(a,b,c,r,start,end)
    
    if isign == 'XX':
        fdd = np.copy(udd)
    if isign == 'YY':
        fdd = np.copy(udd.T)
    
    return fdd

if __name__ == "__main__":
    xl = -1.0
    xr = 1.0
    
    print('#-----------------Dxx-------------------#')
    for i in range(3):
#        dx = 0.1/(2**i)
        nx = 32*2**i #int((xr - xl)/dx)
        dx = (xr - xl)/nx
          
        ny = nx
        
        x = np.linspace(xl,xr,nx+1)
        y = np.linspace(xl,xr,ny+1)
        X,Y = np.meshgrid(x,y, indexing='ij')
        xx = (np.ones((ny+1,1))*x).T
        
        u = np.zeros((nx+1,ny+1))
        uddx = np.zeros((nx+1,ny+1))
        uddn = np.zeros((nx+1,ny+1))
        
        u[:,:] = np.sin(np.pi*X) + np.sin(2.0*np.pi*Y)
        uddx[:,:] = -(np.pi**2)*np.sin(np.pi*X) 
        
        uddn = c6dd_b5_d(u,dx,nx,ny,'XX')
        
        errL2 = np.linalg.norm(uddx - uddn)/np.sqrt(np.size(uddn))
        
        print('#----------------------------------------#')
        print('n = %d' % (nx))
        print('L2 error:  %5.3e' % errL2)
        if i>=1:
            rateL2 = np.log(errL2_0/errL2)/np.log(2.0);
            print('L2 order:  %5.3f' % rateL2)
        
        errL2_0 = errL2
    
    print('#-----------------Dyy-------------------#')
    for i in range(3):
#        dx = 0.1/(2**i)
        nx = 32*2**i #int((xr - xl)/dx)
        dx = (xr - xl)/nx
          
        ny = nx
        
        x = np.linspace(xl,xr,nx+1)
        y = np.linspace(xl,xr,ny+1)
        X,Y = np.meshgrid(x,y, indexing='ij')
        xx = (np.ones((ny+1,1))*x).T
        
        u = np.zeros((nx+1,ny+1))
        uddy = np.zeros((nx+1,ny+1))
        uddn = np.zeros((nx+1,ny+1))
        
        u[:,:] = np.sin(np.pi*X) + np.sin(2.0*np.pi*Y)
        uddy[:,:] = -(4.0*np.pi**2)*np.sin(2.0*np.pi*Y) 
        
        uddn = c6dd_b5_d(u,dx,ny,nx,'YY')
        
        errL2 = np.linalg.norm(uddy - uddn)/np.sqrt(np.size(uddn))
        
        print('#----------------------------------------#')
        print('n = %d' % (nx))
        print('L2 error:  %5.3e' % errL2)
        if i>=1:
            rateL2 = np.log(errL2_0/errL2)/np.log(2.0);
            print('L2 order:  %5.3f' % rateL2)
        
        errL2_0 = errL2
        
    fig, axs = plt.subplots(1,2,figsize=(14,5))
    cs = axs[0].contourf(X, Y, uddy, 60,cmap='jet')
    #cax = fig.add_axes([1.05, 0.25, 0.05, 0.5])
    fig.colorbar(cs, ax=axs[0], orientation='vertical')
    
    cs = axs[1].contourf(X, Y, uddn,60,cmap='jet')
    #cax = fig.add_axes([1.05, 0.25, 0.05, 0.5])
    fig.colorbar(cs, ax=axs[1], orientation='vertical')
    
    plt.show()
    fig.tight_layout()
    
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 18:15:20 2021

@author: suraj
"""

import numpy as np
from numpy.random import seed
seed(1)
import pyfftw
from scipy import integrate
from scipy import linalg
import matplotlib.pyplot as plt 
import time as tm
import matplotlib.ticker as ticker
import os
from numba import jit
from scipy.fftpack import dst, idst

from scipy.ndimage import gaussian_filter
import yaml

font = {'family' : 'Times New Roman',
        'size'   : 14}    
plt.rc('font', **font)

# import matplotlib as mpl
# mpl.rcParams['text.usetex'] = True
# mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

#%%
def tdma(a,b,c,r,s,e):
    
    a_ = np.copy(a)
    b_ = np.copy(b)
    c_ = np.copy(c)
    r_ = np.copy(r)
    
    un = np.zeros((np.shape(r)[0],np.shape(r)[1]), dtype='complex128')
    
    for i in range(s+1,e+1):
        b_[i,:] = b_[i,:] - a_[i,:]*(c_[i-1,:]/b_[i-1,:])
        r_[i,:] = r_[i,:] - a_[i,:]*(r_[i-1,:]/b_[i-1,:])
        
    un[e,:] = r_[e,:]/b_[e,:]
    
    for i in range(e-1,s-1,-1):
        un[i,:] = (r_[i,:] - c_[i,:]*un[i+1,:])/b_[i,:]
    
    del a_, b_, c_, r_
    
    return un

#%%
def spectral(nx, ny, dx, dy, f):
    epsilon = 1.0e-6
    
    kx = np.fft.fftfreq(nx, d=dx)*(2.0*np.pi)
    ky = np.fft.fftfreq(ny, d=dx)*(2.0*np.pi)
    
    kx[0] = epsilon
    ky[0] = epsilon
    
    data = np.empty((nx,ny), dtype='complex128')
    data1 = np.empty((nx,ny), dtype='complex128')
        
    data[:,:] = np.vectorize(complex)(f[0:nx,0:ny],0.0)
    
    a = pyfftw.empty_aligned((nx,ny),dtype= 'complex128')
    b = pyfftw.empty_aligned((nx,ny),dtype= 'complex128')
    
    fft_object = pyfftw.FFTW(a, b, axes = (0,1), direction = 'FFTW_FORWARD')
    fft_object_inv = pyfftw.FFTW(a, b,axes = (0,1), direction = 'FFTW_BACKWARD')
    
    # compute the fourier transform
    e = fft_object(data)
    
    e[0,0] = 0.0
    
    kx, ky = np.meshgrid(kx, ky, indexing='ij')
    data1 = e/(-kx**2 - ky**2)
    
    # compute the inverse fourier transform
    ut = np.real(fft_object_inv(data1))
    
    #periodicity
    u = np.zeros((nx+1,ny+1)) 
    u[0:nx,0:ny] = ut
    u[:,ny] = u[:,0]
    u[nx,:] = u[0,:]
    u[nx,ny] = u[0,0]
    
    return u

# fast poisson solver using second-order central difference scheme
def fps(nx, ny, dx, dy, f):
    epsilon = 1.0e-6
    aa = -2.0/(dx*dx) - 2.0/(dy*dy)
    bb = 2.0/(dx*dx)
    cc = 2.0/(dy*dy)
    
    hx = 2.0*np.pi/np.float64(nx)
    hy = 2.0*np.pi/np.float64(ny)
    
    kx = hx*np.arange(0, nx, dtype='float64')
    ky = hy*np.arange(0, ny, dtype='float64')

    kx[0] = epsilon
    ky[0] = epsilon
    
    cos_kx, cos_ky = np.meshgrid(np.cos(kx), np.cos(ky), indexing='ij')
    
    data = np.empty((nx,ny), dtype='complex128')
    data_f = np.empty((nx,ny), dtype='complex128')
    
    data[:,:] = np.vectorize(complex)(f[0:nx,0:ny],0.0)

    a = pyfftw.empty_aligned((nx,ny),dtype= 'complex128')
    b = pyfftw.empty_aligned((nx,ny),dtype= 'complex128')
    
    fft_object = pyfftw.FFTW(a, b, axes = (0,1), direction = 'FFTW_FORWARD')
    fft_object_inv = pyfftw.FFTW(a, b,axes = (0,1), direction = 'FFTW_BACKWARD')
    
    e = fft_object(data)    
    e[0,0] = 0.0
    data_f[:,:] = e[:,:]/(aa + bb*cos_kx[:,:] + cc*cos_ky[:,:])

    data_i = np.real(fft_object_inv(data_f))
    
    #periodicity
    u = np.zeros((nx+1,ny+1)) 
    u[0:nx,0:ny] = data_i
    u[:,ny] = u[:,0]
    u[nx,:] = u[0,:]
    u[nx,ny] = u[0,0]
    return u

# fast poisson solver using second-order central difference scheme
def fps4_tdma(nx, ny, dx, dy, f):
    epsilon = 1.0e-6
    
    beta = dx/dy
    aa = -10.0*(1.0 + beta**2)
    bb = 5.0 - beta**2
    cc = 5.0*beta**2 -1.0
    dd = 0.5*(1.0 + beta**2)
    ee = 0.5*(dx**2)
        
    kx = np.arange(0,nx)
    kx = np.reshape(kx,[-1,1])
    cos_kx = np.cos(2.0*np.pi*kx/nx) 
    
    data = np.empty((nx,ny+1), dtype='complex128')
    data1 = np.empty((nx,ny+1), dtype='complex128')
    
    data[:,:] = np.vectorize(complex)(f[0:nx,0:ny+1],0.0)

    a = pyfftw.empty_aligned((nx,ny+1),dtype= 'complex128')
    b = pyfftw.empty_aligned((nx,ny+1),dtype= 'complex128')
    
    fft_object = pyfftw.FFTW(a, b, axes = (0,), direction = 'FFTW_FORWARD')
    fft_object_inv = pyfftw.FFTW(a, b,axes = (0,), direction = 'FFTW_BACKWARD')
    
    # data = np.fft.fft(data, axis=0)
    data = fft_object(data)
    
    alpha_k = cc + 2.0*dd*cos_kx
    beta_k = aa + 2.0*bb*cos_kx
    
    jj = np.arange(1,ny)
    
    alpha = np.zeros((nx,ny+1))
    beta = np.zeros((nx,ny+1))
    rr = np.zeros((nx,ny+1),dtype= 'complex128')
    
    alpha[:,jj] =  alpha_k 
    beta[:,jj] =  beta_k 
    rr[:,jj] = ee*(data[:,jj-1] + (8.0 + 2.0*cos_kx)*data[:,jj] + data[:,jj+1]) 
        
    data_f = tdma(alpha.T,beta.T,alpha.T,rr.T,1,ny-1)
    data_ft = data_f.T

    data_i = np.real(fft_object_inv(data_ft))
    
    #periodicity
    u = np.zeros((nx+1,ny+1)) 
    u[0:nx,0:ny+1] = data_i
    # u[:,ny] = u[:,0]
    u[nx,:] = u[0,:]
    u[nx,ny] = u[0,0]
    return u

#%%
def fst(nx,ny,dx,dy,f):
    data = f[1:-1,1:-1]
        
#    e = dst(data, type=2)
    data = dst(data, axis = 1, type = 1)
    data = dst(data, axis = 0, type = 1)
    
    m = np.linspace(1,nx-1,nx-1).reshape([-1,1])
    n = np.linspace(1,ny-1,ny-1).reshape([1,-1])
        
    data1 = np.zeros((nx-1,ny-1))

    alpha = (2.0/(dx*dx))*(np.cos(np.pi*m/nx) - 1.0) + \
            (2.0/(dy*dy))*(np.cos(np.pi*n/ny) - 1.0)           
    data1 = data/alpha
    
    data1 = idst(data1, axis = 1, type = 1)
    data1 = idst(data1, axis = 0, type = 1)
    
    u = data1/((2.0*nx)*(2.0*ny))
    
    ue = np.zeros((nx+1,ny+1))
    ue[1:-1,1:-1] = u
    
    return ue

def fst4(nx,ny,dx,dy,f):
    
    beta = dx/dy
    a = -10.0*(1.0 + beta**2)
    b = 5.0 - beta**2
    c = 5.0*beta**2 -1.0
    d = 0.5*(1.0 + beta**2)
    
    data = f[1:-1,1:-1]

    data = dst(data, axis = 1, type = 1)
    data = dst(data, axis = 0, type = 1)
    
    m = np.linspace(1,nx-1,nx-1).reshape([-1,1])
    n = np.linspace(1,ny-1,ny-1).reshape([1,-1])
    
    data1 = np.zeros((nx-1,ny-1))
    
    alpha = a + 2.0*b*np.cos(np.pi*m/nx) + 2.0*c*np.cos(np.pi*n/ny) + \
            4.0*d*np.cos(np.pi*m/nx)*np.cos(np.pi*n/ny)
            
    gamma = 8.0 + 2.0*np.cos(np.pi*m/nx) + 2.0*np.cos(np.pi*n/ny)
               
    data1 = data*(dx**2)*0.5*gamma/alpha
    
    data1 = idst(data1, axis = 1, type = 1)
    data1 = idst(data1, axis = 0, type = 1)
    
    data1 = data1/((2.0*nx)*(2.0*ny))
    
    ue = np.zeros((nx+1,ny+1))
    ue[1:-1,1:-1] = data1
    
    return ue

def fst4c(nx,ny,dx,dy,f):
        
    alpha_ = 1.0/10.0
    a = (-12.0/5.0)*(1.0/dx**2 + 1.0/dy**2)
    b = (6.0/25.0)*(5.0/dx**2 - 1/(dy**2))
    c = (6.0/25.0)*(5.0/dy**2 - 1/(dx**2))
    d =  (3.0/25.0)*(1.0/dx**2 + 1.0/dy**2)
    
    data = f[1:-1,1:-1]

    data = dst(data, axis = 1, type = 1)
    data = dst(data, axis = 0, type = 1)
    
    m = np.linspace(1,nx-1,nx-1).reshape([-1,1])
    n = np.linspace(1,ny-1,ny-1).reshape([1,-1])
    
    data1 = np.zeros((nx-1,ny-1))
       
    beta = a + 2.0*b*np.cos(np.pi*m/nx) + 2.0*c*np.cos(np.pi*n/ny) + \
            4.0*d*np.cos(np.pi*m/nx)*np.cos(np.pi*n/ny)
            
    gamma = 1.0 + 2.0*alpha_*np.cos(np.pi*m/nx) + 2.0*alpha_*np.cos(np.pi*n/ny) + \
            4.0*(alpha_**2)*np.cos(np.pi*m/nx)*np.cos(np.pi*n/ny)
                              
    data1 = data*gamma/(beta)
    
    data1 = idst(data1, axis = 1, type = 1)
    data1 = idst(data1, axis = 0, type = 1)
    
    data1 = data1/((2.0*nx)*(2.0*ny))
    
    ue = np.zeros((nx+1,ny+1))
    ue[1:-1,1:-1] = data1
    
    return ue

def fst6(nx,ny,dx,dy,f):
    
    lambda_ = 1.0/10.0
    a = (-12.0/5.0)*(1.0/dx**2 + 1.0/dy**2)
    b = (6.0/25.0)*(5.0/dx**2 - 1/(dy**2))
    c = (6.0/25.0)*(5.0/dy**2 - 1/(dx**2))
    d =  (3.0/25.0)*(1.0/dx**2 + 1.0/dy**2)
    
    alpha_ = 2.0/11.0
    center = (-51.0/22.0)*(1.0/dx**2 + 1.0/dy**2)
    ew = (1.0/11.0)*(12.0/dx**2 - alpha_*51.0/(2.0*dy**2))
    ns = (1.0/11.0)*(12.0/dy**2 - alpha_*51.0/(2.0*dx**2))
    corners =  alpha_*(12.0/11.0)*(1.0/dx**2 + 1.0/dy**2)
    ew_far = alpha_*3.0/(44.0*dx**2)
    ns_far = alpha_*3.0/(44.0*dy**2)
    
    data = f[1:-1,1:-1]

    data = dst(data, axis = 1, type = 1)
    data = dst(data, axis = 0, type = 1)
    
    m = np.linspace(1,nx-1,nx-1).reshape([-1,1])
    n = np.linspace(1,ny-1,ny-1).reshape([1,-1])
    
    data1 = np.zeros((nx-1,ny-1))
    data2 = np.zeros((nx-1,ny-1))
    data3 = np.zeros((nx-1,ny-1))
    
    beta = a + 2.0*b*np.cos(np.pi*m/nx) + 2.0*c*np.cos(np.pi*n/ny) + \
            4.0*d*np.cos(np.pi*m/nx)*np.cos(np.pi*n/ny)
            
    gamma = 1.0 + 2.0*lambda_*np.cos(np.pi*m/nx) + 2.0*lambda_*np.cos(np.pi*n/ny) + \
            4.0*(lambda_**2)*np.cos(np.pi*m/nx)*np.cos(np.pi*n/ny)
                              
    data2 = data*gamma/(beta)
    
    lhs_near = center + 2.0*ew*np.cos(np.pi*m/nx) + 2.0*ns*np.cos(np.pi*n/ny) + \
               4.0*corners*np.cos(np.pi*m/nx)*np.cos(np.pi*n/ny)

    lhs_far = 2.0*ew_far*(1.0 + 2.0*np.cos(np.pi*n/ny))*np.cos(2.0*np.pi*m/nx) + \
              2.0*ns_far*(1.0 + 2.0*np.cos(np.pi*m/nx))*np.cos(2.0*np.pi*n/ny) 

    rhs = 1.0 + 2.0*alpha_*np.cos(np.pi*m/nx) + 2.0*alpha_*np.cos(np.pi*n/ny) + \
          4.0*(alpha_**2)*np.cos(np.pi*m/nx)*np.cos(np.pi*n/ny)
               
    data3 = data*rhs/(lhs_near + lhs_far)
    
    data1 = np.copy(data2)
    data1[1:-1,1:-1] = np.copy(data3[1:-1,1:-1])
    
    data1 = idst(data1, axis = 1, type = 1)
    data1 = idst(data1, axis = 0, type = 1)
    
    data1 = data1/((2.0*nx)*(2.0*ny))
    
    ue = np.zeros((nx+1,ny+1))
    ue[1:-1,1:-1] = data1
    
    return ue

def fst4_tdma(nx,ny,dx,dy,f):
    
    beta = dx/dy
    a4 = -10.0*(1.0 + beta**2)
    b4 = 5.0 - beta**2
    c4 = 5.0*beta**2 -1.0
    d4 = 0.5*(1.0 + beta**2)
    e4 = 0.5*(dx**2)
    
    data = f[1:-1,:]

    data = dst(data, axis = 0, type = 1)
    
    kx = np.linspace(1,nx-1,nx-1)
    kx = np.reshape(kx,[-1,1])
    cos_kx = np.cos(np.pi*kx/nx)
    
    alpha_k = c4 + 2.0*d4*cos_kx
    beta_k = a4 + 2.0*b4*cos_kx
    
    jj = np.arange(1,ny)
    
    alpha = np.zeros((nx-1,ny+1))
    beta = np.zeros((nx-1,ny+1))
    rr = np.zeros((nx-1,ny+1),dtype= 'complex128')
    
    alpha[:,jj] =  alpha_k 
    beta[:,jj] =  beta_k 
    rr[:,jj] = e4*(data[:,jj-1] + (8.0 + 2.0*cos_kx)*data[:,jj] + data[:,jj+1]) 
            
    data_f = tdma(alpha.T,beta.T,alpha.T,rr.T,1,ny-1)
    data_ft = data_f.T
    
    data_i = idst(data_ft[:,1:-1], axis = 0, type = 1)
    data_i = np.real(data_i)/((2.0*nx))
    
    ue = np.zeros((nx+1,ny+1))
    ue[1:-1,1:-1] = data_i
    
    return ue

#%% 
if __name__ == "__main__":
    with open(r'poisson_solver.yaml') as file:
        input_data = yaml.load(file, Loader=yaml.FullLoader)    
    file.close()


    for i in range(3):
        nx = 16*(2**i)
        ny = 16*(2**i)
    
        
        x_l = input_data['x_l']
        x_r = input_data['x_r']
        y_b = input_data['y_b']
        y_t = input_data['y_t']
        ipr = input_data['ipr']
        ips = input_data['ips']

        dx = (x_r-x_l)/nx
        dy = (y_t-y_b)/ny
        
        x = np.linspace(x_l, x_r, nx+1)
        y = np.linspace(y_b, y_t, ny+1)
        
        xm, ym = np.meshgrid(x,y, indexing='ij')
        
        if ipr == 1:
            ue = np.sin(2.0*np.pi*xm)*np.sin(2.0*np.pi*ym)    
            f = -8.0*np.pi**2*np.sin(2.0*np.pi*xm)*np.sin(2.0*np.pi*ym)
        elif ipr == 2:
            km = 16.0
            c1 = (1.0/km)**2
            c2 = -2.0*np.pi**2
            
            ue = np.sin(2.0*np.pi*xm)*np.sin(2.0*np.pi*ym) + \
                  c1*np.sin(km*np.pi*xm)*np.sin(km*np.pi*ym)
            
            f = 4.0*c2*np.sin(2.0*np.pi*xm)*np.sin(2.0*np.pi*ym) + \
                c2*np.sin(km*np.pi*xm)*np.sin(km*np.pi*ym)
        
        if ips == 1:
            un = spectral(nx,ny,dx,dy,f)
        elif ips == 2:
            un = fst(nx,ny,dx,dy,f)
        elif ips == 3:
            un = fst4(nx,ny,dx,dy,f)
        elif ips == 4:
            un = fst4_tdma(nx,ny,dx,dy,f)
        elif ips == 5:
            un = fps(nx,ny,dx,dy,f)
        elif ips == 6:
            un = fps4_tdma(nx,ny,dx,dy,f)
        elif ips == -1:
            un = fst4(nx,ny,dx,dy,f)
            
        errL2 = np.linalg.norm(un - ue)/np.sqrt(np.size(ue))
        
        print('#----------------------------------------#')
        print('n = %d' % (nx))
        print('L2 error:  %5.3e' % errL2)
        if i>=1:
            rateL2 = np.log(errL2_0/errL2)/np.log(2.0);
            print('L2 order:  %5.3f' % rateL2)
        
        errL2_0 = errL2
        
    fig, axs = plt.subplots(1,2,figsize=(14,5))
    cs = axs[0].contourf(xm, ym, ue, 60,cmap='jet')
    #cax = fig.add_axes([1.05, 0.25, 0.05, 0.5])
    fig.colorbar(cs, ax=axs[0], orientation='vertical')
    
    cs = axs[1].contourf(xm, ym, un,60,cmap='jet')
    #cax = fig.add_axes([1.05, 0.25, 0.05, 0.5])
    fig.colorbar(cs, ax=axs[1], orientation='vertical')
    
    plt.show()
    fig.tight_layout()
        

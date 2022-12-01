#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 17:29:12 2021

@author: suraj
"""


import numpy as np
from scipy.fftpack import dst, idst
#from numpy import empty,arange,exp,real,imag,pi
#from numpy.fft import rfft,irfft
import matplotlib.pyplot as plt 
import time
import yaml

#%%
def compute_residual(nx, ny, dx, dy, f, u_n):
    r = np.zeros((nx+1, ny+1))
    d2udx2 = np.zeros((nx+1, ny+1))
    d2udy2 = np.zeros((nx+1, ny+1))
    ii = np.arange(1,nx)
    jj = np.arange(1,ny)
    i,j = np.meshgrid(ii,jj,indexing='ij')
    
    d2udx2[i,j] = (u_n[i+1,j] - 2*u_n[i,j] + u_n[i-1,j])/(dx**2)
    d2udy2[i,j] = (u_n[i,j+1] - 2*u_n[i,j] + u_n[i,j-1])/(dy**2)
    r[i,j] = f[i,j]  - d2udx2[i,j] - d2udy2[i,j]
    
    del d2udx2, d2udy2
    
    return r
    
def restriction(nxf, nyf, nxc, nyc, r):
    ec = np.zeros((nxc+1, nyc+1))
    center = np.zeros((nxc+1, nyc+1))
    grid = np.zeros((nxc+1, nyc+1))
    corner = np.zeros((nxc+1, nyc+1))
    
    ii = np.arange(1,nxc)
    jj = np.arange(1,nyc)
    i,j = np.meshgrid(ii,jj,indexing='ij')
    
    # grid index for fine grid for the same coarse point
    center[i,j] = 4.0*r[2*i, 2*j]
    
    # E, W, N, S with respect to coarse grid point in fine grid
    grid[i,j] = 2.0*(r[2*i, 2*j+1] + r[2*i, 2*j-1] +
                r[2*i+1, 2*j] + r[2*i-1, 2*j])
    
    # NE, NW, SE, SW with respect to coarse grid point in fine grid
    corner[i,j] = 1.0*(r[2*i+1, 2*j+1] + r[2*i+1, 2*j-1] +
                  r[2*i-1, 2*j+1] + r[2*i-1, 2*j-1])
    
    # restriction using trapezoidal rule
    ec[i,j] = (center[i,j] + grid[i,j] + corner[i,j])/16.0
    
    del center, grid, corner
    
    i = np.arange(0,nxc+1)
    ec[i,0] = r[2*i, 0]
    ec[i,nyc] = r[2*i, nyf]
    
    j = np.arange(0,nyc+1)
    ec[0,j] = r[0, 2*j]
    ec[nxc,j] = r[nxf, 2*j]
    
    return ec

def prolongation(nxc, nyc, nxf, nyf, unc):
    ef = np.zeros((nxf+1, nyf+1))
    ii = np.arange(0,nxc)
    jj = np.arange(0,nyc)
    i,j = np.meshgrid(ii,jj,indexing='ij')
    
    ef[2*i, 2*j] = unc[i,j]
    # east neighnour on fine grid corresponding to coarse grid point
    ef[2*i, 2*j+1] = 0.5*(unc[i,j] + unc[i,j+1])
    # north neighbout on fine grid corresponding to coarse grid point
    ef[2*i+1, 2*j] = 0.5*(unc[i,j] + unc[i+1,j])
    # NE neighbour on fine grid corresponding to coarse grid point
    ef[2*i+1, 2*j+1] = 0.25*(unc[i,j] + unc[i,j+1] + unc[i+1,j] + unc[i+1,j+1])
    
    i = np.arange(0,nxc+1)
    ef[2*i,nyf] = unc[i,nyc]
    
    j = np.arange(0,nyc+1)
    ef[nxf,2*j] = unc[nxc,j]
    
    return ef

def gauss_seidel_mg(nx, ny, dx, dy, f, un, V):
    rt = np.zeros((nx+1,ny+1))
    den = -2.0/dx**2 - 2.0/dy**2
    omega = 1.0
    unr = np.copy(un)
    
    for k in range(V):
        for j in range(1,nx):
            for i in range(1,ny):
                rt[i,j] = f[i,j] - \
                (unr[i+1,j] - 2.0*unr[i,j] + unr[i-1,j])/dx**2 - \
                (unr[i,j+1] - 2.0*unr[i,j] + unr[i,j-1])/dy**2
  
                unr[i,j] = unr[i,j] + omega*rt[i,j]/den
    
    # ii = np.arange(1,nx)
    # jj = np.arange(1,ny)
    # i,j = np.meshgrid(ii,jj,indexing='ij')
    
    # for k in range(V):
    #     rt[i,j] = f[i,j] - \
    #               (unr[i+1,j] - 2.0*unr[i,j] + unr[i-1,j])/dx**2 - \
    #               (unr[i,j+1] - 2.0*unr[i,j] + unr[i,j-1])/dy**2
                  
    #     unr[i,j] = unr[i,j] + omega*rt[i,j]/den
    
    return unr


def mg_n_solver(f, dx, dy, nx, ny, input_data, iprint=False):
    
    # with open(r'../ldc_parameters.yaml') as file:
    #     input_data = yaml.load(file, Loader=yaml.FullLoader)
    
    n_level = input_data['nlevel']
    max_iterations = input_data['pmax']
    v1 = input_data['v1']
    v2 = input_data['v2']
    v3 = input_data['v3']
    tolerance = float(input_data['tolerance'])

    un = np.zeros((nx+1,ny+1))    
    u_mg = []
    f_mg = []    
    
    u_mg.append(un)
    f_mg.append(f)
    
    r = compute_residual(nx, ny, dx, dy, f_mg[0], u_mg[0])
    
    rms = np.linalg.norm(r)/np.sqrt((nx-1)*(ny-1))
    init_rms = np.copy(rms)
    
    if iprint:
        print('%0.2i %0.5e %0.5e' % (0, rms, rms/init_rms))
    
    if nx < 2**n_level:
        print("Number of levels exceeds the possible number.\n")
    
    lnx = np.zeros(n_level, dtype='int')
    lny = np.zeros(n_level, dtype='int')
    ldx = np.zeros(n_level)
    ldy = np.zeros(n_level)
    
    
    # initialize the mesh details at fine level
    lnx[0] = nx
    lny[0] = ny
    ldx[0] = dx
    ldy[0] = dy
    
    for i in range(1,n_level):
        lnx[i] = int(lnx[i-1]/2)
        lny[i] = int(lny[i-1]/2)
        ldx[i] = ldx[i-1]*2
        ldy[i] = ldy[i-1]*2
        
        fc = np.zeros((lnx[i]+1, lny[i]+1))
        unc = np.zeros((lnx[i]+1, lny[i]+1))
        
        u_mg.append(unc)
        f_mg.append(fc)
    
    # allocate matrix for storage at fine level
    # residual at fine level is already defined at global level
    prol_fine = np.zeros((lnx[1]+1, lny[1]+1))    
    
    # temporaty residual which is restricted to coarse mesh error
    # the size keeps on changing
    temp_residual = np.zeros((lnx[1]+1, lny[1]+1))    
    
    start = time.time()
    
    # start main iteration loop
    for iteration_count in range(max_iterations):  
        k = 0
        u_mg[k][:,:] = gauss_seidel_mg(lnx[k], lny[k], ldx[k], ldy[k], f_mg[k], u_mg[k], v1)
        
        r = compute_residual(lnx[k], lny[k], ldx[k], ldy[k], f_mg[k], u_mg[k])
        
        rms = np.linalg.norm(r)/np.sqrt((nx-1)*(ny-1))
        
        if iprint:
            print('%0.2i %0.5e %0.5e' % (iteration_count+1, rms, rms/init_rms))
        
        if rms/init_rms <= tolerance:
            break
        
        for k in range(1,n_level):
#            print(k, lnx[k])
            # if k == 1:
            #     temp_residual = r
            # else:
            temp_residual = compute_residual(lnx[k-1], lny[k-1], ldx[k-1], ldy[k-1], 
                                                 f_mg[k-1], u_mg[k-1])
                
            f_mg[k] = restriction(lnx[k-1], lny[k-1], lnx[k], lny[k], temp_residual)
            
            # solution at kth level to zero
            u_mg[k][:,:] = 0.0
            
            if k < n_level-1:
                u_mg[k][:,:] = gauss_seidel_mg(lnx[k], lny[k], ldx[k], ldy[k], f_mg[k], u_mg[k], v1)
            elif k == n_level-1:
                u_mg[k][:,:] = gauss_seidel_mg(lnx[k], lny[k], ldx[k], ldy[k], f_mg[k], u_mg[k], v2)
        
        for k in range(n_level-1,0,-1):
            prol_fine = prolongation(lnx[k], lny[k], lnx[k-1], lny[k-1],
                             u_mg[k])
            
            ii = np.arange(1,lnx[k-1])
            jj = np.arange(1,lny[k-1])
            i,j = np.meshgrid(ii,jj,indexing='ij')
            
            u_mg[k-1][i,j] = u_mg[k-1][i,j] + prol_fine[i,j]
            
            u_mg[k-1][:,:] = gauss_seidel_mg(lnx[k-1], lny[k-1], ldx[k-1], ldy[k-1],
                                f_mg[k-1], u_mg[k-1], v3)
        
        # if iprint:
        #     print('Time = ', time.time() - start)   
            
    return u_mg[0]


#%% 
if __name__ == "__main__":
    input_data = {}
    
    input_data['nlevel'] = 8
    input_data['pmax'] = 15
    input_data['v1'] = 2 
    input_data['v2'] = 2
    input_data['v3'] = 2
    input_data['tolerance'] = 1e-10
    
    nx = 512
    ny = 512
    
    x_l = 0.0
    x_r = 1.0
    y_b = 0.0
    y_t = 1.0
    
    dx = (x_r-x_l)/nx
    dy = (y_t-y_b)/ny
    
    x = np.linspace(x_l, x_r, nx+1)
    y = np.linspace(y_b, y_t, ny+1)
    
    xm, ym = np.meshgrid(x,y, indexing='ij')
    
    km = 16.0
    c1 = (1.0/km)**2
    c2 = -2.0*np.pi**2
    
    # ue = np.sin(2.0*np.pi*xm) * np.sin(2.0*np.pi*ym) + \
    #             c1*np.sin(16.0*np.pi*xm) * np.sin(16.0*np.pi*ym)
    
    # f = 4.0*c2*np.sin(2.0*np.pi*xm) * np.sin(2.0*np.pi*ym) + \
    #          c2*np.sin(16.0*np.pi*xm) * np.sin(16.0*np.pi*ym)
                     
    ue = np.sin(2.0*np.pi*xm)*np.sin(2.0*np.pi*ym) + \
          c1*np.sin(km*np.pi*xm)*np.sin(km*np.pi*ym)
    
    f = 4.0*c2*np.sin(2.0*np.pi*xm)*np.sin(2.0*np.pi*ym) + \
        c2*np.sin(km*np.pi*xm)*np.sin(km*np.pi*ym)
             
    
    
    un = mg_n_solver(f, dx, dy, nx, ny, input_data, iprint=True)    
    
    fig, axs = plt.subplots(1,2,figsize=(14,5))
    cs = axs[0].contourf(xm, ym, ue, 60,cmap='jet')
    #cax = fig.add_axes([1.05, 0.25, 0.05, 0.5])
    fig.colorbar(cs, ax=axs[0], orientation='vertical')
    
    cs = axs[1].contourf(xm, ym, un,60,cmap='jet')
    #cax = fig.add_axes([1.05, 0.25, 0.05, 0.5])
    fig.colorbar(cs, ax=axs[1], orientation='vertical')
    
    plt.show()
    fig.tight_layout()
    fig.savefig('mg.png', bbox_inches = 'tight', pad_inches = 0, dpi = 300)
    
    print(np.linalg.norm(un-ue)/np.sqrt((nx*ny)))    

    
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 17:24:39 2021

@author: suraj
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import inv


def tdma(a,b,c,r,s,e):
    
    a_ = np.copy(a)
    b_ = np.copy(b)
    c_ = np.copy(c)
    r_ = np.copy(r)
    
    un = np.zeros((np.shape(r)[0],np.shape(r)[1]))
    
    for i in range(s+1,e+1):
        # print(i)
        b_[i,:] = b_[i,:] - a_[i,:]*(c_[i-1,:]/b_[i-1,:])
        # if b_[i,:] == 0:
        #     b_[i,:] = 1e-8
        r_[i,:] = r_[i,:] - a_[i,:]*(r_[i-1,:]/b_[i-1,:])
        
    un[e,:] = r_[e,:]/b_[e,:]
    
    for i in range(e-1,s-1,-1):
        un[i,:] = (r_[i,:] - c_[i,:]*un[i+1,:])/b_[i,:]
    
    del a_, b_, c_, r_
    
    return un

def tdmsv(a,b,c,r,s,e,n):
    gam = np.zeros((e+1,n+1))
    u = np.zeros((e+1,n+1))
    bet = np.zeros((1,n+1))
    
    bet[0,:] = b[s,:]
    u[s,:] = r[s,:]/bet[0,:]
    
    for i in range(s+1,e+1):
        gam[i,:] = c[i-1,:]/bet[0,:]
        bet[0,:] = b[i,:] - a[i,:]*gam[i,:]
        u[i,:] = (r[i,:] - a[i,:]*u[i-1,:])/bet[0,:]
    
    for i in range(e-1,s-1,-1):
        u[i,:] = u[i,:] - gam[i+1,:]*u[i+1,:]
    
    return u
        
#-----------------------------------------------------------------------------#
# Solution to tridigonal system using cyclic Thomas algorithm
#-----------------------------------------------------------------------------#
def ctdmsv(a,b,c,alpha,beta,r,s,e,n):
    bb = np.zeros((e+1,n+1))
    u = np.zeros((e+1,n+1))
    gamma = np.zeros((1,n+1))
    
    gamma[0,:] = -b[s,:]
    bb[s,:] = b[s,:] - gamma[0,:]
    bb[e,:] = b[e,:] - alpha*beta/gamma[0,:]
    
#    for i in range(s+1,e):
#        bb[i] = b[i]
    
    bb[s+1:e,:] = b[s+1:e,:]
    
    x = tdmsv(a,bb,c,r,s,e,n)
    
    u[s,:] = gamma[0,:]
    u[e,:] = alpha[0,:]
    
    z = tdmsv(a,bb,c,u,s,e,n)
    
    fact = (x[s,:] + beta[0,:]*x[e,:]/gamma[0,:])/(1.0 + z[s,:] + beta[0,:]*z[e,:]/gamma[0,:])
    
#    for i in range(s,e+1):
#        x[i] = x[i] - fact*z[i]
    
    x[s:e+1,:] = x[s:e+1,:] - fact*z[s:e+1,:]
        
    return x

if __name__ == "__main__":
    
    print('3 X 3 matrix, s = 0, e = 3')
    A = np.array([[10,2,0,0],[3,10,4,0],[0,1,7,5],[0,0,3,4]],dtype=float)   

    a = np.array([0.,3.,1,3]) 
    b = np.array([10.,10.,7.,4.])
    c = np.array([2.,4.,5.,0])
    d = np.array([3.,4.,5.,6.])
    
    a = np.reshape(a,[-1,1])
    b = np.reshape(b,[-1,1])
    c = np.reshape(c,[-1,1])
    d = np.reshape(d,[-1,1])
    
    print(tdma(a, b, c, d, 0, 3) - np.linalg.solve(A, d))
    
    print(tdmsv(a,b,c,d,0,3,1) - np.linalg.solve(A, d))
    
    print('4 X 4 matrix, s = 0, e = 4')
    A = np.array([[1,0,0,0,0],[3,10,4,0,0],[0,1,7,5,0],[0,0,2,8,9],[0,0,0,0,1]],dtype=float)   

    a = np.array([0.,3.,1.,2.,0.]) 
    b = np.array([1.,10.,7.,8.,1.])
    c = np.array([0.,4.,5.,9.,0])
    d = np.array([3.,4.,5.,6.,7.])
    
    a = np.reshape(a,[-1,1])
    b = np.reshape(b,[-1,1])
    c = np.reshape(c,[-1,1])
    d = np.reshape(d,[-1,1])
    
    r = tdma(a, b, c, d, 0, 4)
    re = np.linalg.solve(A, d)
    print(r - re)
    
    print('4 X 4 matrix, s = 1, e = 3')    
    A = np.array([[1,0,0,0,0],[3,10,4,0,0],[0,1,7,5,0],[0,0,2,8,9],[0,0,0,0,1]],dtype=float)   

    a = np.array([0.,3.,1.,2.,0.]) 
    b = np.array([1.,10.,7.,8.,4.])
    c = np.array([0.,4.,5.,9.,0])
    d = np.array([3.,4.,5.,6.,7.])
    dd = np.array([3.,4.-3.*3.,5.,6.-9.*7.,7.])
    # dd = np.array([3.,4.,5.,6.,7.])
    
    a = np.reshape(a,[-1,1])
    b = np.reshape(b,[-1,1])
    c = np.reshape(c,[-1,1])
    dd = np.reshape(dd,[-1,1])
    
    r = tdma(a, b, c, dd, 1, 3)[1:-1,0]
    re = np.linalg.solve(A, d)[1:-1]
    print(r - re)

#%%    
    nx = 8
    A = csc_matrix((nx+1, nx+1))
    ii = np.arange(1,nx)
    A[ii,ii] = 1.0
    A[ii,ii-1] = 1.0/4.0
    A[ii,ii+1] = 1.0/4.0
    
    ii = 0
    A[ii,ii] = 1.0
    A[ii,ii+1] = 2.0
    
    ii = nx
    A[ii,ii] = 1.0
    A[ii,ii-1] = 2.0
    
    print(A.todense())
        
            
    
    
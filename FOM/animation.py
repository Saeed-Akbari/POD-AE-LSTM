#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 17:31:45 2021

@author: suraj
"""

import matplotlib.pyplot as plt
import numpy as np

import matplotlib.animation as animation

font = {'family' : 'Times New Roman',
        'size'   : 16}    
plt.rc('font', **font)


#'weight' : 'bold'

import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']


lx = 2.0
ly = 1.0

nx = 32
ny = 16

x = np.linspace(0.0,lx,nx+1)
y = np.linspace(0.0,ly,ny+1)
X, Y = np.meshgrid(x, y, indexing='ij')
ns = 50

w = np.zeros((ns, nx+1, ny+1))
s = np.zeros((ns, nx+1, ny+1))

for i in range(50):
    data = np.load(f'ws_{i}_32_16_100000.0_2_3_1.npz')
    s[i,:,:] = data['s']
    w[i,:,:] = data['w']

    
#%%    
fig = plt.figure(figsize=(14,7))

plt.xticks([])
plt.yticks([])
    
def animate(i): 
    cont = plt.contourf(X,Y,s[i,:,:],120,cmap='jet')
    return cont  
    
anim = animation.FuncAnimation(fig, animate, frames=50)
fig.tight_layout()
# anim.save('animation.mp4')
writergif = animation.PillowWriter(fps=10)
anim.save('filename.gif',writer=writergif)

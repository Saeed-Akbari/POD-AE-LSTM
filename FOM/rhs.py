#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 18:19:49 2021

@author: suraj
"""

from rhs_schemes.rhs_conservative import *
from rhs_schemes.rhs_compact import *

def rhs(nx,ny,dx,dy,re,pr,w,s,th,input_data):
    isolver = input_data['isolver']
    icompact = input_data['icompact']
        
    if isolver == 1:
        rw, rth = rhs_arakawa(nx,ny,dx,dy,re,pr,w,s,th)
        return rw, rth
    elif isolver == 2:
        rw, rth = rhs_cs(nx,ny,dx,dy,re,pr,w,s,th)
        return rw, rth
    elif isolver == 3:
        rw, rth = rhs_compact_scheme(nx,ny,dx,dy,re,pr,w,s,th,icompact)
        return rw, rth
# Kernel denoisers with guide image

import numpy as np
from .utils import *

def nlm(noisy_img,guide_img,patch_rad,window_rad,sigma):
    if(len(noisy_img.shape) > 2):
        raise ValueError('Input must be a 2D array')
    height,width = noisy_img.shape
    A = np.zeros((width**2, (2*window_rad+1)**2))
    M_nsy = np.zeros((width**2, (2*window_rad+1)**2))
    Y_swp = np.zeros((width**2, (2*window_rad+1)**2))
    A1 = np.zeros((width**2, (2*window_rad+1)**2))
    U = np.zeros((height, width))   # To hold denoised image
    Z = np.zeros((height, width))   # To hold accumulated weights
    
    padded_guide = np.pad(guide_img,patch_rad,mode='symmetric')
    padded_v = np.pad(noisy_img,window_rad,mode='symmetric')
    
    for dx in np.arange(-window_rad,window_rad+1):
        for dy in np.arange(-window_rad,window_rad+1):
            sd,diff,t = integral_img_sq_diff(padded_guide,dx,dy)
            hat = triangle(dx,dy,window_rad)
            temp1 = img2Dshift(sd,patch_rad,patch_rad)
            temp2 = img2Dshift(sd,-patch_rad-1,-patch_rad-1)
            temp3 = img2Dshift(sd,-patch_rad-1,patch_rad)
            temp4 = img2Dshift(sd,patch_rad,-patch_rad-1)
            res = temp1 + temp2 - temp3 - temp4
            sqdist1 = res[patch_rad:patch_rad+height,patch_rad:patch_rad+width]
            w = hat * np.exp(-sqdist1/(sigma**2))
            row = dx+window_rad
            col = dy+window_rad
            row1 = dx+window_rad
            col1 = dy+window_rad
            A1[:,col1*(2*window_rad+1) + row1] = np.reshape(w.T,(height*width,1),order='F').T
            v = padded_v[window_rad+dx:window_rad+dx+height,window_rad+dy:window_rad+dy+width]
            v_swp = padded_v[window_rad-dx:window_rad-dx+height,window_rad-dy:window_rad-dy+width]
            M_nsy[:,col*(2*window_rad+1) + row] = np.reshape(v.T,(height*width,1),order='F').T
            Y_swp[:,col*(2*window_rad+1) + row] = np.reshape(v_swp.T,(height*width,1),order='F').T
            U = U + w*v
            Z = Z + w
    U = U/Z
    return(U,Z)     # U = Denoised image, Z = Normalization coefficients


def dsg_nlm(noisy_img,guide_img,patch_rad,window_rad,sigma):
    if(len(noisy_img.shape) > 2):
        raise ValueError('Input must be a 2D array')
    height,width = noisy_img.shape
    u = np.zeros((height,width))
    
    padded_guide = np.pad(guide_img,patch_rad,mode='symmetric')
    padded_v = np.pad(noisy_img,window_rad,mode='symmetric')

    # 0th loop
    W0 = np.zeros((height,width))
    for dx in np.arange(-window_rad,window_rad+1):
        for dy in np.arange(-window_rad,window_rad+1):
            sd,diff,t = integral_img_sq_diff(padded_guide,dx,dy)
            hat = triangle(dx,dy,window_rad)
            temp1 = img2Dshift(sd,patch_rad,patch_rad)
            temp2 = img2Dshift(sd,-patch_rad-1,-patch_rad-1)
            temp3 = img2Dshift(sd,-patch_rad-1,patch_rad)
            temp4 = img2Dshift(sd,patch_rad,-patch_rad-1)
            res = temp1 + temp2 - temp3 - temp4
            sqdist1 = res[patch_rad:patch_rad+height,patch_rad:patch_rad+width]
            w = hat * np.exp(-sqdist1/(sigma**2))
            W0 = W0 + w

    # 1st loop
    W1 = np.zeros((height,width))
    for dx in np.arange(-window_rad,window_rad+1):
        for dy in np.arange(-window_rad,window_rad+1):
            sd,diff,t = integral_img_sq_diff(padded_guide,dx,dy)
            hat = triangle(dx,dy,window_rad)
            temp1 = img2Dshift(sd,patch_rad,patch_rad)
            temp2 = img2Dshift(sd,-patch_rad-1,-patch_rad-1)
            temp3 = img2Dshift(sd,-patch_rad-1,patch_rad)
            temp4 = img2Dshift(sd,patch_rad,-patch_rad-1)
            res = temp1 + temp2 - temp3 - temp4
            sqdist1 = res[patch_rad:patch_rad+height,patch_rad:patch_rad+width]
            w = hat * np.exp(-sqdist1/(sigma**2))
            W0_pad = np.pad(W0,window_rad,mode='symmetric')
            W0_shift = img2Dshift(W0_pad,dx,dy)
            W0_temp = W0_shift[window_rad:window_rad+height,window_rad:window_rad+width]
            w1 = w / (np.sqrt(W0)*np.sqrt(W0_temp))
            W1 = W1 + w1
    
    # 2nd loop
    alpha = 1/np.max(W1)
    W2 = np.zeros((height,width))
    for dx in np.arange(-window_rad,window_rad+1):
        for dy in np.arange(-window_rad,window_rad+1):
            if((dx != 0) or (dy != 0)):
                sd,diff,t = integral_img_sq_diff(padded_guide,dx,dy)
                hat = triangle(dx,dy,window_rad)
                temp1 = img2Dshift(sd,patch_rad,patch_rad)
                temp2 = img2Dshift(sd,-patch_rad-1,-patch_rad-1)
                temp3 = img2Dshift(sd,-patch_rad-1,patch_rad)
                temp4 = img2Dshift(sd,patch_rad,-patch_rad-1)
                res = temp1 + temp2 - temp3 - temp4
                sqdist1 = res[patch_rad:patch_rad+height,patch_rad:patch_rad+width]
                w = hat * np.exp(-sqdist1/(sigma**2))
                W0_pad = np.pad(W0,window_rad,mode='symmetric')
                W0_shift = img2Dshift(W0_pad,dx,dy)
                W0_temp = W0_shift[window_rad:window_rad+height,window_rad:window_rad+width]
                w2 = (alpha*w) / (np.sqrt(W0)*np.sqrt(W0_temp))
                v = padded_v[window_rad+dx:window_rad+dx+height,window_rad+dy:window_rad+dy+width]
                u = u + w2*v
                W2 = W2 + w2
    
    u = u + (1-W2)*noisy_img
    return u        # u = Denoised image


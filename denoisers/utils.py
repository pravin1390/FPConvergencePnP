# Utility functions required for denoisers

import numpy as np

def integral_img_sq_diff(v,dx,dy):
    t = img2Dshift(v,dx,dy)
    diff = (v-t)**2
    sd = np.cumsum(diff,axis=0)
    sd = np.cumsum(sd,axis=1)
    return(sd,diff,t)


def triangle(dx,dy,Ns):
    r1 = np.abs(1 - np.abs(dx)/(Ns+1))
    r2 = np.abs(1 - np.abs(dy)/(Ns+1))
    return r1*r2


def img2Dshift(v,dx,dy):
    row,col = v.shape
    t = np.zeros((row,col))
    typ = (1 if dx>0 else 0)*2 + (1 if dy>0 else 0)
    if(typ==0):
        t[-dx:,-dy:] = v[0:row+dx,0:col+dy]
    elif(typ==1):
        t[-dx:,0:col-dy] = v[0:row+dx,dy:]
    elif(typ==2):
        t[0:row-dx,-dy:] = v[dx:,0:col+dy]
    elif(typ==3):
        t[0:row-dx,0:col-dy] = v[dx:,dy:]
    return t


#pseudopermanent 1D linear solution with conditions pressure-null flow

import numpy as np

def radial_pseudo_1d(Po,qw,h,k,re,rw,theta,ct,u,t_final):

    t=np.linspace(0,t_final,100)
    r=np.linspace(rw,re,100)

    R,T=np.meshgrid(r,t)

    arg1=(2*k*T)/(theta*ct*u*(re**2))
    arg2=np.log(R/rw)
    arg3=(1/2)*((R/re)**2)
    arg4=np.log(re/rw)
    fator=(qw*u)/(2*np.pi*h*k)

    P=Po-fator*(arg1- arg2 + arg3 + arg4- (3/4))

    return r,t,R,T,P

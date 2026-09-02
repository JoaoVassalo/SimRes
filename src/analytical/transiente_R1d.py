#transient 1D radial solution with conditions flow-pressure
import numpy as np
from scipy.special import exp1

def radial_trans_1d_FP(p0, qw,h, k, phi, ct, mu, t_final, rw, re):

    t = np.linspace(0, t_final, 100)
    r = np.linspace(rw, re, 101)

    R, T = np.meshgrid(r,t)

    P=[]

    P[0, :] = p0
    
    T_pos = T[1:, :]
    R_pos = R[1:, :]
    
    P[1:, :] = p0 - (qw * mu)/(4 * np.pi * k * h) * exp1((phi * mu * ct * R_pos**2)/(4 * k * T_pos))

    return r, t, R, T, P
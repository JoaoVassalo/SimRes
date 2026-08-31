#transient 1D radial solution with conditions flow-pressure
import numpy as np
from scipy.special import exp1

def linear_trans_1d_PP(p0, qw, L, h, k, phi, ct, mu, t_final):

    t = np.linspace(0, t_final, 100)
    r = np.linspace(0, L, 100)

    R, T = np.meshgrid(r,t)


    P = p0 - (qw * mu)/(4 * np.pi * k * h) * exp1((phi * mu * ct * r**2)/(4 * k * T))

    return r, t, R, T, P
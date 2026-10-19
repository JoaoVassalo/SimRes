# Soluções analíticas radiais 1D.
import numpy as np
from scipy.special import exp1


def radial_trans_1d_FP(p0, qw,h, k, phi, ct, mu, t_final, rw, re):

    t = np.linspace(0, t_final, 100)
    r = np.linspace(rw, re, 101)

    R, T = np.meshgrid(r,t)

    P = np.empty_like(R)
    P[0, :] = p0
    
    T_pos = T[1:, :]
    R_pos = R[1:, :]
    
    P[1:, :] = p0 - (qw * mu)/(4 * np.pi * k * h) * exp1((phi * mu * ct * R_pos**2)/(4 * k * T_pos))

    return r, t, R, T, P


def radial_pseudo_1d(p0, qw, h, k, re, rw, phi, ct, mu, t_final):

    t = np.linspace( 0, t_final, 100 )
    r = np.linspace( rw, re, 101 )

    R, T = np.meshgrid( r, t )

    fator = ( qw  * mu ) / (2 * np.pi* k * h ) 
    arg1 = ( 2 *  k * T )/( phi * mu * ct * ( re**2 ) )
    arg2 = np.log( R / rw )
    arg3 = ( 1 / 2 ) * ( ( R / re )**2 )
    arg4 = np.log( re / rw )

    P = p0 - fator * ( arg1 - arg2 + arg3 + arg4 - ( 3 / 4 ))

    return r, t, R, T, P
#pseudopermanent 1D linear solution with conditions pressure-null flow

import numpy as np

def radial_pseudo_1d(p0, qw, h, k, re, rw, phi, ct, mu, B, t_final, C1, C2):

    t = np.linspace( 0, t_final, 100 )
    r = np.linspace( rw, re, 100 )

    R, T = np.meshgrid( r, t )

    fator = ( C2 * qw * B * mu ) / ( k * h )
    arg1 = ( 2 * C1 * k * T )/( phi * mu * ct * ( re**2 ) )
    arg2 = np.log( R / rw )
    arg3 = ( 1 / 2 ) * ( ( R / re )**2 )
    arg4 = np.log( re / rw )

    P = p0 - fator * ( arg1 - arg2 + arg3 + arg4 - ( 3 / 4 ))

    return r, t, R, T, P
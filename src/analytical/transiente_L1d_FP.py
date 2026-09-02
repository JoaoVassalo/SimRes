#transient 1D linear solution with conditions flow rate-pressure
import numpy as np
from scipy.special import erfc



def linear_trans_1d_FP(p0, qw, L, k, A, mu, eta, t_final):

    t=np.linspace(0,t_final,100)
    x=np.linspace(0,L,101)

    X,T=np.meshgrid(x,t)

    P = np.empty_like(X)

    P[0, :] = p0

    T_pos = T[1:, :]
    X_pos = X[1:, :]

    P[1:, :] = p0 - (qw * mu * L)/(k * A) * (np.sqrt((4 * eta * T_pos)/(np.pi * L**2)) * np.exp((-X_pos**2)/(4 * eta * T_pos)) - (X_pos / L) * erfc(X_pos / np.sqrt(4 * eta * T_pos)))

    print(P[1, 0] / 1e6)    # x = 0, primeiro instante > 0
    print(P[1, -1] / 1e6)   # x = L, primeiro instante > 0

    return x,t,X,T,P
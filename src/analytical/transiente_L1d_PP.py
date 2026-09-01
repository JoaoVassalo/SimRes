#transient 1D linear solution with conditions pressure-pressure
import numpy as np

def linear_trans_1d_PP(p0, Pw, L, k, phi, ct, mu, t_final, n_modos):

    t=np.linspace(0,t_final,100)
    x=np.linspace(0,L,101)

    X,T=np.meshgrid(x,t)

    serie=0

    for n in range(1, n_modos+1):
        arg=((np.exp(-(((n*np.pi)/L)**2)*(k/(phi*mu*ct))*T))/n)*np.sin((np.pi*X*n/L))

        serie+=arg

    P=(p0-Pw)*((X/L)+(2/np.pi)*serie)+Pw
    P[0, :] = p0

    return x,t,X,T,P
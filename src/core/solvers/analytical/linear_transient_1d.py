import numpy as np
import matplotlib.pyplot as plt

#parameters
Pw=120
Pe=100
L=1000
k=0.3
theta=0.2
ct=0.000001
u=2
t_final=1000
n_modos=100

def linear_trans_1d(Pe,Pw,L,k,theta,ct,u,t_final,n_modos):

    t=np.linspace(0,t_final,100)
    x=np.linspace(0,L,100)

    X,T=np.meshgrid(x,t)

    serie=0

    for n in range(1, n_modos+1):
        arg=((np.exp(-(((n*np.pi)/L)**2)*(k/(theta*u*ct))*T))/n)*np.sin((np.pi*X*n/L))

        serie+=arg

    P=(Pe-Pw)*((X/L)+(2/np.pi)*serie)+Pw

    return x,t,X,T,P

x,t,X,T,P=linear_trans_1d(
    Pe,
    Pw,
    L,
    k,
    theta,
    ct,
    u,
    t_final,
    n_modos
)
# P em função de x
t_escolhido=500

indice_t=np.argmin(np.abs(t-t_escolhido))

plt.plot(x,P[indice_t,:])

plt.xlabel('x')
plt.ylabel('P')
plt.title(f'Pressão em função de x - t={t[indice_t]:.2f}')

plt.grid()

plt.show()

# P em função de t
x_escolhido=500

indice_x=np.argmin(np.abs(x-x_escolhido))

plt.plot(t,P[:,indice_x])

plt.xlabel('t')
plt.ylabel('P')
plt.title(f'Pressão em função de t - x={x[indice_x]:.2f}')

plt.grid()

plt.show()
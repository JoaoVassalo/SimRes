import numpy as np


def explicita_1d_FP(Pi, Pe, qw, h, L, k, phi, ct, mu, t_final, nx, nt):

    A = h * L
    # malha espacial
    dx = L / nx
    x = np.linspace(dx / 2, L - dx / 2, nx)

    # coeficiente de difusividade hidraulica
    eta = k / (phi * ct * mu)

    # malha temporal
    dt = t_final / nt
    t = np.linspace(0, t_final, nt + 1)

    rx = dt / dx**2
    lambd = eta * rx

    # condição de estabilidade
    if lambd > 0.5:
        print("A formulação explícita não satisfaz a condição de estabilidade.")

    # condição inicial
    P_old = np.ones(nx) * Pi

    # armazenamento
    P = np.zeros((nt + 1, nx))
    P[0, :] = P_old

    # loop temporal
    for n in range(nt):

        P_new = np.zeros(nx)

        # fronteira esquerda - fluxo prescrito
        P_new[0] = ((1 - lambd) * P_old[0] + lambd * P_old[1] + lambd *(qw*mu*dx)/(k*A))

        # nós internos - diferença centrada
        for i in range(1, nx - 1):

            P_new[i] = (lambd * P_old[i - 1] + (1 - 2 * lambd) * P_old[i] + lambd * P_old[i + 1])

        # fronteira direita - pressão prescrita
        P_new[-1] = ((1 - 4*lambd) * P_old[-1] + (4/3)*lambd * P_old[-2] +(8/3)*lambd *Pe)
        P[n + 1, :] = P_new

        P_old = P_new.copy()

    return x, t, P, lambd

def explicita_1d_PP(p0, Pw, h, L, k, phi, ct, mu, t_final, nx, nt):

    A = h * L
    # malha espacial
    dx = L / nx
    x = np.linspace(dx / 2, L - dx / 2, nx)

    # coeficiente de difusividade hidraulica
    eta = k / (phi * ct * mu)

    # malha temporal
    dt = t_final / nt
    t = np.linspace(0, t_final, nt + 1)

    rx = dt / dx**2
    lambd = eta * rx

    # condição de estabilidade
    if lambd > 0.5:
        print("A formulação explícita não satisfaz a condição de estabilidade.")

    # condição inicial
    P_old = np.ones(nx) * p0

    # armazenamento
    P = np.zeros((nt + 1, nx))
    P[0, :] = P_old

    # loop temporal
    for n in range(nt):

        P_new = np.zeros(nx)

        # fronteira esquerda - fluxo prescrito
        P_new[0] = ((1 - 4*lambd) * P_old[0] +(4/3)*lambd * P_old[1]+(8/3)*lambd *Pw)
        # nós internos - diferença centrada
        for i in range(1, nx - 1):

            P_new[i] = (lambd * P_old[i - 1] + (1 - 2 * lambd) * P_old[i] + lambd * P_old[i + 1])

        # fronteira direita - pressão prescrita
        P_new[-1] = ((1 - 4*lambd) * P_old[-1]+ (4/3)*lambd * P_old[-2]+(8/3)*lambd *p0)

        P[n + 1, :] = P_new

        P_old = P_new.copy()

    return x, t, P, lambd
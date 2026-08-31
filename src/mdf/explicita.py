import numpy as np


def explicita_1d_PF(
    Pi,
    Pw,
    qw,
    h,
    L,
    largura,
    k,
    phi,
    ct,
    mu,
    t_final,
    nx,
    nt,
):

    A = h * largura

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
    else:
        print("A condição de estabilidade foi satisfeita.")

    # condição inicial
    P_old = np.ones(nx) * Pi

    # armazenamento
    P = np.zeros((nt + 1, nx))
    P[0, :] = P_old

    # loop temporal
    for n in range(nt):

        P_new = np.zeros(nx)

        # fronteira esquerda - pressão prescrita
        P_new[0] = (
            (1 - 4 * lambd) * P_old[0]
            + (4 / 3) * lambd * P_old[1]
            + (8 / 3) * lambd * Pw
        )

        # nós internos - diferença centrada
        for i in range(1, nx - 1):

            P_new[i] = (
                lambd * P_old[i - 1]
                + (1 - 2 * lambd) * P_old[i]
                + lambd * P_old[i + 1]
            )

        # fronteira direita - fluxo prescrito
        P_new[-1] = (
            (1 - lambd) * P_old[-1]
            + lambd * P_old[-2]
            + lambd * (qw * mu * dx) / (k * A)
        )

        P[n + 1, :] = P_new

        P_old = P_new.copy()

    return x, t, P, lambd
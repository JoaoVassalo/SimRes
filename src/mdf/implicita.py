import numpy as np
from scipy.linalg import solve

from core.solvers.linear.TDMA import TDMA
from core.solvers.linear.gauss import gauss_pivotamento
from core.solvers.linear.jacobi import jacobi

# código inspirado no que foi feito em métodos numéricos 2


# implícita 1d - fluxo-pressão
def implicita_1d_FP(
    p0,
    Pe,
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
    metodo="TDMA",
):

    A = h * largura

    # malha espacial
    dx = L / nx
    x = np.linspace(dx / 2, L - dx / 2, nx)

    # difusividade
    eta = k / (phi * ct * mu)

    # malha temporal
    dt = t_final / nt
    t = np.linspace(0, t_final, nt + 1)

    rx = dt / dx**2
    lambd = eta * rx

    # condição inicial
    P_old = np.ones(nx) * p0

    # armazenamento
    P = np.zeros((nt + 1, nx))
    P[0, :] = P_old

    iteracoes_jacobi = []

    # loop temporal
    for n in range(nt):

        matriz = np.zeros((nx, nx))
        D = np.zeros(nx)

        # esquerda - fluxo
        matriz[0, 0] = 1 + lambd
        matriz[0, 1] = -lambd

        D[0] = (
            P_old[0]
            + lambd * (qw * mu * dx) / (k * A)
        )

        # nós internos
        for i in range(1, nx - 1):

            matriz[i, i - 1] = -lambd
            matriz[i, i] = 1 + 2 * lambd
            matriz[i, i + 1] = -lambd

            D[i] = P_old[i]

        # direita - pressão
        matriz[-1, -2] = -(4 / 3) * lambd
        matriz[-1, -1] = 1 + 4 * lambd

        D[-1] = (
            P_old[-1]
            + (8 / 3) * lambd * Pe
        )

        # solução do sistema linear
        if metodo == "TDMA":

            P_new = TDMA(matriz, D)

        elif metodo == "Gauss":

            P_new = gauss_pivotamento(matriz, D)

        elif metodo == "Jacobi":

            x0 = P_old

            P_new, iteracoes = jacobi(
                matriz,
                D,
                x0,
                1e-7
            )

            iteracoes_jacobi.append(iteracoes)

        elif metodo == "Scipy":

            P_new = solve(matriz, D)

        else:

            raise ValueError(
                "Método deve ser TDMA, Gauss, Jacobi ou Scipy."
            )

        P[n + 1, :] = P_new
        P_old = P_new.copy()

    if metodo == "Jacobi":
        print(
            "Média de iterações do Jacobi =",
            np.mean(iteracoes_jacobi)
        )

    return x, t, P, lambd


# implícita 1d - pressão-pressão
def implicita_1d_PP(
    p0,
    Pw,
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
    metodo="TDMA",
):

    Pe = p0

    # malha espacial
    dx = L / nx
    x = np.linspace(dx / 2, L - dx / 2, nx)

    # difusividade
    eta = k / (phi * ct * mu)

    # malha temporal
    dt = t_final / nt
    t = np.linspace(0, t_final, nt + 1)

    rx = dt / dx**2
    lambd = eta * rx

    # condição inicial
    P_old = np.ones(nx) * p0

    # armazenamento
    P = np.zeros((nt + 1, nx))
    P[0, :] = P_old

    iteracoes_jacobi = []

    # loop temporal
    for n in range(nt):

        matriz = np.zeros((nx, nx))
        D = np.zeros(nx)

        # esquerda - pressão
        matriz[0, 0] = 1 + 4 * lambd
        matriz[0, 1] = -(4 / 3) * lambd

        D[0] = (
            P_old[0]
            + (8 / 3) * lambd * Pw
        )

        # nós internos
        for i in range(1, nx - 1):

            matriz[i, i - 1] = -lambd
            matriz[i, i] = 1 + 2 * lambd
            matriz[i, i + 1] = -lambd

            D[i] = P_old[i]

        # direita - pressão
        matriz[-1, -2] = -(4 / 3) * lambd
        matriz[-1, -1] = 1 + 4 * lambd

        D[-1] = (
            P_old[-1]
            + (8 / 3) * lambd * Pe
        )

        # solução do sistema linear
        if metodo == "TDMA":

            P_new = TDMA(matriz, D)

        elif metodo == "Gauss":

            P_new = gauss_pivotamento(matriz, D)

        elif metodo == "Jacobi":

            x0 = P_old

            P_new, iteracoes = jacobi(
                matriz,
                D,
                x0,
                1e-7
            )

            iteracoes_jacobi.append(iteracoes)

        elif metodo == "Scipy":

            P_new = solve(matriz, D)

        else:

            raise ValueError(
                "Método deve ser TDMA, Gauss, Jacobi ou Scipy."
            )

        P[n + 1, :] = P_new
        P_old = P_new.copy()

    if metodo == "Jacobi":
        print(
            "Média de iterações do Jacobi =",
            np.mean(iteracoes_jacobi)
        )

    return x, t, P, lambd
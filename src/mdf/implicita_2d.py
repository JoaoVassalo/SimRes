import numpy as np
from scipy.linalg import solve

from mdf.peaceman import (
    well_index,
    vazao_pressao_prescrita,
    raio_equivalente,
    termos_pressao_prescrita
)


def implicita_2d(p0, pwf, rw, h,Lx, Ly, k, phi, ct, mu,t_final, nx, ny, nt, iw=None, jw=None):
    # malha espacial
    dx = Lx / nx
    dy = Ly / ny
    x = np.linspace(dx / 2, Lx - dx / 2, nx)
    y = np.linspace(dy / 2, Ly - dy / 2, ny)

    n_blocos = nx * ny
    volume = dx * dy * h

    # parâmetros
    beta = 1 / (phi * ct * mu)

    dt = t_final / nt
    t = np.linspace(0, t_final, nt + 1)
    rx = dt / dx**2
    ry = dt / dy**2

    # poço do poço
    if iw is None:
        iw = nx // 2

    if jw is None:
        jw = ny // 2

    pw = iw + nx * jw

    #Peaceman
    
    req = raio_equivalente(dx)
    WI = well_index(req, rw, h)

    # condição inicial
    P_old = np.ones(n_blocos) * p0

    # armazenamento
    P = np.zeros((nt + 1, n_blocos))
    P[0, :] = P_old

    q_poco = np.zeros(nt + 1)

    # loop temporal
    for n in range(nt):

        matriz = np.zeros((n_blocos, n_blocos))
        D = np.zeros(n_blocos)

        for j in range(ny):

            for i in range(nx):

                p = i + nx * j

                matriz[p, p] = 1.0
                D[p] = P_old[p]

                # oeste
                if i > 0:
                    W = p - 1
                    kW = 2 / (1 / k[j, i] + 1 / k[j, i - 1])
                    matriz[p, p] += beta * rx * kW
                    matriz[p, W] = -beta * rx * kW

                # leste
                if i < nx - 1:
                    E = p + 1
                    kE = 2 / ( 1 / k[j, i] + 1 / k[j, i + 1])
                    matriz[p, p] += beta * rx * kE
                    matriz[p, E] = -beta * rx * kE

                # inferior
                if j > 0:
                    S = p - nx
                    kS = 2 / (1 / k[j, i] + 1 / k[j - 1, i])
                    matriz[p, p] += beta * ry * kS
                    matriz[p, S] = -beta * ry * kS

                # superior
                if j < ny - 1:
                    N = p + nx
                    kN = 2 / (1 / k[j, i] + 1 / k[j + 1, i])
                    matriz[p, p] += beta * ry * kN
                    matriz[p, N] = -beta * ry * kN

                # poço produtor
                if i == iw and j == jw:
                    termo_diag, termo_fonte = termos_pressao_prescrita(
                        pwf,k[j, i],mu,
                        WI,dt,phi,ct,volume)
                    matriz[p, p] += termo_diag
                    D[p] += termo_fonte

        # solução do sistema
        P_new = solve(matriz, D)

        # vazão do poço
        q_poco[n + 1] = vazao_pressao_prescrita(pwf,P_new[pw],k[jw, iw],mu,WI)

        P[n + 1, :] = P_new
        P_old = P_new.copy()

    P_2d = P.reshape(nt + 1, ny, nx)

    return x, y, t, P_2d, q_poco
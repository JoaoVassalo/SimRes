import numpy as np


def calcular_velocidade(P, k, mu, dx, dy):

    nt, ny, nx = P.shape

    Vx = np.zeros_like(P)
    Vy = np.zeros_like(P)

    for n in range(nt):

        for j in range(ny):

            for i in range(nx):

                # direção x

                if i > 0:

                    kW = 2 / (1 / k[j, i] + 1 / k[j, i - 1])
                    vW = -(kW / mu) * (P[n, j, i] - P[n, j, i - 1]) / dx

                else:
                    vW = 0.0


                if i < nx - 1:

                    kE = 2 / (1 / k[j, i] + 1 / k[j, i + 1])
                    vE = -(kE / mu) * (P[n, j, i + 1]- P[n, j, i]) / dx

                else:
                    vE = 0.0

                Vx[n, j, i] = (vW + vE) / 2


                # direção y

                if j > 0:

                    kS = 2 / (1 / k[j, i] + 1 / k[j - 1, i])
                    vS = -(kS / mu) * (P[n, j, i]- P[n, j - 1, i]) / dy

                else:
                    vS = 0.0


                if j < ny - 1:

                    kN = 2 / (1 / k[j, i]+ 1 / k[j + 1, i])
                    vN = -(kN / mu) * (P[n, j + 1, i] - P[n, j, i]) / dy

                else:
                    vN = 0.0

                Vy[n, j, i] = (vS + vN) / 2


    V = np.sqrt(Vx**2 + Vy**2)

    return Vx, Vy, V
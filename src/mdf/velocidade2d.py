import numpy as np


def calcular_velocidade(P, k, mu, dx, dy, internal_no_flow_faces=None):

    nt, ny, nx = P.shape

    Vx = np.zeros_like(P)
    Vy = np.zeros_like(P)
    blocked_x = set()
    blocked_y = set()
    if internal_no_flow_faces is not None:
        for first, second in internal_no_flow_faces:
            j1, i1 = divmod(first, nx)
            j2, i2 = divmod(second, nx)
            if j1 == j2 and abs(i1 - i2) == 1:
                blocked_x.add((j1, min(i1, i2)))
            elif i1 == i2 and abs(j1 - j2) == 1:
                blocked_y.add((min(j1, j2), i1))
            else:
                raise ValueError("As faces internas devem ligar células vizinhas")

    for n in range(nt):

        for j in range(ny):

            for i in range(nx):

                # direção x

                if i > 0 and (j, i - 1) not in blocked_x:

                    kW = 2 / (1 / k[j, i] + 1 / k[j, i - 1])
                    vW = -(kW / mu) * (P[n, j, i] - P[n, j, i - 1]) / dx

                else:
                    vW = 0.0


                if i < nx - 1 and (j, i) not in blocked_x:

                    kE = 2 / (1 / k[j, i] + 1 / k[j, i + 1])
                    vE = -(kE / mu) * (P[n, j, i + 1]- P[n, j, i]) / dx

                else:
                    vE = 0.0

                Vx[n, j, i] = (vW + vE) / 2


                # direção y

                if j > 0 and (j - 1, i) not in blocked_y:

                    kS = 2 / (1 / k[j, i] + 1 / k[j - 1, i])
                    vS = -(kS / mu) * (P[n, j, i]- P[n, j - 1, i]) / dy

                else:
                    vS = 0.0


                if j < ny - 1 and (j, i) not in blocked_y:

                    kN = 2 / (1 / k[j, i]+ 1 / k[j + 1, i])
                    vN = -(kN / mu) * (P[n, j + 1, i] - P[n, j, i]) / dy

                else:
                    vN = 0.0

                Vy[n, j, i] = (vS + vN) / 2

    for j, i in blocked_x:
        Vx[:, j, i] = 0.0
        Vx[:, j, i + 1] = 0.0
    for j, i in blocked_y:
        Vy[:, j, i] = 0.0
        Vy[:, j + 1, i] = 0.0

    V = np.sqrt(Vx**2 + Vy**2)

    return Vx, Vy, V

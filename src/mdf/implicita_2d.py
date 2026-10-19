import numpy as np
from scipy.linalg import solve

from mdf.peaceman import (
    well_index,
    vazao_pressao_prescrita,
    raio_equivalente,
    termos_pressao_prescrita
)
from mdf.boundary_conditions import boundary_condition, create_boundary_mask


def implicita_2d(p0, pwf, rw, h, Lx, Ly, k, phi, ct, mu, t_final, nx, ny, nt,
    iw=None, jw=None, boundary_conditions=None, boundary_values=None, internal_no_flow_faces=None,
    injection_wells=None,):
    """Resolve a pressão 2D totalmente implícita.

    ``boundary_values`` recebe pressão em faces ``pressure`` e vazão por face
    em ``flow``; vazão positiva entra no domínio e ``q=0`` representa fluxo nulo.
    """

    # malha espacial
    dx = Lx / nx
    dy = Ly / ny
    x = np.linspace(dx / 2, Lx - dx / 2, nx)
    y = np.linspace(dy / 2, Ly - dy / 2, ny)

    n_blocos = nx * ny
    volume = dx * dy * h
    if boundary_conditions is None:
        boundary_conditions = {}
    if boundary_values is None:
        boundary_values = {}
    if injection_wells is None:
        injection_wells = []

    # parâmetros
    beta = 1 / (phi * ct * mu)

    dt = t_final / nt
    t = np.linspace(0, t_final, nt + 1)
    rx = dt / dx**2
    ry = dt / dy**2
    flow_coefficient = dt / (phi * ct * volume)
    boundary_mask, boundary_value_mask = create_boundary_mask(nx, ny, boundary_conditions, boundary_values, internal_no_flow_faces)
    faces = (("left", 0, -1, 0, rx), ("right", 1, 1, 0, rx), ("bottom", 2, 0, -1, ry), ("top", 3, 0, 1, ry))

    # posição do poço
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

                for side, face, di, dj, r in faces:
                    condition = boundary_mask[j, i, face]
                    if condition == "internal":
                        neighbor = p + di + nx * dj
                        face_k = 2 / (1 / k[j, i] + 1 / k[j + dj, i + di])
                        transmissibility = beta * r * face_k
                        matriz[p, p] += transmissibility
                        matriz[p, neighbor] = -transmissibility
                    else:
                        inward_neighbor = p - di - nx * dj
                        coefficient = beta * r * k[j, i] if condition == "pressure" else flow_coefficient
                        boundary_condition(matriz, D, (p, inward_neighbor), condition, coefficient, boundary_value_mask[j, i, face])

                # poço produtor
                if i == iw and j == jw:
                    termo_diag, termo_fonte = termos_pressao_prescrita(
                        pwf,k[j, i],mu,
                        WI,dt,phi,ct,volume)
                    matriz[p, p] += termo_diag
                    D[p] += termo_fonte

                for injection_well in injection_wells:
                    if i == injection_well["i"] and j == injection_well["j"]:
                        D[p] += injection_well["rate"] * flow_coefficient

        # solução do sistema
        P_new = solve(matriz, D)

        # vazão do poço
        q_poco[n + 1] = vazao_pressao_prescrita(pwf,P_new[pw],k[jw, iw],mu,WI)

        P[n + 1, :] = P_new
        P_old = P_new.copy()

    P_2d = P.reshape(nt + 1, ny, nx)

    return x, y, t, P_2d, q_poco

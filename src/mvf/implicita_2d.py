import numpy as np
from scipy.sparse import lil_matrix

from core.peaceman import raio_equivalente, vazao_pressao_prescrita, well_index
from core.solvers.linear.gauss import fatorar_gauss, resolver_gauss
from mvf.boundary_conditions import aplicar_condicoes_contorno


def implicita_2d(p0, pwf, rw, h, Lx, Ly, k, phi, ct, mu, t_final, nx, ny, nt,
    iw=None, jw=None, boundary_conditions=None, boundary_values=None,
    internal_no_flow_faces=None, injection_wells=None):
    k = np.asarray(k, dtype=float)
    if k.shape != (ny, nx) or not np.all(np.isfinite(k)) or np.any(k <= 0):
        print("k deve ser um campo finito e positivo com forma (ny, nx)")
        return
    if nx < 1 or ny < 1 or nt < 1:
        print("nx, ny e nt devem ser positivos")
        return
    values = (p0, pwf, rw, h, Lx, Ly, phi, ct, mu, t_final)
    if not np.all(np.isfinite(values)):
        print("os parâmetros do modelo devem ser finitos")
        return
    if min(rw, h, Lx, Ly, phi, ct, mu, t_final) <= 0:
        print("dimensões, propriedades, duração e rw devem ser positivos")
        return

    dx = Lx / nx
    dy = Ly / ny
    volume = dx * dy * h
    dt = t_final / nt
    x = (np.arange(nx) + 0.5) * dx
    y = (np.arange(ny) + 0.5) * dy
    t = np.linspace(0.0, t_final, nt + 1)
    size = nx * ny

    if iw is None:
        iw = nx // 2
    if jw is None:
        jw = ny // 2
    if not (0 <= iw < nx and 0 <= jw < ny):
        print("posição do poço produtor fora da malha")
        return

    boundary_conditions = boundary_conditions or {}
    boundary_values = boundary_values or {}
    injection_wells = injection_wells or []

    blocked = set()
    for first, second in internal_no_flow_faces or []:
        if not (0 <= first < size and 0 <= second < size):
            print("índice de face interna fora da malha")
            return
        r1, c1 = divmod(first, nx)
        r2, c2 = divmod(second, nx)
        if abs(r1 - r2) + abs(c1 - c2) != 1:
            print("as faces internas devem ligar células vizinhas")
            return
        blocked.add(tuple(sorted((first, second))))

    matrix = lil_matrix((size, size), dtype=float)
    storage = phi * ct * volume / dt
    matrix.setdiag(np.full(size, storage))

    def add_connection(i1, j1, i2, j2, area, distance):
        p1 = i1 + nx * j1
        p2 = i2 + nx * j2
        if tuple(sorted((p1, p2))) in blocked:
            return
        k_face = 2.0 * k[j1, i1] * k[j2, i2] / (k[j1, i1] + k[j2, i2])
        transmissibility = k_face * area / (mu * distance)
        matrix[p1, p1] += transmissibility
        matrix[p2, p2] += transmissibility
        matrix[p1, p2] -= transmissibility
        matrix[p2, p1] -= transmissibility

    for j in range(ny):
        for i in range(nx):
            if i + 1 < nx:
                add_connection(i, j, i + 1, j, h * dy, dx)
            if j + 1 < ny:
                add_connection(i, j, i, j + 1, h * dx, dy)

    rhs_boundary = np.zeros(size)
    if not aplicar_condicoes_contorno(
        matrix, rhs_boundary, k, mu, dx, dy, h, nx, ny,
        boundary_conditions, boundary_values,
    ):
        return

    req = raio_equivalente(dx)
    wi = well_index(req, rw, h)
    well_index_factor = k[jw, iw] * wi / mu
    producer = iw + nx * jw
    matrix[producer, producer] += well_index_factor
    producer_rhs = well_index_factor * pwf

    for well in injection_wells:
        i, j, rate = well["i"], well["j"], well["rate"]
        if not (0 <= i < nx and 0 <= j < ny):
            print("posição de injetor fora da malha")
            return
        rhs_boundary[i + nx * j] += rate

    matrix = matrix.toarray()
    fatoracao = fatorar_gauss(matrix)
    if fatoracao is None:
        print("falha ao montar o sistema de pressões")
        return
    pressure_old = np.full(size, p0, dtype=float)
    pressure = np.empty((nt + 1, ny, nx), dtype=float)
    pressure[0] = pressure_old.reshape(ny, nx)
    q_producer = np.zeros(nt + 1)

    for step in range(nt):
        rhs = storage * pressure_old + rhs_boundary
        rhs[producer] += producer_rhs
        pressure_new = resolver_gauss(fatoracao, rhs)
        if pressure_new is None:
            print("falha ao resolver o sistema de pressões")
            return
        pressure_new = np.asarray(pressure_new, dtype=float).reshape(-1)
        if not np.all(np.isfinite(pressure_new)):
            print("a solução de pressão contém valores não finitos")
            return
        pressure[step + 1] = pressure_new.reshape(ny, nx)
        q_producer[step + 1] = vazao_pressao_prescrita(
            pwf, pressure_new[producer], k[jw, iw], mu, wi
        )
        pressure_old = pressure_new

    return x, y, t, pressure, q_producer

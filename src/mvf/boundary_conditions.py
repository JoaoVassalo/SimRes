import numpy as np


def aplicar_condicoes_contorno(matrix, rhs, k, mu, dx, dy, h, nx, ny,
    boundary_conditions=None, boundary_values=None):
    boundary_conditions = boundary_conditions or {}
    boundary_values = boundary_values or {}
    sides = {"left", "right", "bottom", "top"}
    if not set(boundary_conditions).issubset(sides):
        print("fronteira desconhecida")
        return False
    if any(kind not in {"pressure", "flow"} for kind in boundary_conditions.values()):
        print("as condições de fronteira devem ser 'pressure' ou 'flow'")
        return False
    for side, kind in boundary_conditions.items():
        if kind == "pressure" and side not in boundary_values:
            print(f"informe a pressão prescrita para {side}")
            return False
    if any(not np.isfinite(value) for value in boundary_values.values()):
        print("os valores de fronteira devem ser finitos")
        return False

    faces = (
        ("left", [(0, j, h * dy, dx / 2) for j in range(ny)]),
        ("right", [(nx - 1, j, h * dy, dx / 2) for j in range(ny)]),
        ("bottom", [(i, 0, h * dx, dy / 2) for i in range(nx)]),
        ("top", [(i, ny - 1, h * dx, dy / 2) for i in range(nx)]),
    )
    for side, side_faces in faces:
        kind = boundary_conditions.get(side, "flow")
        value = boundary_values.get(side, 0.0)
        for i, j, area, distance in side_faces:
            cell = i + nx * j
            if kind == "pressure":
                transmissibility = k[j, i] * area / (mu * distance)
                matrix[cell, cell] += transmissibility
                rhs[cell] += transmissibility * value
            elif kind == "flow":
                rhs[cell] += value
    return True

import numpy as np


def create_boundary_mask(nx, ny, boundary_conditions=None, boundary_values=None, internal_no_flow_faces=None):
    if boundary_conditions is None:
        boundary_conditions = {}
    if boundary_values is None:
        boundary_values = {}

    mask = np.full((ny, nx, 4), "internal", dtype=object)
    values = np.zeros((ny, nx, 4))
    sides = {"left": 0, "right": 1, "bottom": 2, "top": 3}

    for side, condition in boundary_conditions.items():
        if side not in sides:
            print(f"Borda desconhecida: {side}")
            return
        if condition != "pressure" and condition != "flow":
            print("condition deve ser 'pressure' ou 'flow'")
            return
        if condition == "pressure" and side not in boundary_values:
            print(f"Informe a pressão prescrita para a borda {side}")
            return

    mask[:, 0, sides["left"]] = boundary_conditions.get("left", "flow")
    mask[:, -1, sides["right"]] = boundary_conditions.get("right", "flow")
    mask[0, :, sides["bottom"]] = boundary_conditions.get("bottom", "flow")
    mask[-1, :, sides["top"]] = boundary_conditions.get("top", "flow")
    values[:, 0, sides["left"]] = boundary_values.get("left", 0.0)
    values[:, -1, sides["right"]] = boundary_values.get("right", 0.0)
    values[0, :, sides["bottom"]] = boundary_values.get("bottom", 0.0)
    values[-1, :, sides["top"]] = boundary_values.get("top", 0.0)

    if internal_no_flow_faces is not None:
        for first, second in internal_no_flow_faces:
            j1, i1 = divmod(first, nx)
            j2, i2 = divmod(second, nx)
            if not (0 <= i1 < nx and 0 <= i2 < nx and 0 <= j1 < ny and 0 <= j2 < ny):
                print("Índice de face interna fora da malha")
                return
            if j1 == j2 and abs(i1 - i2) == 1:
                mask[j1, min(i1, i2), sides["right"]] = "flow"
                mask[j1, max(i1, i2), sides["left"]] = "flow"
                values[j1, min(i1, i2), sides["right"]] = 0.0
                values[j1, max(i1, i2), sides["left"]] = 0.0
            elif i1 == i2 and abs(j1 - j2) == 1:
                mask[min(j1, j2), i1, sides["top"]] = "flow"
                mask[max(j1, j2), i1, sides["bottom"]] = "flow"
                values[min(j1, j2), i1, sides["top"]] = 0.0
                values[max(j1, j2), i1, sides["bottom"]] = 0.0
            else:
                print("As faces internas devem ligar células vizinhas")
                return

    return mask, values


def boundary_condition(matrix, rhs, index, condition, coefficient, value=0.0):
    """Aplica pressão prescrita ou vazão por face no balanço heterogêneo.

    Para ``flow``, ``coefficient`` é ``dt / (phi * ct * volume)`` e ``value``
    é a vazão volumétrica na face, positiva quando entra no reservatório.
    """
    cell, inward_neighbor = index
    if condition == "pressure":
        matrix[cell, cell] += 4 * coefficient
        matrix[cell, inward_neighbor] -= 4 * coefficient / 3
        rhs[cell] += 8 * coefficient * value / 3
    elif condition == "flow":
        rhs[cell] += coefficient * value
    else:
        print("condition deve ser 'pressure' ou 'flow'")

import numpy as np


def calcular_velocidade(P, k, mu, dx, dy, internal_no_flow_faces=None,
    boundary_conditions=None, boundary_values=None, h=1.0):
    P = np.asarray(P, dtype=float)
    k = np.asarray(k, dtype=float)
    if P.ndim != 3 or k.shape != P.shape[1:]:
        print("P deve ter forma (nt, ny, nx) e k deve ter forma (ny, nx)")
        return
    if not np.all(np.isfinite(P)) or not np.all(np.isfinite(k)) or np.any(k <= 0):
        print("P e k devem conter valores finitos, e k deve ser positivo")
        return
    if not np.isfinite(mu) or not np.isfinite(dx) or not np.isfinite(dy) or not np.isfinite(h):
        print("mu, dx, dy e h devem ser finitos")
        return
    if min(mu, dx, dy, h) <= 0:
        print("mu, dx, dy e h devem ser positivos")
        return

    nt, ny, nx = P.shape
    boundary_conditions = boundary_conditions or {}
    boundary_values = boundary_values or {}
    sides = {"left", "right", "bottom", "top"}
    if not set(boundary_conditions).issubset(sides):
        print("fronteira desconhecida")
        return
    if any(kind not in {"pressure", "flow"} for kind in boundary_conditions.values()):
        print("as condições de fronteira devem ser 'pressure' ou 'flow'")
        return
    for side, kind in boundary_conditions.items():
        if kind == "pressure" and side not in boundary_values:
            print(f"informe a pressão prescrita para {side}")
            return
    if any(not np.isfinite(value) for value in boundary_values.values()):
        print("os valores de fronteira devem ser finitos")
        return

    blocked_x = set()
    blocked_y = set()
    for first, second in internal_no_flow_faces or []:
        if not (0 <= first < nx * ny and 0 <= second < nx * ny):
            print("índice de face interna fora da malha")
            return
        j1, i1 = divmod(first, nx)
        j2, i2 = divmod(second, nx)
        if j1 == j2 and abs(i1 - i2) == 1:
            blocked_x.add((j1, max(i1, i2)))
        elif i1 == i2 and abs(j1 - j2) == 1:
            blocked_y.add((max(j1, j2), i1))
        else:
            print("as faces internas devem ligar células vizinhas")
            return

    vx_faces = np.zeros((nt, ny, nx + 1))
    vy_faces = np.zeros((nt, ny + 1, nx))

    if nx > 1:
        kx = 2 * k[:, :-1] * k[:, 1:] / (k[:, :-1] + k[:, 1:])
        vx_faces[:, :, 1:-1] = -kx[np.newaxis, :, :] * np.diff(P, axis=2) / (mu * dx)
    if ny > 1:
        ky = 2 * k[:-1, :] * k[1:, :] / (k[:-1, :] + k[1:, :])
        vy_faces[:, 1:-1, :] = -ky[np.newaxis, :, :] * np.diff(P, axis=1) / (mu * dy)

    for j, i_face in blocked_x:
        vx_faces[:, j, i_face] = 0.0
    for j_face, i in blocked_y:
        vy_faces[:, j_face, i] = 0.0

    left = boundary_conditions.get("left", "flow")
    right = boundary_conditions.get("right", "flow")
    bottom = boundary_conditions.get("bottom", "flow")
    top = boundary_conditions.get("top", "flow")

    if left == "pressure":
        vx_faces[:, :, 0] = -k[np.newaxis, :, 0] * (P[:, :, 0] - boundary_values["left"]) / (mu * dx / 2)
    elif left == "flow":
        vx_faces[:, :, 0] = boundary_values.get("left", 0.0) / (h * dy)
    if right == "pressure":
        vx_faces[:, :, -1] = -k[np.newaxis, :, -1] * (boundary_values["right"] - P[:, :, -1]) / (mu * dx / 2)
    elif right == "flow":
        vx_faces[:, :, -1] = -boundary_values.get("right", 0.0) / (h * dy)
    if bottom == "pressure":
        vy_faces[:, 0, :] = -k[np.newaxis, 0, :] * (P[:, 0, :] - boundary_values["bottom"]) / (mu * dy / 2)
    elif bottom == "flow":
        vy_faces[:, 0, :] = boundary_values.get("bottom", 0.0) / (h * dx)
    if top == "pressure":
        vy_faces[:, -1, :] = -k[np.newaxis, -1, :] * (boundary_values["top"] - P[:, -1, :]) / (mu * dy / 2)
    elif top == "flow":
        vy_faces[:, -1, :] = -boundary_values.get("top", 0.0) / (h * dx)

    vx = (vx_faces[:, :, :-1] + vx_faces[:, :, 1:]) / 2
    vy = (vy_faces[:, :-1, :] + vy_faces[:, 1:, :]) / 2
    v = np.sqrt(vx**2 + vy**2)
    return vx, vy, v

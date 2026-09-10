import numpy as np

#codigo desenvolvido em metodos numericos 2
def jacobi(A, b, x0, Eppara, max_iter=10000):

    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    x_old = np.array(x0, dtype=float)

    n = len(b)

    for k in range(max_iter):

        x_new = np.copy(x_old)

        for i in range(n):

            soma = 0

            for j in range(n):
                if j != i:
                    soma += A[i, j] * x_old[j]

            x_new[i] = (b[i] - soma) / A[i, i]

        erro = np.max(
            np.abs((x_new - x_old) / x_new) * 100
        )

        if erro <= Eppara:
            return x_new, k+1

        x_old = np.copy(x_new)
    print("Jacobi atingiu o número máximo de iterações.")

    return x_new, max_iter
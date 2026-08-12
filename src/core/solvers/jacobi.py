import numpy as np


def jacobi(A, b, x0, Eppara, max_iter=1000):
    """
    Resolve o sistema linear Ax = b pelo método de Jacobi.

    Parâmetros
    ----------
    A : array_like
        Matriz dos coeficientes.
    b : array_like
        Vetor dos termos independentes.
    x0 : array_like
        Chute inicial.
    Eppara : float
        Erro percentual máximo permitido.
    max_iter : int
        Número máximo de iterações.

    Retorna
    -------
    x : numpy.ndarray
        Vetor solução aproximada.
    """

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
            return x_new

        x_old = np.copy(x_new)

    return x_new
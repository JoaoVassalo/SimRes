import numpy as np

#codigo desenvolvido em metodos numericos 2
def gauss_ingenuo(A, b):
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)

    n = len(b)

    # Matriz aumentada [A | b]
    Aum = np.hstack((A, b.reshape(-1, 1)))

    # Eliminação progressiva
    for i in range(n - 1):

        if np.isclose(Aum[i, i], 0):
            print("Pivô nulo encontrado. Use Gauss com pivotamento.")
            return

        for j in range(i + 1, n):

            fator = Aum[j, i] / Aum[i, i]

            Aum[j, i:n + 1] -= fator * Aum[i, i:n + 1]

    # Substituição regressiva
    x = np.zeros(n)

    x[n - 1] = Aum[n - 1, n] / Aum[n - 1, n - 1]

    for i in range(n - 2, -1, -1):

        soma = 0

        for j in range(i + 1, n):
            soma += Aum[i, j] * x[j]

        x[i] = (Aum[i, n] - soma) / Aum[i, i]

    return x


def fatorar_gauss(A):
    A = np.array(A, dtype=float)
    n = A.shape[0]
    pivots = np.arange(n)
    tolerancia = np.finfo(float).eps * np.max(np.abs(A)) * n

    for i in range(n - 1):
        max_index = np.argmax(np.abs(A[i:n, i])) + i
        if abs(A[max_index, i]) <= tolerancia:
            print("O sistema não possui solução única.")
            return
        if max_index != i:
            A[[i, max_index]] = A[[max_index, i]]
            pivots[[i, max_index]] = pivots[[max_index, i]]
        fatores = A[i + 1:n, i] / A[i, i]
        A[i + 1:n, i] = fatores
        A[i + 1:n, i + 1:n] -= fatores[:, np.newaxis] * A[i, i + 1:n]

    if abs(A[n - 1, n - 1]) <= tolerancia:
        print("O sistema não possui solução única.")
        return
    return A, pivots


def resolver_gauss(fatoracao, b):
    if fatoracao is None:
        return
    A, pivots = fatoracao
    x = np.array(b, dtype=float)[pivots]
    n = len(x)

    for i in range(n):
        x[i] -= np.dot(A[i, :i], x[:i])
    for i in range(n - 1, -1, -1):
        x[i] = (x[i] - np.dot(A[i, i + 1:n], x[i + 1:n])) / A[i, i]
    return x


def gauss_pivotamento(A, b):
    fatoracao = fatorar_gauss(A)
    if fatoracao is None:
        return
    return resolver_gauss(fatoracao, b)

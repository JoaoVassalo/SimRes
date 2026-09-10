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
            raise ValueError("Pivô nulo encontrado. Use Gauss com pivotamento.")

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


def gauss_pivotamento(A, b):
    
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)

    n = len(b)

    # Matriz aumentada [A | b]
    Aum = np.hstack((A, b.reshape(-1, 1)))

    # Eliminação progressiva com pivotamento
    for i in range(n - 1):

        # Procura o maior elemento da coluna atual
        max_index = np.argmax(np.abs(Aum[i:n, i])) + i

        if np.isclose(Aum[max_index, i], 0):
            raise ValueError("O sistema não possui solução única.")

        # Troca de linhas
        if max_index != i:
            Aum[[i, max_index]] = Aum[[max_index, i]]

        # Eliminação
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
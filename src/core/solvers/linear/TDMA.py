import numpy as np

#codigo desenvolvido em metodos numericos 2
def TDMA(matriz, D):

    a = np.diagonal(matriz, offset=-1)
    b = np.diagonal(matriz, offset=0)
    c = np.diagonal(matriz, offset=1)

    n = len(D)

    c_ = np.zeros(n - 1)
    d_ = np.zeros(n)
    x = np.zeros(n)

    c_[0] = c[0] / b[0]
    d_[0] = D[0] / b[0]

    for i in range(1, n - 1):

        c_[i] = c[i] / (b[i] - a[i - 1] * c_[i - 1])

    for i in range(1, n):

        d_[i] = (
            D[i] - a[i - 1] * d_[i - 1]
        ) / (
            b[i] - a[i - 1] * c_[i - 1]
        )

    x[-1] = d_[-1]

    for i in range(n - 2, -1, -1):

        x[i] = d_[i] - c_[i] * x[i + 1]

    return x
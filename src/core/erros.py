import numpy as np


# erro local
def erro_local(P_ref, P_num):
    erro = np.abs(P_ref - P_num)

    return erro


def erro_relativo_percentual(P_ref, P_num):
    erro = np.abs(P_ref - P_num)/P_ref * 100

    return erro


# norma L1 - erro absoluto
def erro_L1(P_ref, P_num):
    erro = np.sum(np.abs(P_ref - P_num))

    return erro


# norma L1 - erro relativo
def erro_L1_relativo(P_ref, P_num):
    erro = np.sum(np.abs((P_ref - P_num) / P_ref))

    return erro


# norma L2 - erro quadrático
def erro_L2(P_ref, P_num):
    erro = np.sqrt(np.sum((P_ref - P_num)**2))

    return erro


# norma L2 - erro quadrático relativo
def erro_L2_relativo(P_ref, P_num):
    erro = np.sqrt(np.sum(((P_ref - P_num) / P_ref)**2))

    return erro


# erro quadrático médio
def erro_MSE(P_ref, P_num):
    erro = np.mean((P_ref - P_num)**2)

    return erro


# raiz do erro quadrático médio
def erro_RMSE(P_ref, P_num):
    erro = np.sqrt(np.mean((P_ref - P_num)**2))

    return erro


# norma Linf - erro máximo
def erro_Linf(P_ref, P_num):
    erro = np.max(np.abs(P_ref - P_num))

    return erro
import numpy as np


def aplicar_falha(k_mD,i_inicio,i_fim,j_inicio,j_fim,k_falha):
    k_falha_mD = k_mD.copy()
    k_falha_mD[j_inicio:j_fim + 1, i_inicio:i_fim + 1] = k_falha

    return k_falha_mD
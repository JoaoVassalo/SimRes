import numpy as np


def gerar_permeabilidade(nx, ny, k_min, k_max, seed=42):
    np.random.seed(seed)
    k_mD = np.random.uniform(k_min,k_max,size=(ny, nx))

    return k_mD

def salvar_permeabilidade(k_mD, nome_arquivo):
    np.savetxt(nome_arquivo,k_mD,delimiter=",")

def carregar_permeabilidade(nome_arquivo):
    k_mD = np.loadtxt(nome_arquivo,delimiter=",")

    return k_mD
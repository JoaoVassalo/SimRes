import numpy as np


def raio_equivalente(dx):
    req = 0.208 * dx
    
    return req


def well_index(req, rw, h):
    WI = (2 * np.pi * h / np.log(req / rw))

    return WI

def vazao_pressao_prescrita(pwf, p_bloco, k, mu, WI):
    lambd = k / mu
    qw = lambd * WI * (pwf - p_bloco)

    return qw

def termos_pressao_prescrita(pwf,k,mu,WI,dt,phi,ct,volume):
    lambd = k / mu
    alpha = (dt* lambd* WI / (phi * ct * volume))
    termo_diagonal = alpha
    termo_fonte = alpha * pwf

    return termo_diagonal, termo_fonte
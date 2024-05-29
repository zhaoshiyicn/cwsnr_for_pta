import json
import glob
import pickle
import warnings
import time

import numpy as np
from enterprise.signals import signal_base
from enterprise.pulsar import Pulsar
from enterprise_extensions.frequentist import optimal_statistic as OS
from enterprise_extensions import deterministic as det

from pptadr3models import getnoise_dict, dr3models


def cw_residual(num, psr, 
                cos_gwtheta = None,
                gwphi       = None,
                cos_inc     = None,
                log10_mc    = None,
                log10_fgw   = None,
                log10_dist  = None,
                phase0      = None,
                psi         = None):
    """
    params note:
    :num: number of inject cw
    :psr: a enterprise pulsar object

    return:
    a list of cw residual
    """
    
    def get_value(param, name, low, high):
        """fix or random params"""
        return np.random.uniform(low, high) if param is None else param

    rrlst = []
    for _ in range(num):
        current_cos_gwtheta = get_value(cos_gwtheta, "cos_gwtheta", -1, 1)
        current_gwphi       = get_value(gwphi, "gwphi", 0, 2 * np.pi)
        current_cos_inc     = get_value(cos_inc, "cos_inc", -1, 1)
        current_log10_mc    = get_value(log10_mc, "log10_mc", 7, 10)
        current_log10_fgw   = get_value(log10_fgw, "log10_fgw", -9, -7.7)
        current_log10_dist  = get_value(log10_dist, "log10_dist", 1.5, 2.5)
        current_phase0      = get_value(phase0, "phase0", 0, 2 * np.pi)
        current_psi         = get_value(psi, "psi", 0, np.pi)

        params_dir = {
            "cos_gwtheta": current_cos_gwtheta,
            "gwphi": current_gwphi,
            "cos_inc": current_cos_inc,
            "log10_mc": current_log10_mc,
            "log10_fgw": current_log10_fgw,
            "log10_dist": current_log10_dist,
            "phase0": current_phase0,
            "psi": current_psi
        }

        """cw residual"""
        rr = det.cw_delay(psr.toas, psr.pos, psr.pdist,
                          psrTerm=False, p_dist=1, p_phase=None,
                          evolve=False, phase_approx=False, check=False,
                          tref=psr.toas.min(), **params_dir)
        rrlst.append(rr)

    return rrlst

def get_snr(num_inject, psrs, models, noisedict, mean_snr=True, **cw_params):

    """
    params:
    :num_inject: number of inject cw
    :psrs: a list of enterprise pulsar object
    :models: a list of enterprise signalcollection object
    :noisedict: 
    :mean_snr: if True get mean snr, else get PDF
    :cw_params: cos_gwtheta, gwphi, cos_inc, log10_mc, log10_fgw, log10_h, phase0, psi

    :return: 
    a dict: {psrname: snr}
    """

    if mean_snr:
        SNR = {psr.name: 0. for psr in psrs}
    else:
        SNR = {psr.name: [] for psr in psrs}

    def get_G(psr): 
        """The G matrix is obtained by svd decomposition of the design matrix."""
        M = psr.Mmat
        n = M.shape[1]
        U, _, _ = np.linalg.svd(M)
        G = U[:, n:]
        return n, G

    for ii, p in enumerate(psrs):

        res_lst = cw_residual(num_inject, p, **cw_params)

        pta  = signal_base.PTA(models[ii](p))
        n, G = get_G(p)  # G-martix
        N    = np.diag(pta.get_ndiag(noisedict)[0])  # white noise cov-martix
        B    = np.diag(pta.get_phi(noisedict)[0][n: ])  # GP prior martix
        T    = pta.get_basis()[0][:, n:]  # design martix collection

        C    = N + np.dot(np.dot(T, B), T.T)  # cov-martix
        G_GT = np.dot(np.dot(G, np.linalg.inv(np.dot(np.dot(G.T, C), G))), G.T)

        for ii, res in enumerate(res_lst):
            snr = np.dot(np.dot(res.T, G_GT), res)  # (S_i|S_i)
            if mean_snr:
                SNR[p.name] += (snr / num_inject)
            else:
                SNR[p.name].append(snr)

        print(f"SNR of {p.name}: {SNR[p.name]}")
        
    return SNR


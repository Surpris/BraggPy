# -*- coding: utf-8 -*-

from numpy import zeros, exp

def calc_modulus(coors, qxx, qyy, qzz):
    """
    Calculate modulus

    < Input >
        coors: spatial coordinates (positions of atoms)
        qxx, qyy, qzz: momentum coordinates
    """

    F = zeros(qxx.shape, dtype=complex)
    for coor in coors:
        phase = coor[0] * qxx + coor[1] * qyy + coor[2] * qzz
        F += exp(-1j*phase)
    return F

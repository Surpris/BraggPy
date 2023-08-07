"""calcfunctions.py

functions to calculate modulus
"""

import numpy


def calc_modulus(
    coors: numpy.ndarray,
    qxx: numpy.ndarray,
    qyy: numpy.ndarray,
    qzz: numpy.ndarray
) -> numpy.ndarray:
    """
    Calculate modulus

    Parameters
    ----------
    coors : numpy.ndarrry
        spatial coordinates (positions of atoms).
        coors has a shape of (N, 3), where N is
        the number of atoms in the target crystal.
    qxx : numpy.ndarrry
        momentum coordinates in the X axis.
    qyy : numpy.ndarrry
        momentum coordinates in the Y axis.
    qzz : numpy.ndarrry
        momentum coordinates in the Z axis (same as the incident X-ray beam).
    """

    F = numpy.zeros(qxx.shape, dtype=complex)
    for coor in coors:
        phase = coor[0] * qxx + coor[1] * qyy + coor[2] * qzz
        F += numpy.exp(-1j*phase)
    return F

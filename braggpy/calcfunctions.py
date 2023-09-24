"""calcfunctions.py

functions to calculate modulus
"""

import concurrent.futures
import numpy


def _calc_modulus(
    coors: numpy.ndarray,
    qxx: numpy.ndarray,
    qyy: numpy.ndarray,
    qzz: numpy.ndarray
) -> numpy.ndarray:
    """
    Base function to calculate a Fourier modulus.

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

    Returns
    -------
    numpy.ndarray : the Fourier modulus.
    """

    dst = numpy.zeros(qxx.shape, dtype=complex)
    for coor in coors:
        phase = coor[0] * qxx + coor[1] * qyy + coor[2] * qzz
        dst += numpy.exp(-1j*phase)
    return dst


def calc_modulus(
    coors: numpy.ndarray,
    qxx: numpy.ndarray,
    qyy: numpy.ndarray,
    qzz: numpy.ndarray,
    n_workers: int = 1
) -> numpy.ndarray:
    """
    Calculate a Fourier modulus.

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
    n_worker : int (default : 1)
        the number of workers for multi processing.

    Returns
    -------
    numpy.ndarray : the Fourier modulus.
    """
    assert isinstance(n_woeker, int), TypeError('`n_worker` must be an integer.')
    assert n_worker >= 1, ValueError('`n_worker` must be >= 1.')

    if n_workers == 1:
        return _calc_modulus(coors, qxx, qyy, qzz)

    n_coors_per_worker = len(coors) // n_workers
    dst = numpy.zeros(qxx.shape, dtype=complex)
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = []
        for index in range(n_workers - 1):
            futures.append(executor.submit(
                _calc_modulus,
                coors[index * n_coors_per_worker:(index + 1) * n_coors_per_worker],
                qxx, qyy, qzz
            ))
        futures.append(executor.submit(
            _calc_modulus,
            coors[(n_workers - 1) * n_coors_per_worker:],
            qxx, qyy, qzz
        ))
        for future_ in concurrent.futures.as_completed(futures):
            dst += future_.result()
    return dst

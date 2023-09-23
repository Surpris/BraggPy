"""momentum.py

functions to calculate coordinates in the momentum (reciprocal) space
"""

import numpy as np


def generate_momentum(photon_energy: float, qmax: float, q_step: float = 0.01) -> dict:
    """
    Generate momemtum coordinates in the Cartesian coordinate

    Parameters
    ----------
    photon_energy : float
        photon energy. The unit is keV.
    qmax : float
        maximum momentum. The unit is 1/angstrom.
    q_step : flaot (default : 0.01)
        momentum step. The unit is 1/angstrom.

    Returns
    -------
    dict object including the following parameters
        qxx, qyy, qzz: target momentum coordinates
        wavelength: wavelength of incident photon
        qmin, qmax: min / max of the target momentum
        q_stepx, q_stepy: step width in qx / qy direction
    """
    wavelength = 12.3849 / photon_energy  # photon_energy = 12.3849 [keV * A] / wavelength [A]
    k_0 = 2. * np.pi / wavelength  # [1/A]

    qrange: np.ndarray = None
    if k_0 < qmax:
        qrange = np.arange(-k_0, k_0 - q_step / 2.0, q_step)
    else:
        qrange = np.arange(-qmax, qmax - q_step / 2.0, q_step)
    qxx, qyy = np.meshgrid(qrange, qrange)
    qzz = np.sqrt(k_0**2 - qxx**2 - qyy**2) - k_0

    res = {
        "qxx": qxx, "qyy": qyy, "qzz": qzz,
        "wavelength": wavelength, "k_0": k_0,
        "qmin": -qmax, "qmax": qmax,
        "q_stepx": q_step, "q_stepy": q_step
    }
    return res


def generate_momentum_polar(
    photon_energy: float, qmax: float, q_step: float = 0.01,
    pht_step: float = 1.0, qmin: float = 0.0
) -> dict:
    """
    Generate momemtum coordinates in the polar coordinate

    Parameters
    ----------
    photon_energy : float
        photon energy. The unit is keV.
    qmax : float
        maximum momentum. The unit is 1/angstrom.
    q_step : flaot (default : 0.01)
        momentum step. The unit is 1/angstrom.
    pht_step : float (default: 1.0)
        angle step in phi direction. The unit is degree.
    qmin : float (default : 0.0)
        minimum momentum. The unit is 1/angstrorm.

    Returns
    -------
    dict object including the following parameters
        qxx, qyy, qzz: target momentum coordinates
        wavelength: wavelength of incident photon
        qmin, qmax: min / max of the target momentum
        q_step, pht_step: step width in q / phi direction
    """
    wavelength = 12.3849 / photon_energy  # photon_energy = 12.3849 [keV * A] / wavelength [A]
    k_0 = 2. * np.pi / wavelength  # [1/A]

    qrange: np.ndarray = None
    if k_0 < qmax:
        qrange = np.arange(-k_0, k_0 - q_step / 2.0, q_step)
    else:
        qrange = np.arange(-qmax, qmax - q_step / 2.0, q_step)
    qrange = np.arange(qmin, qmax + q_step / 2.0, q_step)
    phis = np.arange(0.0, 2.0 * np.pi, np.deg2rad(pht_step))

    qrr, pphi = np.meshgrid(qrange, phis)
    qxx, qyy = qrr*np.cos(pphi), qrr*np.sin(pphi)
    qzz = np.sqrt(k_0**2 - qrr**2) - k_0

    res = {
        "qxx": qxx, "qyy": qyy, "qzz": qzz,
        "wavelength": wavelength, "k_0": k_0,
        "qmin": qmin, "qmax": qmax,
        "q_step": q_step, "pht_step": pht_step
    }
    return res

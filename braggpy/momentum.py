# -*- coding: utf-8 -*-

import numpy as np

def generate_momentum(hv, qmax, dq=0.01, **kwargs):
    """
    Generate momemtum coordinates in Cartesian coordinate

    < Input >
        hv: photon energy (unit of keV)
        qmax: max of the target momentum (unit of angstrom)
        dq: step width
        kwargs: options

    < Output >
        dict object including the following parameters
            qxx, qyy, qzz: target momentum coordinates
            wavelength: wavelength of incident photon
            qmin, qmax: min / max of the target momentum
            dqx, dqy: step width in qx / qy direction
    """
    wavelength = 12.3849 / hv # hv = 12.3849 [keV * A] / wavelength [A]
    k0 = 2.*np.pi/wavelength # [1/A]
    if k0 < qmax:
        qrange = np.arange(-k0, k0-dq/2.0, dq)
    else:
        qrange = np.arange(-qmax, qmax-dq/2.0, dq)
    qxx, qyy = np.meshgrid(qrange, qrange)
    qzz = np.sqrt(k0**2 - qxx**2 - qyy**2) - k0

    res = {"qxx":qxx, "qyy":qyy, "qzz":qzz,
           "wavelength":wavelength, "qmin":-qmax, "qmax":qmax,
           "dqx":dq, "dqy":dq}
    return res

def generate_momentum_polar(hv, qmax, dq=0.01, dphi=1.0, qmin=0.0, **kwargs):
    """
    Generate momemtum coordinates in polar coordinate

    < Input >
        hv: photon energy (unit of keV)
        qmax: max of the target momentum (unit of angstrom)
        dq: step width
        dphi: step width in phi direction
        qmin: min of the target momentum
        kwargs: options
    
    < Output >
        dict object including the following parameters
            qxx, qyy, qzz: target momentum coordinates
            wavelength: wavelength of incident photon
            qmin, qmax: min / max of the target momentum
            dq, dphi: step width in q / phi direction
    """
    wavelength = 12.3849 / hv # hv = 12.3849 [keV * A] / wavelength [A]
    k0 = 2.*np.pi/wavelength # [1/A]
    dq = 0.01
    qmin = 0.5
    qmax = 2.5
    dphi = 1.0 # [deg]

    qrange = np.arange(qmin, qmax+dq/2.0, dq)
    phis = np.arange(0.0, 2.0*np.pi, np.deg2rad(dphi))

    qrr, pphi = np.meshgrid(qrange, phis)
    qxx, qyy = qrr*np.cos(pphi), qrr*np.sin(pphi)
    qzz = np.sqrt(k0**2 - qrr**2) - k0

    res = {"qxx":qxx, "qyy":qyy, "qzz":qzz,
           "wavelength":wavelength, "qmin":qmin, "qmax":qmax,
           "dq":dq, "dphi":dphi}
    return res

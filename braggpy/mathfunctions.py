# -*- coding: utf-8 -*-

import numpy as np


def make_lattice_points(a=1.0, lattice_type="sc", ind_min=-10, ind_max=10, CAR=None):
    """
    Calculate coordinates of each lattice point.
    The implemented lattices are followings:
        sc : simple cubic
        fcc: face-centered cubic
        bcc: body-centered cubic
        hcp: hexagonal close-packed

    <Input>
        a: lattice constant (default: 1.0)
        lattice_type: type of lattice (default: "sc")
        ind_min: Min of Miller's index (default: -10)
        ind_max: Max of Miller's index (default: +10)
        CAR: C-to-A ratio for hcp (default: None=2.0 \\sqrt{2.0/3.0})

    <Output>
        coordinates of lattice points
    """

    # Generate basis vectors
    if lattice_type == "sc":
        a1 = np.array([1., 0., 0.])
        a2 = np.array([0., 1., 0.])
        a3 = np.array([0., 0., 1.])
    elif lattice_type == "fcc":
        a1 = np.array([0.5, 0.5, 0.])
        a2 = np.array([0., 0.5, 0.5])
        a3 = np.array([0.5, 0., 0.5])
    elif lattice_type == "bcc":
        a1 = np.array([0.5, 0.5, -0.5])
        a2 = np.array([-0.5, 0.5, 0.5])
        a3 = np.array([0.5, -0.5, 0.5])
    elif lattice_type == "hcp":
        if CAR is None:
            # c-to-a ratio = 2.0 * np.sqrt(2.0 / 3.0) for ideal lattice
            CAR = 2.0 * np.sqrt(2.0 / 3.0)
        a1 = np.array([1.0, 0., 0.])
        a2 = np.array([-0.5, 0.5*np.sqrt(3.0), 0.])
        a3 = np.array([0.0, 0.0, CAR])
    else:
        raise ValueError("Invalid lattice_type: ", lattice_type)

    # Generate pairs of Miller indices
    ind_list = np.arange(ind_min, ind_max + 1, 1)
    h, k, l = np.meshgrid(ind_list, ind_list, ind_list)

    h = np.reshape(h, (h.size, 1))
    k = np.reshape(k, (k.size, 1))
    l = np.reshape(l, (l.size, 1))

    hkl = np.hstack((np.hstack((h, k)), l))

    # Calculation
    if lattice_type != "hcp":
        A = np.vstack((np.vstack((a1, a2)), a3))  # lattice
        return a * np.dot(hkl, A)
    else:
        A = np.vstack((np.vstack((a1, a2)), a3))  # lattice
        A_coor = a * np.dot(hkl, A)
        B_coor = np.zeros((2 * len(A_coor), 3))
        B_coor[::2] = A_coor
        B = 2./3. * a1 + 1./3. * a2 + 0.5 * a3  # the other atom in the basis
        B_coor[1::2] = A_coor + a * np.tile(B[None, :], (len(A_coor), 1))
        return B_coor.copy()


def euler_rotate(coor, euler_angle, mode=0):
    """
    Rotate the input coordinates by a set of Euler angles.

    < Input >
        coor: coordinates to rotate (3-N or N-3 numpy.2darray)
        euler_angle: a set of Euler angles (length of 3)
        mode: 0=normal rotation, 1=inverse rotation

    < Output >
        rotated coordinates
    """
    # Check the validity of coordinate.
    if len(coor.shape) != 2:
        raise ValueError("'coor' must have exactly 2 dimensions.")
    height, width = coor.shape
    if width != 3 and height != 3:
        raise Exception('Coordinate must be 3-dimensional.')

    length_euler = len(euler_angle)
    if length_euler != 3:
        raise ValueError("'euler_angle' must have exactly 3 elements.")
    if euler_angle[0] == 0 and euler_angle[1] == 0 and euler_angle[2] == 0:
        return coor.copy()

    # Substitute Euler angles.
    alpha = euler_angle[0]*np.pi/180
    beta = euler_angle[1]*np.pi/180
    gamma = euler_angle[2]*np.pi/180

    # mode 0 : normal rotation, 1 : inverse rotation.
    if mode is None:
        # First rotation with respect to z0 axis.
        euler_z0 = np.array([[np.cos(alpha), np.sin(alpha), 0],
                             [-np.sin(alpha), np.cos(alpha), 0],
                             [0, 0, 1]])
        # Second rotation with respect to x1 axis.
        euler_x1 = np.array([[1, 0, 0],
                             [0, np.cos(beta), np.sin(beta)],
                             [0, -np.sin(beta), np.cos(beta)]])
        # Third rotation with respect to z2 axis.
        euler_z2 = np.array([[np.cos(gamma), np.sin(gamma), 0],
                             [-np.sin(gamma), np.cos(gamma), 0],
                             [0, 0, 1]])
        euler_all = np.dot(euler_z2, np.dot(euler_x1, euler_z0))
    else:
        euler_z0 = np.array([[np.cos(alpha), -np.sin(alpha), 0],
                             [np.sin(alpha), np.cos(alpha), 0],
                             [0, 0, 1]])
        euler_x1 = np.array([[1, 0, 0],
                             [0, np.cos(beta), -np.sin(beta)],
                             [0, np.sin(beta), np.cos(beta)]])
        euler_z2 = np.array([[np.cos(gamma), -np.sin(gamma), 0],
                             [np.sin(gamma), np.cos(gamma), 0],
                             [0, 0, 1]])
        euler_all = np.dot(euler_z0, np.dot(euler_x1, euler_z2))

    # Reshaping for output : (N, 3) np.array.
    if height == 3:
        out = np.dot(euler_all, coor)
    else:
        out = np.dot(euler_all, np.transpose(coor))
        out = np.transpose(out)
    return out


def calc_euler_hkl(a, k0, h, k, l):
    """
    return euler angle $(\alpha, \beta, 0)$
    where an intense spot from the (hkl) plane can appear.

    < Input >
        a    : lattice constant
        k0   : wave number of incident photon
        h,k,l: Miller indices

    < Output >
        euler angle (\alpha, \beta, 0)
    """
    q0 = 2.*np.pi / a
    theta = 2. * np.arcsin(q0 * (h**2 + k**2 + l**2)**0.5 / 2.0 / k0)
    phi = np.arctan2(l, k)
    dkx = k0 * np.sin(theta)
    dkz = k0 * np.cos(theta) - k0

    alpha = np.arccos(h * q0 / dkx)
    beta = np.arcsin(dkz / (k**2 + l**2)**0.5 / q0) - phi
    return np.array([alpha, beta, 0.0])


def calculate_lattice_distance(
    lattice_constant: float,
    miller_h: int, miller_k: int, miller_l: int
) -> float:
    """
    calculate the lattice distance of the (hkl) planes.

    Parameters
    ----------
    lattice_constant : float
        lattice constant.
    h,k,l : int
        Miller indices.

    Returns
    -------
    lattice distance (float).
    """
    return lattice_constant / np.linalg.norm([miller_h, miller_k, miller_l])

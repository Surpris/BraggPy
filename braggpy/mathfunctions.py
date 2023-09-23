"""mathfunctions.py
"""

import numpy as np


def make_lattice_points(
    lattice_constant: float = 1.0, lattice_type: str = "sc",
    ind_min: int = -10, ind_max: int = 10, car: float = None
) -> np.ndarray:
    """
    Calculate coordinates of each lattice point.
    The implemented lattices are followings:
        sc : simple cubic
        fcc: face-centered cubic
        bcc: body-centered cubic
        hcp: hexagonal close-packed

    Parameters
    ----------
    lattice_constant : float (default: 1.0)
        lattice constant.
    lattice_type : str (default: "sc")
        lattice type.
    ind_min : int (default: -10)
        minimum of Miller's index.
    ind_max : int (default: +10)
        maximum of Miller's index.
    car : float (default: None = 2.0 / \\sqrt{2.0/3.0})
        C-to-A ratio for hcp.

    Returns
    -------
    numpy.ndarry : coordinates of lattice points.
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
        if car is None:
            # c-to-a ratio = 2.0 * np.sqrt(2.0 / 3.0) for ideal lattice
            car = 2.0 * np.sqrt(2.0 / 3.0)
        a1 = np.array([1.0, 0., 0.])
        a2 = np.array([-0.5, 0.5*np.sqrt(3.0), 0.])
        a3 = np.array([0.0, 0.0, car])
    else:
        raise ValueError(f"Invalid lattice_type: {lattice_type}.")

    # Generate pairs of Miller indices
    ind_list = np.arange(ind_min, ind_max + 1, 1)
    h, k, l = np.meshgrid(ind_list, ind_list, ind_list)

    h = np.reshape(h, (h.size, 1))
    k = np.reshape(k, (k.size, 1))
    l = np.reshape(l, (l.size, 1))

    hkl = np.hstack((np.hstack((h, k)), l))

    # Calculation
    if lattice_type == "hcp":
        A = np.vstack((np.vstack((a1, a2)), a3))  # lattice
        A_coor = lattice_constant * np.dot(hkl, A)
        B_coor = np.zeros((2 * len(A_coor), 3))
        B_coor[::2] = A_coor
        B = 2./3. * a1 + 1./3. * a2 + 0.5 * a3  # the other atom in the basis
        B_coor[1::2] = A_coor + lattice_constant * np.tile(B[None, :], (len(A_coor), 1))
        return B_coor.copy()

    A = np.vstack((np.vstack((a1, a2)), a3))  # lattice
    return lattice_constant * np.dot(hkl, A)    


def euler_rotate(coor: np.ndarray, euler_angle: list, mode: int = 0) -> np.ndarray:
    """
    Rotate the input coordinates by a set of Euler angles.

    Parameters
    ----------
    coor : numpy.ndarray (3xN or Nx3)
        coordinates to rotate.
    euler_angle : array-liky (length of 3)
        a set of Euler angles.
    mode : int (0 ot 1)
        0 = normal rotation, 1 = inverse rotation.

    Returns
    -------
    numpy.ndarray : rotated coordinates.
    """
    # Check the validity of coordinate.
    if len(coor.shape) != 2:
        raise ValueError("'coor' must have exactly 2 dimensions.")
    height, width = coor.shape
    if width != 3 and height != 3:
        raise ValueError('Coordinate must be 3-dimensional.')

    length_euler = len(euler_angle)
    if length_euler != 3:
        raise ValueError("'euler_angle' must have exactly 3 elements.")
    if euler_angle[0] == 0 and euler_angle[1] == 0 and euler_angle[2] == 0:
        return coor.copy()

    # Substitute Euler angles.
    alpha = euler_angle[0] * np.pi / 180
    beta = euler_angle[1] * np.pi / 180
    gamma = euler_angle[2] * np.pi / 180

    # mode 0 : normal rotation, 1 : inverse rotation.
    if mode == 0:
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
        return np.dot(euler_all, coor)

    out = np.dot(euler_all, np.transpose(coor))
    return np.transpose(out)


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


def calc_euler_hkl(
    lattice_constant: float, k_0: float,
    h: int, k: int, l: int
) -> np.ndarray:
    """
    return euler angle $(\alpha, \beta, 0)$
    where an intense spot from the (hkl) plane can appear.

    Parameters
    ----------
        lattice_constant : lattice constant
        k_0   : wave number of incident photon
        h,k,l: Miller indices

    Returns
    -------
    euler angle (\alpha, \beta, 0)
    """
    q_0 = 2.*np.pi / lattice_constant
    theta = 2. * np.arcsin(q_0 * (h**2 + k**2 + l**2)**0.5 / 2.0 / k_0)
    phi = np.arctan2(l, k)
    dkx = k_0 * np.sin(theta)
    dkz = k_0 * np.cos(theta) - k_0

    alpha = np.arccos(h * q_0 / dkx)
    beta = np.arcsin(dkz / (k**2 + l**2)**0.5 / q_0) - phi
    return np.array([alpha, beta, 0.0])

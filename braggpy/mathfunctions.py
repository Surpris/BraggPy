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
        CAR: C-to-A ratio for hcp (default: None=2.0\sqrt{2.0/3.0})
        
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
            CAR = 2.0 * np.sqrt(2.0 / 3.0) # c-to-a ratio = 2.0 * np.sqrt(2.0 / 3.0) for ideal lattice
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

    hkl = np.hstack((np.hstack((h, k)),l))

    # Calculation
    if lattice_type != "hcp":
        A = np.vstack((np.vstack((a1, a2)), a3)) # lattice
        return a * np.dot(hkl, A)
    else:
        A = np.vstack((np.vstack((a1, a2)), a3)) # lattice
        A_coor = a * np.dot(hkl, A)
        B_coor = np.zeros((2*len(A_coor), 3))
        B_coor[::2] = A_coor
        B = 2./3.*a1 +  1./3.*a2 + 0.5*a3 # the other atom in the basis
        B_coor[1::2] = A_coor + a * np.tile(B[None, :], (len(A_coor), 1))
        return B_coor.copy()

def EulerRotation(Coor, EulerAngle=[0,0,0], mode=0):
    """
    Rotate the input coordinates by a set of Euler angles.

    < Input >
        Coor: coordinates to rotate (3-N or N-3 numpy.2darray)
        EulerAngle: a set of Euler angles (length of 3)
        mode: 0=normal rotation, 1=inverse rotation
    
    < Output >
        rotated coordinates
    """
    # Check the validity of coordinate.
    try: h = Coor.shape[0]
    except: h = 1
    try: w = Coor.shape[1]
    except: w = 1
    if w != 3 and h != 3:
        raise Exception('Coordinate must be 3-dimensional.')

    l = len(EulerAngle)
    if l is not 3:
        raise Exception('Coordinate must be 3-dimensional.')
    if EulerAngle[0] == 0 and EulerAngle[1] == 0 and EulerAngle[2] == 0:
        return Coor

    # Substitute Euler angles.
    alpha = EulerAngle[0]*np.pi/180
    beta = EulerAngle[1]*np.pi/180
    gamma = EulerAngle[2]*np.pi/180

    # mode 0 : normal rotation, 1 : inverse rotation.
    if mode is None:
        # First rotation with respect to z0 axis.
        Euler_z0_ori = np.array([[np.cos(alpha), np.sin(alpha), 0],
            [-np.sin(alpha), np.cos(alpha), 0],
            [0, 0, 1]])
        # Second rotation with respect to x1 axis.
        Euler_x1_ori = np.array([[1, 0, 0],
            [0, np.cos(beta), np.sin(beta)],
            [0, -np.sin(beta), np.cos(beta)]])
        # Third rotation with respect to z2 axis.
        Euler_z2_ori = np.array([[np.cos(gamma), np.sin(gamma), 0],
            [-np.sin(gamma), np.cos(gamma), 0],
            [0, 0, 1]])
        Euler_all = np.dot(Euler_z2_ori,np.dot(Euler_x1_ori,Euler_z0_ori))
    else:
        Euler_z0_inv = np.array([[np.cos(alpha), -np.sin(alpha), 0],
            [np.sin(alpha), np.cos(alpha), 0],
            [0, 0, 1]])
        Euler_x1_inv = np.array([[1, 0, 0],
            [0, np.cos(beta), -np.sin(beta)],
            [0, np.sin(beta), np.cos(beta)]])
        Euler_z2_inv = np.array([[np.cos(gamma), -np.sin(gamma), 0],
            [np.sin(gamma), np.cos(gamma), 0],
            [0, 0, 1]])
        Euler_all = np.dot(Euler_z0_inv,np.dot(Euler_x1_inv,Euler_z2_inv))

    # Reshaping for output : (N, 3) np.array.
    if h == 3:
        out = np.dot(Euler_all,Coor)
    else:
        out = np.dot(Euler_all,np.transpose(Coor))
        out = np.transpose(out)
    return out

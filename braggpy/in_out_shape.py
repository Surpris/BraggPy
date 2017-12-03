# -*- coding: utf-8 -*-

import numpy as np

def isInsideSphereIndex(coor, r, **kwargs):
    """
    return indices of a part of coordinates of "coor"
    inside a sphere with the radius of "r"

    <Input>
        coor: coordinates (3-N or N-3 numpy.2darray)
        r: radius of sphere

    <Output>
        The indices of coordinates inside a sphere with the radius of `r`
    """
    if coor.shape[1] == 3:
        buff = coor.transpose()
    else:
        buff = coor.copy()
    return buff[0]**2 + buff[1]**2 + buff[2]**2 <= r**2


def isInsideSphere(coor, r, **kwargs):
    """
    return a part of coordinates of "coor" inside a sphere with the radius of "r"

    <Input>
        coor: coordinates (3-N or N-3 numpy.2darray)
        r: radius of sphere

    <Output>
        The coordinates inside a sphere with the radius of `r`
    """

    index = isInsideSphereIndex(coor, r, **kwargs)
    if coor.shape[1] == 3:
        return coor[index].copy()
    else:
        return coor[:, index].copy()


def isOutsideSphereIndex(coor, r, **kwargs):
    """
    return indices of a part of coordinates of "coor"
    outside a sphere with the radius of "r"

    <Input>
        coor: coordinates (3-N or N-3 numpy.2darray)
        r: radius of sphere

    <Output>
        The indices of coordinates outside a sphere with the radius of `r`
    """
    if coor.shape[1] == 3:
        buff = coor.transpose()
    else:
        buff = coor.copy()
    return buff[0]**2 + buff[1]**2 + buff[2]**2 >= r**2


def isOutsideSphere(coor, r, **kwargs):
    """
    return a part of coordinates of "coor" outside a sphere with the radius of "r"

    <Input>
        coor: coordinates (3-N or N-3 numpy.2darray)
        r: radius of sphere

    <Output>
        coordinates outside a sphere with the radius of `r`
    """

    index = isOutsideSphereIndex(coor, r, **kwargs)
    if coor.shape[1] == 3:
        return coor[index].copy()
    else:
        return coor[:, index].copy()

def isInsideSphereShellIndex(coor, r_outer, **kwargs):
    """
    return indices of a part of coordinates of "coor"
    inside a spherical shell defined by ("r_inner", "r_outer"]
    kwargs must include both "r_inner".

    <Input>
        coor: coordinates (N*3 or 3*N shape)
        r_outer: outer radius of spherical shell
        r_inner: inner radius of spherical shell

    <Output>
        The indices of coordinates inside a spherical shell defined by (r_inner, r_outer]
    """
    if "r_inner" not in kwargs.keys():
        raise KeyError("'kwargs' must have kwarg 'r_inner' (numeric).")
    r_inner = kwargs["r_inner"]
    ind_inside = isInsideSphereIndex(coor, r_outer)
    ind_outside = isOutsideSphereIndex(coor, r_inner)
    return ind_inside & ind_outside


def isInsideSphereShell(coor, r_outer, **kwargs):
    """
    return a part of coordinates of "coor"
    inside a spherical shell defined by ("r_inner", "r_outer"]
    kwargs must include both "r_inner".

    <Input>
        coor: coordinates (N*3 or 3*N shape)
        r_outer: outer radius of spherical shell
        r_inner: inner radius of spherical shell

    <Output>
        The coordinates inside a spherical shell defined by (r_inner, r_outer]
    """

    index = isInsideSphereShellIndex(coor, r_outer, **kwargs)
    if coor.shape[1] == 3:
        return coor[index].copy()
    else:
        return coor[:, index].copy()

def isInsideCubeIndex(coor, a, **kwargs):
    """
    return indices of a part of coordinates of "coor"
    inside a cube with the edge length of "a"

    <Input>
        coor: coordinates (3-N or N-3 numpy.2darray)
        a: edge length of cube

    <Output>
        The indices of coordinates inside a cube with the edge length of "a"
    """
    if coor.shape[1] == 3:
        buff = coor.transpose()
    else:
        buff = coor.copy()
    return (np.abs(buff[0]) <= 0.5*a) & (np.abs(buff[1]) <= 0.5*a) & (np.abs(buff[2]) <= 0.5*a)


def isInsideCube(coor, a, **kwargs):
    """
    return a part of coordinates of "coor"
    inside a cube with the edge length of "a"

    <Input>
        coor: coordinates (3-N or N-3 numpy.2darray)
        a: edge length of cube

    <Output>
        The coordinates inside a cube with the edge length of "a"
    """

    index = isInsideCubeIndex(coor, a, **kwargs)
    if coor.shape[1] == 3:
        return coor[index].copy()
    else:
        return coor[:, index].copy()

SHAPE_NAME = ["cube", "sphere", "sphereshell"]

INSIDEINDEX = {"cube":isInsideCubeIndex,
               "sphere":isInsideSphereIndex,
               "sphereshell":isInsideSphereShellIndex}

def isInsideIndex(coor, a, shape_name, **kwargs):
    """
    return indices of coordinates in "coor" within the specific shape

    <Input>
        coor: coordinates (3-N or N-3 numpy.2darray)
        a: chacacteristic length
        shape_name: shape name (in SHAPE_NAME)
        kwargs: options

    <Output>
        The indices of coordinates inside a cube with the edge length of "a"
    """

    if shape_name not in SHAPE_NAME:
        raise ValueError("'shape_name' must be one of `SHAPE_NAME`.")

    func = INSIDEINDEX[shape_name]
    return func(coor, a, **kwargs)

def isInside(coor, a, shape_name, **kwargs):
    """
    return a part of coordinates in "coor" within the specific shape

    <Input>
        coor: coordinates (3-N or N-3 numpy.2darray)
        a: chacacteristic length
        shape_name: shape name (in SHAPE_NAME)
        kwargs: options

    <Output>
        The coordinates inside a cube with the edge length of "a"
    """

    if shape_name not in SHAPE_NAME:
        raise ValueError("'shape_name' must be one of `SHAPE_NAME`.")

    func = INSIDEINDEX[shape_name]
    index = func(coor, a, **kwargs)
    if coor.shape[1] == 3:
        return coor[index].copy()
    else:
        return coor[:, index].copy()

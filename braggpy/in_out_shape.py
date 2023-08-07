# -*- coding: utf-8 -*-

import numpy as np


def is_inside_sphere_index(coor, r, **kwargs):
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


def is_inside_sphere(coor, r, **kwargs):
    """
    return a part of coordinates of "coor" inside a sphere with the radius of "r"

    <Input>
        coor: coordinates (3-N or N-3 numpy.2darray)
        r: radius of sphere

    <Output>
        The coordinates inside a sphere with the radius of `r`
    """

    index = is_inside_sphere_index(coor, r, **kwargs)
    if coor.shape[1] == 3:
        return coor[index].copy()
    else:
        return coor[:, index].copy()


def is_outside_sphere_index(coor, r, **kwargs):
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


def is_outside_sphere(coor, r, **kwargs):
    """
    return a part of coordinates of "coor" outside a sphere with the radius of "r"

    <Input>
        coor: coordinates (3-N or N-3 numpy.2darray)
        r: radius of sphere

    <Output>
        coordinates outside a sphere with the radius of `r`
    """

    index = is_outside_sphere_index(coor, r, **kwargs)
    if coor.shape[1] == 3:
        return coor[index].copy()
    else:
        return coor[:, index].copy()


def is_inside_sphere_shell_index(coor, r_outer, **kwargs):
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
    ind_inside = is_inside_sphere_index(coor, r_outer)
    ind_outside = is_outside_sphere_index(coor, r_inner)
    return ind_inside & ind_outside


def is_inside_sphere_shell(coor, r_outer, **kwargs):
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

    index = is_inside_sphere_shell_index(coor, r_outer, **kwargs)
    if coor.shape[1] == 3:
        return coor[index].copy()
    else:
        return coor[:, index].copy()


def is_inside_cube_index(coor, a, **kwargs):
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


def is_inside_cube(coor, a, **kwargs):
    """
    return a part of coordinates of "coor"
    inside a cube with the edge length of "a"

    <Input>
        coor: coordinates (3-N or N-3 numpy.2darray)
        a: edge length of cube

    <Output>
        The coordinates inside a cube with the edge length of "a"
    """

    index = is_inside_cube_index(coor, a, **kwargs)
    if coor.shape[1] == 3:
        return coor[index].copy()
    else:
        return coor[:, index].copy()


SHAPE_NAME = ["cube", "sphere", "sphereshell"]

INSIDEINDEX = {
    "cube": is_inside_cube_index,
    "sphere": is_inside_sphere_index,
    "sphereshell": is_inside_sphere_shell_index
}


def is_inside_index(coor, a, shape_name, **kwargs):
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


def is_inside(coor, a, shape_name, **kwargs):
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

# -*- coding: utf-8 -*-

import os.path as path
import numpy as np
import matplotlib.pyplot as plt


def make_figure(width=6, height=5, dpi=100, fignum=None):
    """
    make a figure
    """
    if fignum is None:
        fig = plt.figure(figsize=(width, height), dpi=dpi)
    else:
        fig = plt.figure(fignum, figsize=(width, height), dpi=dpi)
    return fig


def plot_modulus(F, figsize=None, **kwargs):
    """
    give a basic format of plot of modulus
    """
    fig = plt.figure(figsize=figsize, dpi=100)
    if "origin" not in kwargs.keys():
        kwargs["origin"] = "lower"
    if "aspect" not in kwargs.keys():
        kwargs["aspect"] = "auto"
    plt.imshow(np.abs(F), **kwargs)
    return fig


def arrange_figure(xlabel=None, ylabel=None, title=None,
                   ticks_fontsize=14, label_fontsize=14,
                   grid=True, linewidth=0.75):
    """
    arrange a figure
    """
    if xlabel is not None:
        plt.xlabel(xlabel, fontsize=label_fontsize)
    if ylabel is not None:
        plt.ylabel(ylabel, fontsize=label_fontsize)
    if title is not None:
        plt.title(title, fontsize=label_fontsize)
    plt.xticks(fontsize=ticks_fontsize)
    plt.yticks(fontsize=ticks_fontsize)
    if grid is True:
        plt.grid(which='major', axis='x', linewidth=linewidth,
                 linestyle='-', color='0.75')
        plt.grid(which='major', axis='y', linewidth=linewidth,
                 linestyle='-', color='0.75')
    plt.gca().set_axisbelow(True)


def save_figure(filepath, bbox_inches="tight", pad_inches=0.0, overwrite=True, dpi=100):
    """
    save the current figure
    """
    save_count = 0
    _filepath = filepath + ""
    if overwrite is False:
        while path.exists(filepath):
            _ext = _filepath.split('.')[-1]
            save_count += 1
            _filepath = _filepath.replace(
                '.'+_ext, '_save_{0:04d}.{1}'.format(save_count, _ext))
    plt.savefig(_filepath, bbox_inches=bbox_inches,
                pad_inches=pad_inches, dpi=dpi)

# BraggPy

a package to calculate a Fourier modulus corresponding to Bragg reflection from a single crystal. This module calculates Fourier moduli by using the given formula:

$$
F(\vec{q}_\perp) = \sum_{j = 1}^{N} \exp(-\rm{i}\vec{q}\cdot\vec{r}_j).
$$

Here $F$ is a Fourier modulus, $\vec{q}_\perp$ is a momentum transfer perpendiular to the z asix, $\vec{q}$ is a momentum transfer, $N$ is the number of atoms in the crystal, $\vec{r}_j$ is the coordinates of the $j$-th atom.

## Requirements

* Python 3.8+
* numpy
* concurrent

## installation

This module is NOT supposed to be published in PyPI for users to avoid mistaking this module for an official one. You can install this module in the usual ways you do from a given Git repository. One method is as follows:

```sh
pip install git+https://github.com/Surpris/BraggPy.git
```

## Examples & Usage

Please see notebooks in the [notebooks](./notebooks) directory, expecially [Basic functions.ipynb](./notebooks/Basic%20functions.ipynb).

## Contributions

We are welcom to your contributions for improving this module. Any reports, requests, etc. can be posted as issues. 

### Template for issues

Please use the following template when you post your reports, etc. about this module:

```
Title: ...
Background (optional): ...
Purpose of your post: ...
Detail of your post: ...
```

### Coding style

We do not set any coding style other than docstring of functions.

#### Style of docstring

We recommend you to using [the NumPy docstring style](https://numpydoc.readthedocs.io/en/latest/format.html).

"""
PALB is an exact, robust, high-performance solver for the Least-Absolute-Deviations-Line (LAD) problem, i.e. one dimensional affine linear L1 regression. This is the Python API to the core Rust implementation.
"""

from functools import wraps

import numpy as np

import palb_py._core as _core
from palb_py._core import RegressionResult


@wraps(_core.l1line)
def l1line(
    points: np.ndarray,
    normalize_input: bool = True,
) -> RegressionResult:
    """Fit a least absolute deviations (LAD / L1) line to the given points. Points should be an (N, 2) array."""
    return _core.l1line(points.astype(np.float64), normalize_input)


@wraps(_core.l1line_xy)
def l1line_xy(
    xs: np.ndarray,
    ys: np.ndarray,
    normalize_input: bool = True,
) -> RegressionResult:
    """Fit a least absolute deviations (LAD / L1) line to the given points. Points are given as separate 1-dimensional arrays of x and y coordinates."""
    return _core.l1line_xy(
        xs.astype(np.float64), ys.astype(np.float64), normalize_input
    )

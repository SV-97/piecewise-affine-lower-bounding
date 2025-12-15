# `palb` and `palb-py`
PALB is an exact, robust, high-performance solver for the Least-Absolute-Deviations-Line (LAD) problem, i.e. one dimensional affine linear L1 regression.
The core (`palb`) is implemented in Rust, but it also comes with a Python API (`palb_py`).

## The LAD Problem

Let $(x_i, y_i), i=1,...,N$ be points in the plane $\mathbb{R}^2$.
The associated LAD problem seeks to determine a slope $m$ and intercept $t$ that solve the problem

$$
\min_{m,t \in \mathbb{R}} f(m,t), \quad \text{with } f(m,t) = \sum_{i=1}^N |mx_i + t - y_i|.
$$

Least-absolute-deviations (LAD) line fitting is robust to outliers but computationally more involved than least squares regression. Although the literature includes linear and near-linear time algorithms for the LAD line fitting problem, these methods are difficult to implement and, to our knowledge, lack maintained public implementations. As a result, practitioners often resort to linear programming (LP) based methods such as the simplex-based Barrodale-Roberts method and interior-point methods, or on iteratively reweighted least squares (IRLS) approximation which does not guarantee exact solutions.

## PALB

Piecewise Affine Lower-Bounding (PALB) aims to close this gap by being an exact algorithm that is comparatively simple to implement and scales very well in practice.
It guarantees termination with an exact solution in a finite number of steps (for bounds on the number of steps please see the associated paper.
PALB comes with both deterministic and probabilistic bounds, as well as bounds in terms of the quality of an initial guess), and we empirically found that PALB scales log-linearly on both synthetic and real data in practice.
It is consistently faster than publicly available implementations of LP based and IRLS based solvers.

Moreover PALB is relatively straightforward to implement and sports a simple, static memory profile.
This may make it an interesting choice for embedded applications.

## Performance profiles

TODO

## Preprint

TODO

## Documentation

TODO

* crates.io page
* docs.rs API docs
* pypi.org page


## Citing

If you find PALB useful in your work please cite the associated paper as:
```bibtex
@article{
    ...
}
```

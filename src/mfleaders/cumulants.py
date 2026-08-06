"""Log-cumulants c1, c2, c3, ... of the wavelet leaders.

C_m(j) is the m-th cumulant of log|L(j, .)| (natural log), computed from the
raw sample moments of log|L(j,.)| via the standard moments-to-cumulants
recursion:

    C_1 = mu_1
    C_m = mu_m - sum_{n=1}^{m-1} binom(m-1, n-1) * C_n * mu_(m-n)   (m >= 2)

The log-cumulants c_m are the slopes of C_m(j) against j, converted from
nats-per-octave to the standard convention via a factor of log2(e) (since
C_m(j) itself is built from a natural log, while the regression runs against
log2-spaced scales j).
"""

import numpy as np
from scipy import special

from .regression import linear_regression


def _cumulants_at_scale(log_abs_leaders, n_cumul):
    finite = ~np.isnan(log_abs_leaders) & ~np.isinf(log_abs_leaders)
    n_valid = int(finite.sum())

    c = np.full(n_cumul, np.nan)
    if n_valid < 3:
        return c, n_valid

    x = log_abs_leaders[finite]
    moments = np.array([np.mean(x ** m) for m in range(1, n_cumul + 1)])

    c[0] = moments[0]
    for m in range(2, n_cumul + 1):
        aux = 0.0
        for n in range(1, m):
            aux += special.binom(m - 1, n - 1) * c[n - 1] * moments[m - n - 1]
        c[m - 1] = moments[m - 1] - aux

    return c, n_valid


def cumulant_scaling_functions(leaders_by_scale, n_cumul):
    """C_m(j) for m = 1..n_cumul, at every scale.

    Returns
    -------
    scales : ndarray, shape (n_j,)
    C : ndarray, shape (n_j, n_cumul)
    nj : ndarray, shape (n_j,)
    """
    scales = sorted(leaders_by_scale)
    C = np.full((len(scales), n_cumul), np.nan)
    nj = np.zeros(len(scales), dtype=int)

    for i, j in enumerate(scales):
        vals = np.abs(leaders_by_scale[j])
        with np.errstate(divide='ignore'):
            log_vals = np.log(vals)
        c, n_valid = _cumulants_at_scale(log_vals, n_cumul)
        C[i] = c
        nj[i] = n_valid

    return np.array(scales), C, nj


def log_cumulants(leaders_by_scale, n_cumul, j1, j2, weighted=None):
    """Estimate c_1..c_n_cumul by regressing C_m(j) against j over [j1, j2].

    Returns
    -------
    scales, C, nj : as in `cumulant_scaling_functions`
    c : ndarray, shape (n_cumul,)
        Log-cumulants c_1, c_2, ..., c_n_cumul.
    """
    scales, C, nj = cumulant_scaling_functions(leaders_by_scale, n_cumul)

    in_range = (scales >= j1) & (scales <= j2)
    x = scales[in_range]
    y = C[in_range]

    w = nj[in_range].astype(float) if weighted == 'Nj' else None

    slope, _ = linear_regression(x, y, w)
    c = slope * np.log2(np.e)

    return scales, C, nj, c

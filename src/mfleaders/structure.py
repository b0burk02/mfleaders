"""Structure functions S(q,j) and the scaling exponents zeta(q).

S(q, j) = <|L(j, .)|^q>, averaged over all valid (non-NaN) leaders at scale
j. zeta(q) is the slope of log2 S(q, j) against j, fit by linear regression
over a chosen scaling range [j1, j2].
"""

import numpy as np

from ._utils import power
from .regression import linear_regression


def structure_functions(leaders_by_scale, q):
    """Compute log2 S(q, j) for every scale and every q.

    Parameters
    ----------
    leaders_by_scale : dict[int, ndarray]
    q : ndarray, shape (n_q,)

    Returns
    -------
    scales : ndarray, shape (n_j,)
    log2_s : ndarray, shape (n_j, n_q)
        NaN at (j, q) where fewer than 3 valid leaders were available at
        scale j.
    nj : ndarray, shape (n_j,)
        Count of valid (non-NaN, non-inf) leaders at each scale.
    """
    q = np.asarray(q, dtype=float)
    scales = sorted(leaders_by_scale)
    log2_s = np.full((len(scales), len(q)), np.nan)
    nj = np.zeros(len(scales), dtype=int)

    eps = np.finfo(float).eps

    for i, j in enumerate(scales):
        vals = np.abs(leaders_by_scale[j])
        finite = ~np.isnan(vals) & ~np.isinf(vals)
        n_valid = int(finite.sum())
        nj[i] = n_valid

        if n_valid < 3:
            continue

        valid_vals = vals[finite]
        # MULTIFRACS's flexEstFun_MFA.m (lines 26-30) discards |L| < eps
        # BEFORE raising to any power, once per scale (not per q), and its
        # mean's denominator is the post-trim count (`size(detail_kq,2)`,
        # i.e. plain `lenA`) -- not the pre-trim `n_valid` reported in `nj`
        # above. `np.mean` below already uses `valid_vals`'s (post-trim)
        # length as its denominator, matching that. `nj` itself is left
        # alone (NaN/inf-exclusion only) -- it is a separate bookkeeping
        # count, not what MULTIFRACS's moment mean divides by.
        valid_vals = valid_vals[valid_vals >= eps]
        if valid_vals.size < 3:
            continue

        for qi, qq in enumerate(q):
            with np.errstate(invalid='ignore', divide='ignore'):
                s = np.mean(power(valid_vals, qq))
                log2_s[i, qi] = np.log2(s)

    log2_s[np.isinf(log2_s)] = np.nan
    return np.array(scales), log2_s, nj


def zeta_q(leaders_by_scale, q, j1, j2, weighted=None):
    """Estimate zeta(q) by regressing log2 S(q,j) against j over [j1, j2].

    Returns
    -------
    scales, log2_s, nj : as in `structure_functions`
    zeta : ndarray, shape (n_q,)
    intercept : ndarray, shape (n_q,)
    """
    scales, log2_s, nj = structure_functions(leaders_by_scale, q)

    in_range = (scales >= j1) & (scales <= j2)
    x = scales[in_range]
    y = log2_s[in_range]

    w = nj[in_range].astype(float) if weighted == 'Nj' else None

    zeta, intercept = linear_regression(x, y, w)

    return scales, log2_s, nj, zeta, intercept

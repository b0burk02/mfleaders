"""Multifractal spectrum D(h): both branches, from a single q sweep.

This uses the "direct" (partition-function) estimator of the Legendre
transform of zeta(q), rather than differentiating a fitted zeta(q) curve
numerically (which is unstable). At each scale j, leaders are turned into a
Gibbs-like weighting

    R(j, q, k) = |L(j,k)|^q / sum_k |L(j,k)|^q

and two auxiliary scaling functions are formed:

    V(j, q) = sum_k R(j,q,k) * log2|L(j,k)|
    U(j, q) = log2(N_j) + sum_k R(j,q,k) * log2 R(j,q,k)

h(q) and D(q) are the slopes of V(j,q) and U(j,q) respectively, regressed
against j over [j1, j2], with D(q) shifted by +1 (the ambient dimension).
Sweeping q over both positive and negative values recovers both branches of
D(h). zeta(q) is concave (a consequence of Holder's inequality, true for any
process), so h(q) = zeta'(q) is monotonically decreasing in q: q > 0
selects the SMALL-h (singular, left) branch and q < 0 selects the LARGE-h
(regular, right) branch -- verified directly against the binomial cascade's
closed-form spectrum in tests/test_analytic_validation.py, where positive q
provably homes in on the small-alpha (large-mass, singular) part of the
measure. This is the entire reason leader-based MFA can recover the left
branch at all, unlike raw-coefficient moments restricted to q >= 0.

D(h) here is a *generalized-dimension* curve: D as a function of the Hölder
exponent h. Its width is a separate quantity, kept explicit and never
conflated with Delta-h or c2 (see `holder_width`).
"""

import numpy as np

from ._utils import power
from .regression import linear_regression


def _xlogx(x, base2=True):
    out = np.zeros_like(x)
    positive = x > 0
    if base2:
        out[positive] = x[positive] * np.log2(x[positive])
    else:
        out[positive] = x[positive] * np.log(x[positive])
    return out


def spectrum_scaling_functions(leaders_by_scale, q):
    """U(j,q) and V(j,q) at every scale.

    Returns
    -------
    scales : ndarray, shape (n_j,)
    U, V : ndarray, shape (n_j, n_q)
    nj : ndarray, shape (n_j,)
    """
    q = np.asarray(q, dtype=float)
    scales = sorted(leaders_by_scale)
    U = np.full((len(scales), len(q)), np.nan)
    V = np.full((len(scales), len(q)), np.nan)
    nj = np.zeros(len(scales), dtype=int)

    for i, j in enumerate(scales):
        vals = np.abs(leaders_by_scale[j])
        finite = ~np.isnan(vals) & ~np.isinf(vals)
        n_valid = int(finite.sum())
        nj[i] = n_valid

        if n_valid < 3:
            continue

        valid_vals = vals[finite]
        log2_vals = np.log2(valid_vals)

        for qi, qq in enumerate(q):
            with np.errstate(invalid='ignore', divide='ignore'):
                weights = power(valid_vals, qq)
            z = weights.sum()
            if z == 0:
                continue
            r = weights / z

            V[i, qi] = np.sum(r * log2_vals)
            U[i, qi] = np.log2(n_valid) + np.sum(_xlogx(r))

    return np.array(scales), U, V, nj


def multifractal_spectrum(leaders_by_scale, q, j1, j2, weighted=None, dim=2):
    """Estimate (h(q), D(q)) -- both branches of D(h) -- over [j1, j2].

    Parameters
    ----------
    dim : int
        Ambient dimension (1 for a 1D signal, 2 for an image, ...). Default
        is 2: mfleaders' public interface is 2D-only (see
        src/mfleaders/__init__.py); every 1D call site (the internal,
        pymultifracs-validated regression guard) pins `dim=1` explicitly --
        see tests/test_oracle_pymultifracs.py, tests/test_analytic_
        validation.py, and pipeline.py's `analyze_1d`. D(q) is the
        regression slope of U(j,q) shifted by +dim, not a fixed +1:
        log2(N_j) -- the term inside U(j,q) that carries the dimension
        dependence -- scales as roughly -dim per octave for a full,
        non-degenerate covering (N_j itself comes straight from the data,
        so it already reflects dim correctly; only the additive shift that
        turns a slope into a dimension value needs to know dim explicitly).
        Getting this wrong for 2D leaders gives D(q) values capped near 1
        instead of 2 -- caught in tests/test_analytic_validation_2d.py by
        fBm2D's D(q) topping out near 1 instead of the ambient dimension 2.

    Returns
    -------
    scales, U, V, nj : as in `spectrum_scaling_functions`
    hq, Dq : ndarray, shape (n_q,)
    """
    scales, U, V, nj = spectrum_scaling_functions(leaders_by_scale, q)

    in_range = (scales >= j1) & (scales <= j2)
    x = scales[in_range]

    w = nj[in_range].astype(float) if weighted == 'Nj' else None

    hq, _ = linear_regression(x, V[in_range], w)
    Dq, _ = linear_regression(x, U[in_range], w)
    Dq = Dq + dim

    return scales, U, V, nj, hq, Dq


def holder_width(hq):
    """Delta-h: the width of the Hölder-spectrum support, max(h) - min(h).

    This is a native leader-based width, distinct from the log-cumulant c2
    and from a generalized-dimension "D-span" (a range in D, not in h) --
    see CLAUDE.md hard rule 5. Never conflate the three.
    """
    hq = np.asarray(hq)
    finite = hq[~np.isnan(hq)]
    if finite.size == 0:
        return np.nan
    return float(finite.max() - finite.min())

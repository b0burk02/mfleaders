"""Linear regression of a per-scale quantity against j, over [j1, j2]."""

import numpy as np


def linear_regression(j, y, weights=None):
    """Ordinary (optionally weighted) least-squares fit of ``y`` vs ``j``.

    Parameters
    ----------
    j : ndarray, shape (n_j,)
        Integer scales.
    y : ndarray, shape (n_j, ...)
        Values to regress, one row per scale. NaN rows are excluded.
    weights : ndarray, shape (n_j,) | None
        Per-scale weights (e.g. number of coefficients). ``None`` means
        unweighted, i.e. all finite rows equally weighted.

    Returns
    -------
    slope, intercept : ndarray, shape (...)
    """
    j = np.asarray(j, dtype=float)
    y = np.asarray(y, dtype=float)

    if weights is None:
        w = np.ones_like(j)
    else:
        w = np.asarray(weights, dtype=float)

    finite = ~np.isnan(y)

    extra_shape = y.shape[1:]
    slope = np.full(extra_shape, np.nan)
    intercept = np.full(extra_shape, np.nan)

    if y.ndim == 1:
        mask = finite
        if mask.sum() < 2:
            return slope, intercept
        wj = w[mask]
        x = j[mask]
        yy = y[mask]
        v0 = wj.sum()
        v1 = (wj * x).sum()
        v2 = (wj * x ** 2).sum()
        denom = v0 * v2 - v1 * v1
        weights_slope = wj * (v0 * x - v1) / denom
        weights_intercept = wj * (v2 - v1 * x) / denom
        slope = (weights_slope * yy).sum()
        intercept = (weights_intercept * yy).sum()
        return slope, intercept

    it = np.ndindex(extra_shape)
    for idx in it:
        mask = finite[(slice(None),) + idx]
        if mask.sum() < 2:
            continue
        wj = w[mask]
        x = j[mask]
        yy = y[(slice(None),) + idx][mask]
        v0 = wj.sum()
        v1 = (wj * x).sum()
        v2 = (wj * x ** 2).sum()
        denom = v0 * v2 - v1 * v1
        weights_slope = wj * (v0 * x - v1) / denom
        weights_intercept = wj * (v2 - v1 * x) / denom
        slope[idx] = (weights_slope * yy).sum()
        intercept[idx] = (weights_intercept * yy).sum()

    return slope, intercept


def linear_regression_se(j, y, weights=None):
    """Same fit as `linear_regression`, for a single (1D) ``y``, plus the
    standard error of the slope from the fit's own residuals.

    ``weights`` are treated as relative precisions (as in weighted least
    squares): Var(slope) = sigma^2 * sum(c_i^2 / w_i), where c_i are the
    slope's linear-combination coefficients and sigma^2 is the weighted
    residual variance. This uses only the single fit's residual scatter --
    no bootstrap, no reference to any other implementation or known answer.
    """
    j = np.asarray(j, dtype=float)
    y = np.asarray(y, dtype=float)

    if weights is None:
        w = np.ones_like(j)
    else:
        w = np.asarray(weights, dtype=float)

    mask = ~np.isnan(y)
    n = int(mask.sum())
    if n < 3:
        return np.nan, np.nan, np.nan

    wj = w[mask]
    x = j[mask]
    yy = y[mask]

    v0 = wj.sum()
    v1 = (wj * x).sum()
    v2 = (wj * x ** 2).sum()
    denom = v0 * v2 - v1 * v1

    weights_slope = wj * (v0 * x - v1) / denom
    weights_intercept = wj * (v2 - v1 * x) / denom

    slope = (weights_slope * yy).sum()
    intercept = (weights_intercept * yy).sum()

    residuals = yy - (slope * x + intercept)
    sigma2 = (wj * residuals ** 2).sum() / (n - 2)
    se_slope = np.sqrt(sigma2 * (weights_slope ** 2 / wj).sum())

    return slope, intercept, se_slope

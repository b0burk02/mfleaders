"""Discrete wavelet transform front-end for the leader pipeline.

This module is dimension-specific (1D today; a 2D counterpart built on
``pywt.wavedec2`` follows the same border-NaN convention at Gate 2). The
dimension-generic part of the pipeline is the leader construction in
``leaders.py``, not the DWT itself.

Boundary convention: coefficients corrupted by zero-padding at the signal's
edges are set to NaN rather than trimmed to a shorter array, so that every
scale's coefficient array stays alignable with its neighbors under dyadic
subsampling. This matches pymultifracs' ``wavelet_analysis`` (mode='zero'
per-level DWT, chained by hand rather than via a single ``wavedec`` call, so
that border corruption is masked out scale by scale before it can propagate
into the next level's input).
"""

from math import ceil

import numpy as np
import pywt


def _dwt_step(approx, wavelet, axis=0, fp=None):
    """One level of DWT with mode='zero' along ``axis``, returning
    (detail, next_approx); other axes (if any) are left untouched (batched).

    Border-corrupted samples are set to NaN in both outputs. ``wavelet.dec_hi``
    is negated to match the sign convention used by pymultifracs.

    ``fp`` : int | None
        Number of border-corrupted samples marked NaN at each end. Default
        (``None``) gives ``ceil((filt_len-1)/2)-1``, the convention this was
        built and verified against (Gate 1's pymultifracs oracle match, 1D)
        -- do not change the default, 1D depends on it exactly. The 2D path
        (``_dwt2.py``) passes ``fp=wavelet.dec_len`` explicitly to match
        MULTIFRACS's ``DxPx2d.m`` (``fp = nl``, the full filter length) --
        a different, 2D-specific convention, not a claim that the default
        is wrong.
    """
    low, high = pywt.dwt(approx, wavelet, mode='zero', axis=axis)
    high = -high

    if fp is None:
        filt_len = wavelet.dec_len
        fp = ceil((filt_len - 1) / 2) - 1
    lp = -fp if fp != 0 else None

    high = high.copy()
    low = low.copy()

    def _mark_nan(arr, sl):
        idx = [slice(None)] * arr.ndim
        idx[axis] = sl
        arr[tuple(idx)] = np.nan

    def _take(arr, sl):
        idx = [slice(None)] * arr.ndim
        idx[axis] = sl
        return arr[tuple(idx)]

    _mark_nan(high, slice(None, fp))
    _mark_nan(low, slice(None, fp))
    if lp is not None:
        _mark_nan(high, slice(lp, None))
        _mark_nan(low, slice(lp, None))

    if approx.shape[axis] % 2 == 1:
        detail = _take(high, slice(None, -2))
        next_approx = _take(low, slice(fp, lp))
    else:
        detail = _take(high, slice(None, -1))
        if lp == -1:
            next_approx = _take(low, slice(fp, None))
        elif lp is not None:
            next_approx = _take(low, slice(fp, lp + 1))
        else:
            next_approx = _take(low, slice(fp, None))

    # The length-fixing slices above (needed because pywt's mode='zero'
    # output is 1 sample longer than the exact half-length) consume a
    # different number of the fp-wide marked border on each end -- e.g. the
    # even branch's `next_approx` keeps 0 marked samples at the front but 1
    # at the back. That's an asymmetry in ARRAY LENGTH bookkeeping, not a
    # real difference in how contaminated each end is (zero-padding taints
    # exactly fp samples at both ends equally). Fix it by re-marking,
    # independent of whatever the slicing above left in place: mark exactly
    # fp samples NaN at EACH end of both outputs, unconditionally. This
    # changes no array length (so `_dwt2_step`'s offset alignment, which
    # depends only on lengths, needs no change), only which elements are
    # flagged invalid.
    if fp > 0:
        _mark_nan(detail, slice(None, fp))
        _mark_nan(detail, slice(-fp, None))
        _mark_nan(next_approx, slice(None, fp))
        _mark_nan(next_approx, slice(-fp, None))

    return detail, next_approx


def _max_decomposition_level(n_samples, filt_len, j2=None):
    max_level = int(np.floor(np.log2(n_samples / (filt_len + 1))))
    max_level = min(int(np.floor(np.log2(n_samples))), max_level)
    if j2 is not None:
        max_level = min(max_level, j2)
    return max_level


def wavelet_transform_1d(signal, wt_name='db3', normalization=1, j2=None):
    """Compute per-scale DWT detail coefficients of a 1D signal.

    Parameters
    ----------
    signal : ndarray, shape (n_samples,)
    wt_name : str
        pywt wavelet name, e.g. 'db3'.
    normalization : int
        Exponent controlling the norm convention: coefficients are scaled by
        ``2 ** (scale * (0.5 - 1/normalization))``. ``normalization=1`` gives
        the L1-normalized convention used throughout this package (matching
        pymultifracs' default), under which coefficient magnitude picks up an
        artifactual ~sqrt(2)-per-octave decay on top of the intrinsic
        orthonormal-DWT decay. The leader construction in ``leaders.py``
        compensates for exactly this via its ``2**-dim`` cone weight.
    j2 : int | None
        Cap on the number of scales to compute.

    Returns
    -------
    dict[int, ndarray]
        Maps scale (1 = finest) to detail-coefficient array, with
        border-corrupted entries set to NaN.
    """
    signal = np.asarray(signal, dtype=float)
    wavelet = pywt.Wavelet(wt_name)
    max_level = _max_decomposition_level(len(signal), wavelet.dec_len, j2)

    coeffs = {}
    approx = signal
    for scale in range(1, max_level + 1):
        detail, approx = _dwt_step(approx, wavelet)
        detail = detail * 2 ** (scale * (0.5 - 1 / normalization))

        if np.sum(~np.isnan(detail)) < 3:
            break

        coeffs[scale] = detail

    return coeffs

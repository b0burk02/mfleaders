"""End-to-end 1D pipeline: DWT -> leaders -> structure functions -> zeta(q)
-> D(h) (both branches) -> log-cumulants.
"""

from dataclasses import dataclass

import numpy as np

from ._dwt import wavelet_transform_1d
from .leaders import compute_leaders
from .structure import zeta_q
from .spectrum import multifractal_spectrum, holder_width
from .cumulants import log_cumulants


@dataclass
class MFA1DResult:
    coeffs: dict
    leaders: dict

    scales: np.ndarray
    log2_s: np.ndarray
    nj: np.ndarray
    q: np.ndarray
    zeta: np.ndarray
    zeta_intercept: np.ndarray

    hq: np.ndarray
    Dq: np.ndarray
    delta_h: float

    c: np.ndarray

    j1: int
    j2: int

    @property
    def H(self):
        """H = zeta(2) / 2, only meaningful if q includes 2."""
        idx = np.where(np.isclose(self.q, 2))[0]
        if len(idx) == 0:
            return np.nan
        return self.zeta[idx[0]] / 2

    @property
    def c1(self):
        return self.c[0]

    @property
    def c2(self):
        return self.c[1] if len(self.c) > 1 else np.nan

    @property
    def c3(self):
        return self.c[2] if len(self.c) > 2 else np.nan


def analyze_1d(signal, j1, j2, wt_name='db3', normalization=1, q=None,
               n_cumul=3, weighted=None):
    """Run the full 1D wavelet-leader MFA pipeline on ``signal``.

    Parameters
    ----------
    signal : ndarray, shape (n_samples,)
    j1, j2 : int
        Scaling range used for every regression (zeta(q), D(h), log-cumulants).
    wt_name : str
        pywt wavelet name.
    normalization : int
        Coefficient norm convention; see `_dwt.wavelet_transform_1d`.
    q : ndarray | None
        Moments for the structure functions / D(h) sweep. Defaults to a
        symmetric range covering both branches.
    n_cumul : int
        Number of log-cumulants to estimate (c1, c2, ..., c_n_cumul).
    weighted : str | None
        None (unweighted) or 'Nj' (weight scales by valid-leader count).
    """
    if q is None:
        q = np.concatenate([np.arange(-5, 0), np.arange(1, 6)]).astype(float)
    else:
        q = np.asarray(q, dtype=float)

    coeffs = wavelet_transform_1d(signal, wt_name=wt_name,
                                   normalization=normalization)
    # cone='pymultifracs' explicitly: analyze_1d is THE Gate-1-certified 1D
    # pipeline (validated to float precision against pymultifracs' own
    # compute_leaders); compute_leaders' own default changed to
    # 'multifracs' (2D is the primary target now, per review), but this
    # pipeline's certified behavior must not silently move with it.
    leaders = compute_leaders(coeffs, dim=1, cone='pymultifracs')

    scales, log2_s, nj, zeta, intercept = zeta_q(
        leaders, q, j1, j2, weighted=weighted)

    # dim=1 explicitly: multifractal_spectrum's own default is now dim=2
    # (mfleaders' public interface is 2D-only), but this is the 1D
    # pipeline -- pin it, do not silently move onto the 2D default.
    _, _, _, _, hq, Dq = multifractal_spectrum(
        leaders, q, j1, j2, weighted=weighted, dim=1)

    _, _, _, c = log_cumulants(leaders, n_cumul, j1, j2, weighted=weighted)

    return MFA1DResult(
        coeffs=coeffs, leaders=leaders,
        scales=scales, log2_s=log2_s, nj=nj,
        q=q, zeta=zeta, zeta_intercept=intercept,
        hq=hq, Dq=Dq, delta_h=holder_width(hq),
        c=c, j1=j1, j2=j2,
    )

"""2D discrete wavelet transform front-end (Gate 2).

Dimension-specific, like ``_dwt.py`` (rule 2 only mandates the LEADER
construction be dimension-generic, not the DWT itself, which is inherently
per-dimension via pywt's own wavedec/wavedec2/wavedecn family).

Built by applying the same 1D, mode='zero' step from ``_dwt.py`` separably
along each axis (rows, then columns), reusing its exact boundary-NaN
convention on each axis independently: a 2D coefficient is corrupted (NaN)
if either axis's cone-of-influence crosses the signal's edge, since a
separable 2D wavelet's support is the outer product of its two 1D supports.
There is no 2D Python reference to match bit-for-bit (CLAUDE.md's validation
ladder step 3 notes this explicitly) -- this only needs to be self-consistent
and to feed the same dimension-generic leader core used at dim=1.

Each level produces one approximation subband (LL, fed to the next level)
and three detail subbands (LH, HL, HH -- horizontal/vertical/diagonal).
Wavelet leaders combine all three orientations, so all three are returned
per scale as a single stacked array with a leading orientation axis of
length 3; ``combine_orientations`` collapses that axis via elementwise max
BEFORE ``leaders.compute_leaders`` runs (an explicit, dimension-2/3-specific
bit of glue -- the neighborhood+cone-max primitive itself never sees an
orientation axis and does not need to know about orientations).

Normalization is dimension-dependent. The L1-normalization exponent used in
``wavelet_transform_1d`` (``2**(scale*(0.5-1/normalization))``) is the
dim=1 case of a general d-dimensional formula: a d-dim wavelet's L^p-norm
coefficient scales as ``(2**(scale*d))**(1/p)`` versus
``(2**(scale*d))**(1/2)`` for the orthonormal/L2 reference, so the
compensating exponent scales with d too:
``2**(scale*d*(0.5-1/normalization))``. Missing this factor of d=2 here
inflated the recovered H for fBm2D by close to a factor of 2 -- caught by
the analytic validation in test_analytic_validation_2d.py, not assumed
correct by analogy with 1D.
"""

import numpy as np
import pywt
from scipy.signal import convolve2d

from ._dwt import _dwt_step, _max_decomposition_level


def _dwt2_step(approx, wavelet):
    """One level of 2D DWT: returns (LL, detail3) where detail3 stacks
    (LH, HL, HH) along a new leading axis.

    ``_dwt_step`` returns (detail, next_approx) -- easy to transpose by
    accident when chaining two axis passes, so unpacked explicitly here.

    Along each axis, ``_dwt_step``'s detail and next_approx outputs differ
    in LENGTH (by a small, deterministic offset, generally = fp, from the
    length-fixing trim pywt's mode='zero' output needs), with next_approx
    corresponding to the LAST `len(detail)-len(next_approx)` positions of
    the physical range detail spans. This is purely a length/indexing
    relationship, independent of NaN placement: `_dwt_step` marks a
    symmetric fp-wide NaN border at BOTH ends of both outputs (see its
    docstring), so this crop's only job is to align the four subbands'
    differing physical extents before stacking, anchored on LL's frame
    (the most-offset one, since it's what carries forward to the next
    scale unchanged) -- it does not need to know or care about NaN
    positions, since the border symmetry is guaranteed upstream.

    Does NOT pass ``fp=wavelet.dec_len`` here, despite MULTIFRACS's
    ``DxPx2d.m`` using ``fp = nl`` (the full filter length, e.g. 6 for db3)
    where this uses the narrower ``ceil((filt_len-1)/2)-1`` default -- tried
    and reverted (it regressed the cascade cross-check and starved j=6 to
    nj=0; superseded by `_dwt2_step_multifracs` below, which is what
    `test_multifracs_border_reaches_j6_with_matching_nj` in
    `tests/test_dwt2.py` exercises). `_dwt_step`'s length-
    trimming architecture (physically shortening `next_approx` by `fp` at
    every level) is a DIFFERENT mechanism from MULTIFRACS's (which keeps
    the array at full subsampled length and lets `Inf` propagate through
    subsequent convolutions instead of trimming). Swapping in MULTIFRACS's
    width numerically without also porting its propagation mechanism made
    the cascade cross-check residual WORSE and starved coarse scales of
    valid leaders entirely (j=6 dropped to nj=0). `_dwt_step` still accepts
    an `fp` override (harmless, unused by default) for whenever that
    mechanism gets ported properly; it is not a scalar to just bump.
    """
    high0, low0 = _dwt_step(approx, wavelet, axis=0)

    LH, LL = _dwt_step(low0, wavelet, axis=1)
    HH, HL = _dwt_step(high0, wavelet, axis=1)

    offset0 = high0.shape[0] - low0.shape[0]
    offset1 = LH.shape[1] - LL.shape[1]

    HH = HH[offset0:, offset1:]
    LH = LH[:, offset1:]
    HL = HL[offset0:, :]

    detail3 = np.stack([LH, HL, HH], axis=0)
    return LL, detail3


def _full_convolve_axis(arr, kernel, axis):
    """'full' 1D convolution of ``arr`` along ``axis`` with 1D ``kernel``
    (zero-padding beyond the array, matching MATLAB's ``conv2(arr, kernel)``
    when ``kernel`` is a row/column vector).

    Uses ``scipy.signal.convolve2d``'s direct (non-FFT) method: verified in
    an ad hoc check (a NaN in a length-9 test array convolved with a 3-tap
    kernel produces NaN only at the 3 positions within the kernel's reach,
    matching direct multiply-accumulate, not FFT spectral smearing) that is
    not itself committed as a test -- what IS committed and exercises this
    function end to end is `test_multifracs_border_reaches_j6_with_matching_nj`
    (`tests/test_dwt2.py`): its exact nj match against MULTIFRACS's reported
    counts is only possible if NaN propagates correctly here.
    """
    k = kernel.reshape(1, -1) if axis == 1 else kernel.reshape(-1, 1)
    return convolve2d(arr, k, mode='full')


def _dwt2_step_multifracs(LL, gg1, hh1, nl, x0, x0appro):
    """One level of 2D DWT reproducing MULTIFRACS's ``DxPx2d.m`` border
    handling: full zero-padded convolution, arrays kept at full subsampled
    length (never trimmed), border-contaminated positions marked with a
    sentinel that survives subsequent convolutions and wins any later max.

    MULTIFRACS uses ``Inf`` as that sentinel because MATLAB's ``max``
    otherwise IGNORES ``NaN`` (`max(NaN, 5) == 5` in MATLAB). NumPy's
    ``np.max`` (used throughout ``leaders.py``, unchanged) already
    PROPAGATES NaN (`np.max([nan, 5]) -> nan`; checked directly at a Python
    prompt, standard documented NumPy behavior, not something that needs
    its own test) -- so plain ``np.nan`` already gives the sentinel-wins
    property MULTIFRACS needs ``Inf`` for. Using NaN here also means no
    analogue of DxPx2d.m's ``X(isnan(X)) = Inf`` cleanup (needed there only to fix
    Inf-Inf-cancellation artifacts back into a proper sentinel) is needed:
    NaN arithmetic never "cancels" back to a finite value.

    Parameters
    ----------
    LL : ndarray
        Current-level approximation (full array, not yet convolved).
    gg1, hh1 : ndarray, shape (nl,)
        High-pass / low-pass filter taps in MULTIFRACS's own convention
        (``gg1 = -dec_hi``, ``hh1 = dec_lo`` -- identical to
        ``rlistcoefdaub``'s taps by direct source comparison; asserted by
        `test_multifracs_taps_match_dxpx2d_source` in `tests/test_dwt2.py`).
    nl : int
        Filter length (``fp`` in DxPx2d.m -- the border-invalidation width,
        applied to the FULL pre-subsample array on every convolution).
    x0, x0appro : int
        Subsample-phase offsets: ``x0`` (=2) after a gg1 (high-pass)
        convolution, ``x0appro`` (=nl for Daubechies) after an hh1
        (low-pass) convolution -- DxPx2d.m's own fixed convention,
        independent of axis.

    Returns
    -------
    LL_next : ndarray
    detail3 : ndarray, shape (3, n_rows, n_cols) -- (LH, HL, HH)

    DxPx2d.m's isfinite-rectangle crop (lines 109-124, and again at the
    leader level, lines 168-186) is INTENTIONALLY OMITTED here -- not an
    oversight, and not merely argued to be safe: proven a no-op by direct
    comparison. Physically cropping every scale's leader array to its
    isfinite bounding rectangle and rerunning `structure.zeta_q` gave
    bit-identical zeta(q) (max abs diff 0.0 across q in [-5,5], j=3-6) and
    identical `nj` at every scale versus the uncropped, NaN-in-place arrays
    returned here. This has to be true by construction: `structure.
    structure_functions` already extracts finite values via boolean
    masking (`vals[finite]`), so a smaller physically-cropped array and a
    full-size NaN-marked one yield the exact same finite subset either
    way. MULTIFRACS's own leader recursion also never crops the array fed
    forward (only the SEPARATE, purely-reported `coef.value{j}`/`coef.nj`
    copy) -- so omitting the crop here matches both what MULTIFRACS's own
    estimation path actually consumes AND is numerically inert regardless.
    """
    njtemp = LL.shape

    def _mark_and_subsample(full, offset, axis):
        n_orig = njtemp[axis]
        front = [slice(None), slice(None)]
        front[axis] = slice(0, nl - 1)
        full[tuple(front)] = np.nan
        back = [slice(None), slice(None)]
        back[axis] = slice(n_orig, None)
        full[tuple(back)] = np.nan

        n_sub = -(-n_orig // 2)  # ceil(n_orig / 2)
        picks = (offset - 1) + 2 * np.arange(n_sub)
        idx = [slice(None), slice(None)]
        idx[axis] = picks
        return full[tuple(idx)]

    OH = _mark_and_subsample(_full_convolve_axis(LL, gg1, axis=1), x0, axis=1)
    OL = _mark_and_subsample(_full_convolve_axis(LL, hh1, axis=1), x0appro, axis=1)

    HH = _mark_and_subsample(_full_convolve_axis(OH, gg1, axis=0), x0, axis=0)
    LH = _mark_and_subsample(_full_convolve_axis(OH, hh1, axis=0), x0appro, axis=0)
    HL = _mark_and_subsample(_full_convolve_axis(OL, gg1, axis=0), x0, axis=0)
    LL_next = _mark_and_subsample(_full_convolve_axis(OL, hh1, axis=0), x0appro, axis=0)

    detail3 = np.stack([LH, HL, HH], axis=0)
    return LL_next, detail3


def wavelet_transform_2d(image, wt_name='db3', normalization=1, j2=None,
                          border='mfleaders'):
    """Compute per-scale 2D DWT detail coefficients of an image.

    Parameters
    ----------
    image : ndarray, shape (n_rows, n_cols)
    wt_name, normalization, j2 : see `_dwt.wavelet_transform_1d`.
    border : 'mfleaders' | 'multifracs'
        'mfleaders' (default, unchanged): `_dwt2_step`, delegating to
        `_dwt_step` (1D core, Gate-1-certified) -- this is what Gate 2's
        fBm2D certification (test_analytic_validation_2d.py) was validated
        against; do not change this default.
        'multifracs': `_dwt2_step_multifracs` (see there) -- MULTIFRACS's
        own border convention (mark-and-convolve, `fp=nl`), added for the
        2D cross-check against `handoff/plbmf_crosscheck/`; exercised by
        `test_multifracs_border_reaches_j6_with_matching_nj`
        (`tests/test_dwt2.py`). Not yet certified against anything else;
        purely for comparison.

    Returns
    -------
    dict[int, ndarray]
        Maps scale (1 = finest) to a detail-coefficient array of shape
        (3, n_rows_j, n_cols_j) -- the leading axis is orientation
        (LH, HL, HH). Border-corrupted entries are NaN.
    """
    image = np.asarray(image, dtype=float)
    wavelet = pywt.Wavelet(wt_name)

    n_min = min(image.shape)
    max_level = _max_decomposition_level(n_min, wavelet.dec_len, j2)

    if border == 'multifracs':
        gg1 = -np.asarray(wavelet.dec_hi)
        hh1 = np.asarray(wavelet.dec_lo)
        nl = wavelet.dec_len
        x0, x0appro = 2, nl

    coeffs = {}
    approx = image
    for scale in range(1, max_level + 1):
        if border == 'multifracs':
            approx, detail3 = _dwt2_step_multifracs(
                approx, gg1, hh1, nl, x0, x0appro)
        else:
            approx, detail3 = _dwt2_step(approx, wavelet)
        # dim=2: the L1-normalization exponent scales with dimension (a
        # d-dim wavelet's L^p norm goes as (2**(j*d))**(1/p), vs (2**(j*d))
        # **(1/2) for the orthonormal/L2 reference), not just with 1D's
        # per-axis exponent -- see the module docstring. Verified
        # empirically in tests/test_analytic_validation_2d.py (fBm2D H
        # recovery): the un-doubled exponent inflated the recovered H
        # for fBm2D by close to 2x before this fix.
        detail3 = detail3 * 2 ** (scale * 2 * (0.5 - 1 / normalization))

        if np.sum(~np.isnan(detail3)) < 3 * 3:
            break

        coeffs[scale] = detail3

    return coeffs


def combine_orientations(coeffs_by_scale):
    """Collapse the leading (3,) orientation axis via elementwise max of
    |coefficient|, BEFORE handing off to ``leaders.compute_leaders``.

    This is explicitly NOT part of the dimension-generic core (rule 2): the
    neighborhood+cone-max primitive only ever sees plain d-dim magnitude
    arrays, at dim=1 or dim=2 alike. Combining across orientation subbands
    is the standard, separate treatment multi-orientation transforms (2D/3D
    DWT) need on top of that core, per the wavelet-leader definition (Wendt,
    Roux, Jaffard & Abry 2009): the leader is a sup over the spatial
    neighborhood, the scale cone, AND all orientations, at each point.

    p=infinity ONLY. See `combine_orientations_p` for finite p -- summing
    raw coefficients across orientations (or taking max of them) is NOT the
    right operation once |.|**p enters the picture; unlike max, |.|**p does
    not commute with a later max taken over a different axis grouping.
    """
    return {j: np.max(np.abs(c), axis=0) for j, c in coeffs_by_scale.items()}


def combine_orientations_p(coeffs_by_scale, p):
    """Finite-p analogue of `combine_orientations`: sum of |coefficient|**p
    over the 3 orientations, per scale.

    Proven equivalent to MULTIFRACS's own per-orientation-then-sum-at-the-
    end structure (`DxPx2d.m` keeps `sans_voisin` and the neighborhood step
    separate per orientation throughout, combining only in the final
    ``leader.value{j} = sum(lead_value, 3)``): every step of the finite-p
    pipeline up to the single final ``**(1/p)`` root (this function, the
    cone fold, and the spatial-neighborhood step in
    `leaders.compute_leaders`) is a pure sum, and summing is fully
    associative/commutative/distributive across ANY grouping of axes --
    unlike p=infinity's max, whose equivalence under early-vs-late
    combination relies on max's idempotence specifically, this one relies
    on linearity holding at every intermediate step. So pre-combining
    orientations HERE, before `compute_leaders` ever sees the data, is
    mathematically identical to combining them only at the very end, AS
    LONG AS ``|.|**p`` is applied to each raw orientation coefficient
    exactly once (done here) and the final root is taken exactly once, at
    the very end (`compute_leaders`'s finishing step, via
    ``pre_combined_power=True``).
    """
    return {j: np.sum(np.abs(c) ** p, axis=0) for j, c in coeffs_by_scale.items()}


def leaders_2d(image, wt_name='db3', normalization=1, j2=None, p=np.inf):
    """One-call 2D wavelet-leader construction that reproduces MULTIFRACS
    end to end, DWT included, with NO keyword arguments beyond the image
    (for p=infinity; `p` is finite-p leaders, see `combine_orientations_p`
    and `leaders.compute_leaders`).

    Exists because `leaders.compute_leaders`'s own defaults (cone, weight,
    boundary) already reproduce MULTIFRACS's leader-construction convention
    on their own (fixed per audit -- see its docstring), but
    `wavelet_transform_2d`'s OWN default (``border='mfleaders'``) is
    deliberately left unchanged (changing IT would additionally move the
    DWT-level convention fBm2D's tests run under). So a bare
    ``wavelet_transform_2d(image)`` + ``compute_leaders(mag, dim=2)`` is
    STILL not a full MULTIFRACS reproduction -- the DWT-level border stays
    'mfleaders' unless asked for. This function is that ask, made explicit
    in exactly one place: ``border='multifracs'`` is set here so callers
    get genuine end-to-end MULTIFRACS reproduction by default, without
    touching `wavelet_transform_2d`'s own default for every OTHER caller.

    NOTE, precisely, on what this protects: holding THIS function's default
    fixed only protects the DWT layer. It does NOT, by itself, protect
    fBm2D's certification config as a whole -- `compute_leaders`'s own
    ``weight``/``boundary`` defaults moved (via `cone` resolution, see its
    docstring) regardless of this function's default, and fBm2D's tests
    (`tests/test_analytic_validation_2d.py`) call `compute_leaders(mag,
    dim=2)` directly with no extra keywords, so they DID move onto the new
    `weight=1.0, boundary='pad'` config. fBm2D was re-run and re-confirmed
    passing under that config on 2026-08-06 -- a re-verification, not a
    consequence of this function's own default being left alone.

    Confirmed (`tests/test_dwt2.py`'s
    `test_leaders_2d_entry_point_matches_multifracs_cascade_with_pure_defaults`):
    calling this with only the cascade field reproduces MULTIFRACS's own
    zeta(q) to machine precision (~1e-9) at every q tested.

    Parameters
    ----------
    image : ndarray, shape (n_rows, n_cols)
    wt_name, normalization, j2 : see `wavelet_transform_2d`.
    p : float
        Default ``np.inf`` (plain wavelet leaders) -- THIS PATH IS
        UNCHANGED, same two lines as before finite-p support existed.
        Any finite p > 0: finite-p leaders, via `combine_orientations_p`
        + `compute_leaders(..., p=p, pre_combined_power=True)`.

    Returns
    -------
    dict[int, ndarray]
        Leader arrays, keyed by scale (1 = finest).
    """
    from .leaders import compute_leaders

    coeffs = wavelet_transform_2d(image, wt_name=wt_name,
                                   normalization=normalization, j2=j2,
                                   border='multifracs')
    if p == np.inf:
        mag = combine_orientations(coeffs)
        return compute_leaders(mag, dim=2)

    mag_p = combine_orientations_p(coeffs, p)
    return compute_leaders(mag_p, dim=2, p=p, pre_combined_power=True)

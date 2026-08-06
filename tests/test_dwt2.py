"""2D DWT front-end sanity checks. No 2D Python reference exists (unlike
1D vs pymultifracs), so these check internal self-consistency: shapes,
alignment, dyadic halving, and boundary NaN placement.
"""

import json
import os

import numpy as np
import pytest

from mfleaders._dwt2 import wavelet_transform_2d, combine_orientations
from mfleaders.leaders import compute_leaders
from mfleaders.structure import zeta_q

HANDOFF_DIR = os.path.join(
    os.path.dirname(__file__), '..', 'handoff', 'plbmf_crosscheck')
CASCADE_MAT = os.path.join(HANDOFF_DIR, 'cascade_input.mat')
MULTIFRACS_PERSCALE_JSON = os.path.join(
    HANDOFF_DIR, 'multifracs_perscale_output.json')


def test_shapes_are_square_and_halve_dyadically():
    rng = np.random.default_rng(0)
    image = rng.standard_normal((256, 256))
    coeffs = wavelet_transform_2d(image, wt_name='db3', normalization=1)

    scales = sorted(coeffs)
    assert len(scales) >= 4

    for j in scales:
        assert coeffs[j].shape[0] == 3  # orientation axis
        assert coeffs[j].shape[1] == coeffs[j].shape[2]  # square image

    sizes = [coeffs[j].shape[1] for j in scales]
    for a, b in zip(sizes, sizes[1:]):
        assert abs(a / 2 - b) <= 2  # roughly halves each scale


def test_orientation_subbands_are_physically_aligned():
    """The three orientations at a given scale must refer to the same
    physical region: a localized feature (a single large-amplitude pixel)
    in the input should show up at CORRESPONDING (not offset) positions in
    all three orientation subbands' magnitude, once boundary-NaN positions
    are excluded.
    """
    rng = np.random.default_rng(1)
    image = rng.standard_normal((256, 256)) * 0.01
    image[128, 128] = 1000.0  # a strong, localized feature

    coeffs = wavelet_transform_2d(image, wt_name='db3', normalization=1)

    # at scale 1, the feature should dominate all three orientations at
    # essentially the same (row, col) -- not at systematically different
    # positions, which is what a leftover alignment bug would look like.
    mags = np.abs(coeffs[1])
    peak_positions = [
        np.unravel_index(np.nanargmax(mags[o]), mags[o].shape)
        for o in range(3)
    ]
    for pos in peak_positions[1:]:
        assert abs(pos[0] - peak_positions[0][0]) <= 1
        assert abs(pos[1] - peak_positions[0][1]) <= 1


def test_border_is_nan_and_interior_is_finite():
    """The alignment crop in `_dwt2_step` consumes the leading NaN border
    entirely (on whichever axis a given orientation was cropped), leaving
    only the trailing edge NaN -- consistently across all three
    orientations, which is itself a check that the crop aligned them
    (a bug would show inconsistent NaN patterns between orientations).
    """
    rng = np.random.default_rng(2)
    image = rng.standard_normal((256, 256))
    coeffs = wavelet_transform_2d(image, wt_name='db3', normalization=1)

    detail1 = coeffs[1]
    for o in range(3):
        assert np.isnan(detail1[o, -1, :]).all()
        assert np.isnan(detail1[o, :, -1]).all()

    center = detail1.shape[1] // 2
    assert np.isfinite(detail1[:, center, center]).all()

    center = detail1.shape[1] // 2
    assert np.isfinite(detail1[:, center, center]).all()


def test_combine_orientations_collapses_leading_axis():
    rng = np.random.default_rng(3)
    image = rng.standard_normal((128, 128))
    coeffs = wavelet_transform_2d(image, wt_name='db3', normalization=1)
    mag = combine_orientations(coeffs)

    for j in coeffs:
        assert mag[j].shape == coeffs[j].shape[1:]
        assert mag[j].ndim == 2
        finite = ~np.isnan(mag[j])
        np.testing.assert_allclose(
            mag[j][finite],
            np.max(np.abs(coeffs[j]), axis=0)[finite],
        )


# ---------------------------------------------------------------------------
# border='multifracs' (mark-and-convolve) -- MULTIFRACS cross-check support
# ---------------------------------------------------------------------------

def test_multifracs_taps_match_dxpx2d_source():
    """Turns the "confirmed by direct source comparison" claim into an
    executed check. Reference values are the literal Nwt=3 constants
    hardcoded in MULTIFRACS's misc/rlistcoefdaub.m (case 3), transformed by
    DxPx2d.m:19-22's own formula (`gg1 = -1*(-1).^(1:nl).*h`,
    `hh1 = fliplr(h)`) -- not re-derived, just transcribed, so this test
    is a transcription check against the .m source, not a re-verification
    of the .m source itself.

    `_dwt2.py`'s `wavelet_transform_2d(..., border='multifracs')` sets
    ``gg1 = -wavelet.dec_hi`` and ``hh1 = wavelet.dec_lo`` -- this test
    checks that mapping is exactly the same numbers as MULTIFRACS's own
    gg1/hh1, not just "the right wavelet family".
    """
    import pywt

    h = np.array([
        0.332670552950, 0.806891509311, 0.459877502118,
        -0.135011020010, -0.085441273882, 0.035226291882,
    ])
    n = np.arange(1, len(h) + 1)
    gg1_multifracs = -1 * (-1.0) ** n * h
    hh1_multifracs = h[::-1]

    wavelet = pywt.Wavelet('db3')
    gg1_mfleaders = -np.asarray(wavelet.dec_hi)
    hh1_mfleaders = np.asarray(wavelet.dec_lo)

    np.testing.assert_allclose(gg1_mfleaders, gg1_multifracs, atol=1e-10)
    np.testing.assert_allclose(hh1_mfleaders, hh1_multifracs, atol=1e-10)


@pytest.mark.skipif(not os.path.exists(CASCADE_MAT),
                     reason="handoff cascade_input.mat not present")
def test_multifracs_border_reaches_j6_with_matching_nj():
    """Exercises `_dwt2_step_multifracs`/`_full_convolve_axis` end to end
    (previously untested -- the auditor found nothing exercised this path).
    Regression-pins the two concrete facts the 2D cross-check report
    depends on: nj matches MULTIFRACS's own reported counts exactly at the
    fitted scales, and j=6 is no longer starved to nj=0 (the failure mode
    of the earlier fp-only attempt).
    """
    import scipy.io

    field = scipy.io.loadmat(CASCADE_MAT)['field'][:-1, :-1]

    coeffs = wavelet_transform_2d(field, wt_name='db3', normalization=1,
                                   border='multifracs')
    mag = combine_orientations(coeffs)
    leaders = compute_leaders(mag, dim=2, weight=1.0, boundary='pad')

    target_nj = {3: 61504, 4: 14400, 5: 3136, 6: 576}
    for j, expected in target_nj.items():
        assert j in leaders, f"scale j={j} not reached"
        nj = int(np.sum(~np.isnan(leaders[j])))
        assert nj == expected, f"j={j}: nj={nj}, expected {expected}"

    assert np.sum(~np.isnan(leaders[6])) > 0


@pytest.mark.skipif(
    not (os.path.exists(CASCADE_MAT) and os.path.exists(MULTIFRACS_PERSCALE_JSON)),
    reason="handoff cross-check artifacts not present")
def test_multifracs_cone_matches_multifracs_cascade_to_machine_precision():
    """Pins the closing result of the 2D cross-check: with the border, taps,
    dim-normalization, and (this test's own subject) cone composition all
    matched to MULTIFRACS, the cascade matches MULTIFRACS's own checked-in
    per-scale output to machine precision (~1e-8/1e-9) at every q tested,
    both branches -- not just the final regression slope zeta(q), but the
    per-scale log2 S(q,j) it's fit FROM and the regression intercept, so a
    scale-independent multiplicative offset in the leaders (which would
    move the intercept but leave the slope alone, or partially cancel
    across scales) can't pass unnoticed.

    Uses `compute_leaders`'s DEFAULT `cone` ('multifracs') deliberately, not
    passed explicitly -- this test exists specifically to pin that default's
    correctness. `weight=1.0` and `boundary='pad'` (already-adjudicated,
    separate parameters -- see test_cone_weight_adjudication.py and
    leaders.py's `_neighborhood_max` docstring) and `border='multifracs'`
    (see `_dwt2.py`) are still passed explicitly here (rather than via
    `leaders_2d`, see the separate entry-point test below): this round's
    fix is the cone composition only, and those other three were settled
    in earlier rounds, not superseded by this one.

    MULTIFRACS reference values are loaded from the checked-in
    `handoff/plbmf_crosscheck/multifracs_perscale_output.json` -- produced by
    an independent run of the actual MULTIFRACS MATLAB toolbox, not
    transcribed from memory.
    """
    import scipy.io

    field = scipy.io.loadmat(CASCADE_MAT)['field'][:-1, :-1]

    coeffs = wavelet_transform_2d(field, wt_name='db3', normalization=1,
                                   border='multifracs')
    mag = combine_orientations(coeffs)
    leaders = compute_leaders(mag, dim=2, weight=1.0, boundary='pad')

    Q = np.concatenate([np.arange(-5, 0), np.arange(1, 6)]).astype(float)
    j1, j2 = 3, 6
    scales, log2_s, nj, zeta, intercept = zeta_q(leaders, Q, j1, j2)
    fit_mask = (scales >= j1) & (scales <= j2)

    mf = json.load(open(MULTIFRACS_PERSCALE_JSON))
    per_q = mf['per_q']
    mf_by_q = {v['q']: v for v in per_q.values()}

    for i, qq in enumerate(Q):
        assert qq in mf_by_q, f"MULTIFRACS reference missing q={qq}"
        ref = mf_by_q[qq]

        diff_zeta = abs(zeta[i] - ref['zeta_slope'])
        assert diff_zeta < 1e-9, (
            f"q={qq}: zeta mfleaders={zeta[i]!r}, "
            f"MULTIFRACS={ref['zeta_slope']!r}, diff={diff_zeta:.3e} "
            f"(expected machine precision, ~1e-9)")

        diff_intercept = abs(intercept[i] - ref['intercept_aest'])
        assert diff_intercept < 1e-8, (
            f"q={qq}: intercept mfleaders={intercept[i]!r}, "
            f"MULTIFRACS={ref['intercept_aest']!r}, "
            f"diff={diff_intercept:.3e} (expected machine precision, ~1e-8)")

        mine_log2s = log2_s[fit_mask, i]
        theirs_log2s = np.array(ref['log2_S_qj'])
        diff_log2s = np.abs(mine_log2s - theirs_log2s)
        assert diff_log2s.max() < 1e-8, (
            f"q={qq}: log2_S(q,j) mfleaders={mine_log2s!r}, "
            f"MULTIFRACS={theirs_log2s!r}, max diff={diff_log2s.max():.3e} "
            f"(expected machine precision, ~1e-8)")


@pytest.mark.skipif(
    not (os.path.exists(CASCADE_MAT) and os.path.exists(MULTIFRACS_PERSCALE_JSON)),
    reason="handoff cross-check artifacts not present")
def test_leaders_2d_entry_point_matches_multifracs_cascade_with_pure_defaults():
    """The audited requirement: calling `leaders_2d` with NO keyword
    arguments beyond the field itself must reproduce MULTIFRACS end to end
    (DWT border included), not just the leader-construction layer in
    isolation (that's the test above). Confirms `leaders_2d`'s own
    docstring claim is actually true, not just asserted.
    """
    import scipy.io

    from mfleaders._dwt2 import leaders_2d

    field = scipy.io.loadmat(CASCADE_MAT)['field'][:-1, :-1]

    leaders = leaders_2d(field)

    Q = np.concatenate([np.arange(-5, 0), np.arange(1, 6)]).astype(float)
    j1, j2 = 3, 6
    _, _, _, zeta, _ = zeta_q(leaders, Q, j1, j2)

    mf = json.load(open(MULTIFRACS_PERSCALE_JSON))
    per_q = mf['per_q']
    mf_zeta = {v['q']: v['zeta_slope'] for v in per_q.values()}

    for i, qq in enumerate(Q):
        diff = abs(zeta[i] - mf_zeta[qq])
        assert diff < 1e-9, (
            f"q={qq}: mfleaders={zeta[i]!r}, MULTIFRACS={mf_zeta[qq]!r}, "
            f"diff={diff:.3e} (expected machine precision, ~1e-9)")

"""Smoke tests for mfleaders.lacunarity -- a diagnostic, NOT cross-validated
against an external reference implementation (see the module's own
docstring). These only confirm the code path behaves sensibly (smooth field
-> Lambda near 1; structured/sparse field -> Lambda well above 1; both
decreasing with box size; the mask-erosion exterior-exclusion path produces
no NaN/Inf contamination when a real invalid region is present) -- they are
not a certification of correctness against any independent implementation.
"""

import numpy as np

from mfleaders.lacunarity import gliding_box_lacunarity_curve

BOX_SIZES = [2, 4, 8, 16, 32]


def test_smooth_field_lacunarity_near_one_and_decreasing():
    rng = np.random.default_rng(0)
    field = 1.0 + 0.01 * rng.standard_normal((128, 128))

    lam, _ = gliding_box_lacunarity_curve(field, BOX_SIZES)

    assert np.all(np.isfinite(lam))
    assert np.all(lam < 1.05), f"expected near-1 lacunarity for a smooth field, got {lam}"
    # non-increasing with box size, allowing tiny float noise
    assert np.all(np.diff(lam) <= 1e-9), f"expected non-increasing Lambda(r), got {lam}"


def test_structured_sparse_field_lacunarity_well_above_one():
    rng = np.random.default_rng(1)
    field = np.full((128, 128), 0.01)
    ys = rng.integers(0, 128, size=250)
    xs = rng.integers(0, 128, size=250)
    field[ys, xs] = 10.0

    lam, _ = gliding_box_lacunarity_curve(field, BOX_SIZES)

    assert np.all(np.isfinite(lam))
    assert lam[0] > 1.2, f"expected well-above-1 lacunarity at the smallest box for a sparse field, got {lam[0]}"
    assert np.all(np.diff(lam) <= 1e-9), f"expected non-increasing Lambda(r), got {lam}"


def test_structured_field_more_lacunar_than_smooth_field():
    rng_smooth = np.random.default_rng(0)
    smooth = 1.0 + 0.01 * rng_smooth.standard_normal((128, 128))

    rng_structured = np.random.default_rng(1)
    structured = np.full((128, 128), 0.01)
    ys = rng_structured.integers(0, 128, size=250)
    xs = rng_structured.integers(0, 128, size=250)
    structured[ys, xs] = 10.0

    lam_smooth, _ = gliding_box_lacunarity_curve(smooth, BOX_SIZES)
    lam_structured, _ = gliding_box_lacunarity_curve(structured, BOX_SIZES)

    assert np.all(lam_structured > lam_smooth)


def test_mask_erosion_excludes_exterior_without_nan_contamination():
    """A genuine invalid (NaN) region, with an explicit valid_mask matching
    it: confirms the erosion-based exclusion actually keeps NaN out of
    every reported Lambda(r), not just at trivially-small box sizes -- this
    is the path an off-by-one between uniform_filter's and binary_erosion's
    window conventions would break, so it's checked directly rather than
    assumed from reading the code.
    """
    rng = np.random.default_rng(2)
    field = 1.0 + 0.01 * rng.standard_normal((128, 128))
    field[:64, :64] = np.nan  # a genuine masked quadrant, not zero-filled
    valid_mask = np.isfinite(field)

    lam, _ = gliding_box_lacunarity_curve(field, BOX_SIZES, valid_mask=valid_mask)

    assert np.all(np.isfinite(lam)), (
        f"NaN/Inf leaked into Lambda(r) despite mask erosion -- possible "
        f"off-by-one between uniform_filter's and binary_erosion's window "
        f"conventions: {lam}"
    )
    # the valid (non-NaN) region is itself smooth, so Lambda should still
    # read out close to 1 -- the mask must not be inflating it.
    assert np.all(lam < 1.1), f"expected near-1 lacunarity on the valid region, got {lam}"


def test_binarize_median_gives_50_percent_coverage_regardless_of_field():
    """Documents the known 'median' caveat directly (see module docstring):
    thresholding at the field's own median always yields ~50% coverage by
    construction, for ANY field with a continuous (non-tied) value
    distribution -- this is not a bug, but a property to be aware of
    before trusting 'median' binarization to reveal true void structure.

    Uses a continuously-varying background (not a single repeated value)
    so the median doesn't land exactly on a tied background value, which
    would itself (correctly) break the ~50% property -- median-tie
    breaking is a real edge case, not what this test is checking.
    """
    rng = np.random.default_rng(3)
    structured = 0.01 + 0.001 * rng.standard_normal((128, 128))
    ys = rng.integers(0, 128, size=250)
    xs = rng.integers(0, 128, size=250)
    structured[ys, xs] = 10.0

    _, threshold = gliding_box_lacunarity_curve(structured, BOX_SIZES, binarize="median")
    coverage = (structured > threshold).mean()
    assert abs(coverage - 0.5) < 0.01

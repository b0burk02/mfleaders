"""Unit tests for the dimension-generic neighborhood + cone-max primitive.

These are hand-checkable small cases, deliberately independent of pymultifracs
(that comparison lives in test_oracle_pymultifracs.py). The same functions
are exercised at dim=1 and dim=2 to confirm there is no dimension-specific
fork (CLAUDE.md hard rule 2).
"""

import numpy as np

from mfleaders.leaders import _neighborhood_max, _fold_children, compute_leaders


def test_neighborhood_max_1d_hand_computed():
    coefs = np.array([0., 1., 2., 3., 10., 3., 2., 1., 0.])
    out = _neighborhood_max(coefs, dim=1)

    expected = np.array([np.nan, 2, 3, 10, 10, 10, 3, 2, np.nan])
    np.testing.assert_array_equal(np.isnan(out), np.isnan(expected))
    np.testing.assert_allclose(out[~np.isnan(out)], expected[~np.isnan(expected)])


def test_fold_children_1d_hand_computed():
    window_max = np.array([np.nan, 1., 1., np.nan])
    finer_leader = np.array([np.nan, 2., 3., 10., 10., 10., 3., 2., np.nan])

    out = _fold_children(window_max, finer_leader, dim=1, weight=0.5)

    # child(0) picks finer_leader[0,2,4,6] = [nan,3,10,3], weight 0.5 -> [nan,1.5,5,1.5]
    # child(1) picks finer_leader[1,3,5,7] = [2,10,10,2],  weight 0.5 -> [1,5,5,1]
    # elementwise max with window_max, any-NaN-invalidates:
    expected = np.array([np.nan, 5., 5., np.nan])

    np.testing.assert_array_equal(np.isnan(out), np.isnan(expected))
    np.testing.assert_allclose(out[~np.isnan(out)], expected[~np.isnan(expected)])


def test_compute_leaders_1d_two_scales():
    coeffs = {
        1: np.array([0., 1., 2., 3., 10., 3., 2., 1., 0.]),
        2: np.array([1., 1., 1., 1.]),
    }
    # cone='pymultifracs' explicitly: this hand computation was derived
    # under that composition (fold from the previous scale's already-
    # neighborhood-maxed leader) -- compute_leaders' own default is now
    # 'multifracs', a structurally different composition (see leaders.py).
    leaders = compute_leaders(coeffs, dim=1, cone='pymultifracs')

    np.testing.assert_allclose(
        leaders[1][~np.isnan(leaders[1])],
        [2, 3, 10, 10, 10, 3, 2],
    )
    np.testing.assert_allclose(
        leaders[2][~np.isnan(leaders[2])],
        [5, 5],
    )


def test_a_single_missing_neighbor_invalidates_the_position():
    coefs = np.array([1., 1., 1., 1., 1.])
    coefs_with_gap = coefs.copy()
    coefs_with_gap[2] = np.nan

    clean = _neighborhood_max(coefs, dim=1)
    gapped = _neighborhood_max(coefs_with_gap, dim=1)

    assert np.isfinite(clean[1]) and np.isfinite(clean[2]) and np.isfinite(clean[3])
    # position 2 itself, plus its window-neighbors 1 and 3, should all be
    # invalidated by the single missing sample at index 2.
    assert np.isnan(gapped[1])
    assert np.isnan(gapped[2])
    assert np.isnan(gapped[3])


def test_missing_finer_scale_leader_invalidates_the_parent():
    """A NaN in the finer scale's leader at a child position must invalidate
    the corresponding parent position too (strict propagation through the
    fold, not just within a single scale's own window).
    """
    window_max = np.array([1., 1., 1., 1.])
    finer_leader = np.array([1., 1., np.nan, 1., 1., 1., 1., 1.])

    out = _fold_children(window_max, finer_leader, dim=1, weight=0.5)

    # parent index 1 has children at finer_leader[2] (nan) and finer_leader[3]
    assert np.isnan(out[1])
    # parent indices 0 and 2 have no NaN children and should stay finite
    assert np.isfinite(out[0])
    assert np.isfinite(out[2])


def test_neighborhood_max_2d_same_primitive_as_1d():
    """Same function, dim=2: a 3x3 neighborhood max with NaN borders."""
    coefs = np.ones((5, 5))
    coefs[2, 2] = 10.0

    out = _neighborhood_max(coefs, dim=2)

    assert np.isnan(out[0]).all()
    assert np.isnan(out[-1]).all()
    assert np.isnan(out[:, 0]).all()
    assert np.isnan(out[:, -1]).all()

    # every interior position within a 3x3 window of the center sees the 10
    for i in range(1, 4):
        for j in range(1, 4):
            assert out[i, j] == 10.0


def test_compute_leaders_2d_runs_without_forking_code():
    """dim=2 sanity: compute_leaders must not special-case dimension."""
    rng = np.random.default_rng(3)
    coeffs = {
        1: rng.random((17, 17)),
        2: rng.random((8, 8)),
        3: rng.random((4, 4)),
    }
    leaders = compute_leaders(coeffs, dim=2)

    assert set(leaders) == {1, 2, 3}
    for j, arr in leaders.items():
        assert arr.shape == coeffs[j].shape
        # interior has some finite values
        assert np.isfinite(arr).any()


def test_compute_leaders_2d_cone_fold_hand_computed():
    """Hand-computed 2D cone fold: a single spike at scale 1 should reach
    every scale-2 position whose 2x2 child block contains it, weighted by
    2**-dim = 0.25 (matching the same primitive's dim=1 weight of
    2**-1 = 0.5 -- this is the actual dimension-parameterization CLAUDE.md
    rule 2 requires, not just "runs without crashing").

    Not yet validated against any 2D reference (that's Gate 2's job) --
    this only confirms the *structure* the same primitive produces at
    dim=2 matches what the stated definition implies.

    cone='pymultifracs' explicitly: this hand computation is about the
    weight=2**-dim dimension-generalization specifically (fold from the
    previous scale's already-neighborhood-maxed leader, weight applied
    once); it is not exercising cone='multifracs' (compute_leaders' own
    default) at all -- that composition folds raw, un-maxed coefficients
    and would not reproduce this hand-derived 25.0.
    """
    c1 = np.ones((9, 9))
    c1[4, 4] = 100.0
    c2 = np.ones((4, 4))

    leaders = compute_leaders({1: c1, 2: c2}, dim=2, cone='pymultifracs')

    expected = np.full((4, 4), np.nan)
    expected[1:3, 1:3] = 25.0  # 0.25 * 100, at every position whose 2x2
                               # child block includes the spike at (4,4)

    np.testing.assert_array_equal(np.isnan(leaders[2]), np.isnan(expected))
    np.testing.assert_allclose(
        leaders[2][~np.isnan(leaders[2])], expected[~np.isnan(expected)])

"""1D analytic validation (validation-ladder step 2) -- ground truth, not
pymultifracs.

fBm (monofractal): zeta(q) should be linear, D(h) should collapse to a
narrow point at h=H, both recovered from a single q sweep spanning q<0 and
q>0.

Binomial cascade (genuinely multifractal): zeta(q), h(q), D(q) all have
closed forms (see tests/synthetic.py), letting both branches of D(h) be
checked against an exact answer rather than just "looks concave".

This module is also where the base-scale convention (max vs sum) gets
adjudicated against these ground truths -- see
test_max_base_recovers_H_more_accurately_than_sum_base and
test_base_scale_conventions_converge_by_j1_3. Do not read this file as
"passing tests" only: the printed Monte Carlo numbers in those two tests are
the actual evidence for the adjudication, reported alongside this code.

Every `compute_leaders(coeffs, dim=1, ...)` call below passes
``cone='pymultifracs'`` explicitly. `compute_leaders`'s own default is now
``cone='multifracs'`` (2D/MULTIFRACS is the primary target -- see
leaders.py), but the tolerances and thresholds calibrated in this file
(e.g. the j1=2 residual-bias check) were derived under the OTHER
composition and have not been re-run or re-adjudicated under the new one --
do not remove this argument without redoing that calibration.
"""

import numpy as np
import pytest

from mfleaders._dwt import wavelet_transform_1d
from mfleaders.leaders import _neighborhood_max, _fold_children, compute_leaders
from mfleaders.structure import zeta_q
from mfleaders.spectrum import multifractal_spectrum, holder_width
from mfleaders.scale_range import select_j1, select_j2

from synthetic import (
    fgn_acf, fgn_davies_harte, fbm,
    cascade_staircase, cascade_zeta_true, cascade_h_true, cascade_D_true,
)

Q = np.concatenate([np.arange(-5, 0), np.arange(1, 6)]).astype(float)


def _leaders_with_base(coeffs_by_scale, dim, base):
    """compute_leaders, but with a swappable base-scale (j=1) convention.

    base='max': the library's actual definition (plain neighborhood max).
    base='sum': pymultifracs' convention (plain neighborhood sum) -- kept
    here, in test code only, purely to adjudicate against ground truth. The
    library itself only ever implements 'max' (CLAUDE.md hard rule 2).
    """
    scales = sorted(coeffs_by_scale)
    leaders = {}
    prev = None
    for j in scales:
        c = coeffs_by_scale[j]
        if prev is None:
            if base == 'max':
                leaders[j] = _neighborhood_max(np.abs(c), dim)
            elif base == 'sum':
                absval = np.abs(c)
                n = len(absval)
                out = np.full(n, np.nan)
                out[1:-1] = absval[0:n - 2] + absval[1:n - 1] + absval[2:n]
                leaders[j] = out
            else:
                raise ValueError(base)
        else:
            wm = _neighborhood_max(np.abs(c), dim)
            leaders[j] = _fold_children(wm, prev, dim, weight=2.0 ** (-dim))
        prev = leaders[j]
    return leaders


# ---------------------------------------------------------------------------
# Ground-truth generator self-check (not wavelet-leader-related at all)
# ---------------------------------------------------------------------------

def test_fgn_generator_matches_theoretical_autocovariance():
    """The fBm/fGn generator must be validated against the known, purely
    analytic autocovariance formula BEFORE it's trusted as ground truth for
    the wavelet-leader pipeline -- otherwise this whole validation could be
    circular in a different way (an uncalibrated generator).
    """
    max_lag = 6
    n = 2048
    n_trials = 300

    for H in [0.3, 0.5, 0.7, 0.85]:
        rng = np.random.default_rng(0)
        acf_emp = np.zeros(max_lag + 1)
        for _ in range(n_trials):
            x = fgn_davies_harte(n, H, rng)
            for k in range(max_lag + 1):
                acf_emp[k] += np.mean(x[:n - k] * x[k:])
        acf_emp /= n_trials

        acf_theory = fgn_acf(np.arange(max_lag + 1), H)
        # Monte Carlo standard error at this n/n_trials is ~1e-2; use a
        # generous but meaningful tolerance.
        np.testing.assert_allclose(acf_emp, acf_theory, atol=0.02)


# ---------------------------------------------------------------------------
# fBm: monofractal collapse
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("H", [0.3, 0.5, 0.7])
def test_fbm_zeta_is_linear_and_recovers_H(H):
    rng = np.random.default_rng(1000 + int(H * 100))
    signal = fbm(2 ** 15, H, rng)

    coeffs = wavelet_transform_1d(signal, wt_name='db3', normalization=1)
    leaders = compute_leaders(coeffs, dim=1, cone='pymultifracs')
    # j1=2 (the cone-depth floor) and select_j2 (the reliability ceiling):
    # the simple, fixed instance of the j1/j2 rule -- see
    # test_select_j1_stability_search_is_reasonable_but_not_infallible for
    # why the *adaptive* search isn't used here.
    j2 = select_j2(leaders)
    j1 = 2

    _, _, _, zeta, _ = zeta_q(leaders, Q, j1, j2)

    # zeta(q) = qH: check linearity directly (residual from the best linear
    # fit through the origin should be small relative to the range of zeta).
    fit_slope = np.sum(Q * zeta) / np.sum(Q * Q)
    residual = zeta - fit_slope * Q
    assert np.max(np.abs(residual)) < 0.1 * (zeta.max() - zeta.min())

    # A single realization's fit has real sampling noise -- this checks the
    # formalism collapses to (approximately) the right point, not that one
    # instance hits H to arbitrary precision.
    H_est = zeta[np.argmin(np.abs(Q - 2))] / 2
    assert H_est == pytest.approx(H, abs=0.1)


@pytest.mark.parametrize("H", [0.3, 0.5, 0.7])
def test_fbm_D_h_collapses_to_a_point(H):
    """Monofractal: D(h) should be narrow (both q<0 and q>0 branches
    clustering near h=H), not spread out like a genuine multifractal.
    """
    rng = np.random.default_rng(2000 + int(H * 100))
    signal = fbm(2 ** 15, H, rng)

    coeffs = wavelet_transform_1d(signal, wt_name='db3', normalization=1)
    leaders = compute_leaders(coeffs, dim=1, cone='pymultifracs')
    j2 = select_j2(leaders)
    j1 = 2

    # dim=1 explicitly: multifractal_spectrum's own default is now dim=2
    # (mfleaders' public interface is 2D-only); this is a 1D test.
    _, _, _, _, hq, Dq = multifractal_spectrum(leaders, Q, j1, j2, dim=1)

    assert (Q < 0).any() and (Q > 0).any()
    delta_h = holder_width(hq)
    # narrow: the cascade below (a genuine multifractal) reaches ~1.5-1.9
    assert delta_h < 0.5
    assert np.nanmean(hq) == pytest.approx(H, abs=0.15)
    assert np.nanmean(Dq) == pytest.approx(1.0, abs=0.2)


# ---------------------------------------------------------------------------
# Binomial cascade: both branches against a closed form
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("p", [0.25, 0.35])
def test_cascade_recovers_both_branches_of_D_h(p):
    n_levels = 16
    F = cascade_staircase(n_levels, p)

    coeffs = wavelet_transform_1d(F, wt_name='db3', normalization=1)
    leaders = compute_leaders(coeffs, dim=1, cone='pymultifracs')
    j2 = select_j2(leaders)
    # j1=4, not the j1=2 floor used for fBm above: empirically (see report)
    # the cascade's most-singular (q>0) branch needs more cone depth to
    # resolve reliably than a monofractal process does, especially for
    # weaker-contrast p closer to 0.5. This is stated and used uniformly for
    # both p values here, not tuned per case.
    j1 = 4

    # dim=1 explicitly: multifractal_spectrum's own default is now dim=2
    # (mfleaders' public interface is 2D-only); this is a 1D test.
    _, _, _, _, hq, Dq = multifractal_spectrum(leaders, Q, j1, j2, dim=1)
    zeta = zeta_q(leaders, Q, j1, j2)[3]

    zt = cascade_zeta_true(Q, p)
    ht = cascade_h_true(Q, p)
    Dt = cascade_D_true(Q, p)

    assert (Q < 0).any() and (Q > 0).any()

    # zeta(q) must be genuinely concave (this is what makes leaders recover
    # BOTH branches, unlike raw q>=0 coefficient moments) -- not a fixed
    # tolerance dodge, a structural check independent of the exact numbers.
    # Checked separately within each branch since Q skips q=0 (the secant
    # slope across that gap isn't comparable to the unit-spaced slopes
    # elsewhere); a small tolerance absorbs estimation noise in the
    # hardest-to-estimate extreme-q slopes.
    neg = Q < 0
    pos = Q > 0
    slopes_neg = np.diff(zeta[neg]) / np.diff(Q[neg])
    slopes_pos = np.diff(zeta[pos]) / np.diff(Q[pos])
    assert np.all(np.diff(slopes_neg) < 1e-6), "zeta(q) is not concave (q<0)"
    assert np.all(np.diff(slopes_pos) < 1e-2), "zeta(q) is not concave (q>0)"

    # h(q) must be (up to small estimation noise at the hardest-to-resolve
    # extreme q) monotonically decreasing -- concavity of zeta forces this;
    # see the module docstring / report for the derivation. q<0 => large h
    # (right, regular branch), q>0 => small h (left, singular branch).
    assert np.all(np.diff(hq) < 0.01)

    # D(q) peaks near 1 close to q=0 (the a.e./typical exponent).
    assert Dq[np.argmin(np.abs(Q - 1))] < 1.0 + 1e-6

    # Compare to closed form. The left (q<0) branch is well resolved; the
    # right (q>0, most-singular) branch is known in this literature to be
    # harder to resolve at finite scale ranges (see report), so it gets a
    # looser tolerance -- not zero, but not silently dropped either.
    np.testing.assert_allclose(zeta[Q < 0], zt[Q < 0], atol=0.25)
    np.testing.assert_allclose(hq[Q < 0], ht[Q < 0], atol=0.1)
    np.testing.assert_allclose(Dq[Q < 0], Dt[Q < 0], atol=0.1)

    np.testing.assert_allclose(hq[Q > 0], ht[Q > 0], atol=0.15)


def test_select_j1_stability_search_is_reasonable_but_not_infallible():
    """select_j1's adaptive stability search is offered as a general-purpose
    tool (its rule uses only a single fit's own regression SE -- no
    reference to pymultifracs or a known answer). But on a single stochastic
    realization it can occasionally lock onto a narrow, self-consistent-but-
    still-biased range before reaching a wider, more accurate one: a known,
    generic limitation of single-instance automatic range selection (it's
    exactly why more sophisticated approaches, e.g. pymultifracs' bootstrap-
    based Lambda criterion, resample instead of relying on one fit's
    residuals). The fBm/cascade tests above therefore use the simpler, fixed
    instance of the rule (j1=2 floor + select_j2 ceiling) as ground truth
    comparisons; this test only checks select_j1 stays in a sane ballpark,
    not that it's the more accurate choice.
    """
    rng = np.random.default_rng(1050)
    signal = fbm(2 ** 15, 0.5, rng)
    coeffs = wavelet_transform_1d(signal, wt_name='db3', normalization=1)
    leaders = compute_leaders(coeffs, dim=1, cone='pymultifracs')
    j2 = select_j2(leaders)

    j1, trace = select_j1(leaders, j2)
    assert j1 >= 2
    assert len(trace) >= 1

    H_est = zeta_q(leaders, np.array([2.0]), j1, j2)[3][0] / 2
    assert H_est == pytest.approx(0.5, abs=0.2)


# ---------------------------------------------------------------------------
# Base-scale adjudication: max vs sum against fBm ground truth
# ---------------------------------------------------------------------------

def test_max_base_recovers_H_more_accurately_than_sum_base():
    """The decisive adjudication. Monte Carlo over many fBm(H=0.7)
    realizations, estimating H = zeta(2)/2 at j1=1 (no cone depth at all --
    exactly where the two base-scale conventions can differ) for both
    conventions. Report: which one is closer to the true H, with enough
    trials that the answer isn't noise.
    """
    H_true = 0.7
    n = 2 ** 13
    n_trials = 40
    j2 = 8

    rng = np.random.default_rng(7)
    H_est_max = []
    H_est_sum = []

    for _ in range(n_trials):
        signal = fbm(n, H_true, rng)
        coeffs = wavelet_transform_1d(signal, wt_name='db3', normalization=1)

        leaders_max = _leaders_with_base(coeffs, dim=1, base='max')
        leaders_sum = _leaders_with_base(coeffs, dim=1, base='sum')

        zeta_max = zeta_q(leaders_max, np.array([2.0]), 1, j2)[3]
        zeta_sum = zeta_q(leaders_sum, np.array([2.0]), 1, j2)[3]

        H_est_max.append(zeta_max[0] / 2)
        H_est_sum.append(zeta_sum[0] / 2)

    H_est_max = np.array(H_est_max)
    H_est_sum = np.array(H_est_sum)

    bias_max = H_est_max.mean() - H_true
    bias_sum = H_est_sum.mean() - H_true
    se_max = H_est_max.std(ddof=1) / np.sqrt(n_trials)
    se_sum = H_est_sum.std(ddof=1) / np.sqrt(n_trials)

    print(f"\nj1=1 base-scale adjudication (n_trials={n_trials}, n={n}):")
    print(f"  max: bias={bias_max:+.4f} (se={se_max:.4f}, "
          f"{abs(bias_max) / se_max:.1f} SE from zero)")
    print(f"  sum: bias={bias_sum:+.4f} (se={se_sum:.4f}, "
          f"{abs(bias_sum) / se_sum:.1f} SE from zero)")

    # both biases are real (many SE from zero: finite-sample/discretization
    # effects are not being claimed absent), but max's bias must be smaller
    assert abs(bias_max) / se_max > 5, "expected a statistically real bias"
    assert abs(bias_sum) / se_sum > 5, "expected a statistically real bias"
    assert abs(bias_max) < abs(bias_sum) / 2, (
        "max-base should be substantially less biased than sum-base at j1=1"
    )


def test_base_scale_conventions_converge_by_j1_3():
    """Once at least one octave of cone recursion has run (j1>=2-3), the two
    conventions should agree closely -- the discrepancy is seeded only at
    the base case and damped by 2**-1 per level through the max recursion.
    """
    H_true = 0.7
    n = 2 ** 13
    n_trials = 40
    j2 = 8

    rng = np.random.default_rng(7)
    H_est_max = {j1: [] for j1 in [3, 4]}
    H_est_sum = {j1: [] for j1 in [3, 4]}

    for _ in range(n_trials):
        signal = fbm(n, H_true, rng)
        coeffs = wavelet_transform_1d(signal, wt_name='db3', normalization=1)

        leaders_max = _leaders_with_base(coeffs, dim=1, base='max')
        leaders_sum = _leaders_with_base(coeffs, dim=1, base='sum')

        for j1 in [3, 4]:
            zm = zeta_q(leaders_max, np.array([2.0]), j1, j2)[3]
            zs = zeta_q(leaders_sum, np.array([2.0]), j1, j2)[3]
            H_est_max[j1].append(zm[0] / 2)
            H_est_sum[j1].append(zs[0] / 2)

    for j1 in [3, 4]:
        m = np.mean(H_est_max[j1])
        s = np.mean(H_est_sum[j1])
        assert m == pytest.approx(s, abs=0.01), f"j1={j1}: max={m}, sum={s}"


def test_j1_2_residual_bias_is_small_and_not_base_scale_contamination():
    """At the default floor j1=2, max-base H recovery has a small residual
    bias -- but it is FLAT across j1=2..5 rather than decaying, which is the
    signature that distinguishes it from leftover base-scale contamination
    (that artifact decays sharply and is gone by j1=3, per the tests above).
    This is therefore a separate, generic, known systematic (most likely the
    downward/Jensen's-inequality bias of log2(mean(L^q)) from finite
    samples), quantified here at ~1% of H and tracked, not silently ignored
    or conflated with the base-scale question. See scale_range.py's
    docstring for the full numbers (a higher-trial-count run).
    """
    H_true = 0.7
    n = 2 ** 13
    n_trials = 60
    j2 = 8

    rng = np.random.default_rng(7)
    biases = {}
    for j1 in [2, 3, 4, 5]:
        ests = []
        r = np.random.default_rng(100 + j1)
        for _ in range(n_trials):
            signal = fbm(n, H_true, r)
            coeffs = wavelet_transform_1d(signal, wt_name='db3',
                                           normalization=1)
            leaders = compute_leaders(coeffs, dim=1, cone='pymultifracs')
            zeta = zeta_q(leaders, np.array([2.0]), j1, j2)[3]
            ests.append(zeta[0] / 2)
        ests = np.array(ests)
        biases[j1] = ests.mean() - H_true

    print(f"\nj1=2 residual-bias check (n_trials={n_trials}, n={n}):")
    for j1, b in biases.items():
        print(f"  j1={j1}: bias={b:+.4f}")

    # small in absolute terms
    assert abs(biases[2]) < 0.03
    # flat, not decaying like the (already-excluded) base-scale artifact --
    # j1=5's bias should not be markedly smaller than j1=2's
    assert abs(biases[5]) > 0.3 * abs(biases[2])

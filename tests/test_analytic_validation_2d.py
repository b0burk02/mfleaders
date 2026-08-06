"""2D analytic validation (Gate 2, validation-ladder step 3) -- ground
truth, no Python 2D reference exists (CLAUDE.md notes this explicitly).

STATUS (read before trusting any "pass" here): two real, dimension-scaling
bugs were found and fixed while building this:

  1. `_dwt2.wavelet_transform_2d`'s L1-normalization exponent needs an
     extra factor of `dim` (a d-dim wavelet's L^p-norm coefficient scales
     as (2**(scale*d))**(1/p) vs (2**(scale*d))**(1/2) for the L2/
     orthonormal reference -- the exponent must scale with d, not just
     copy the 1D formula). Missing this inflated recovered H for fBm2D by
     close to 2x.
  2. `spectrum.multifractal_spectrum`'s `Dq = Dq + 1` was hardcoded to the
     1D ambient dimension; it needs `+ dim`. Missing this capped 2D D(q)
     near 1 instead of 2.

Both are confirmed by fBm2D (isotropic monofractal, both branches
collapsing correctly to h=H, D=2 -- see the fbm2d tests below, which now
pass cleanly).

The cone weight 2**-dim is a THIRD thing this was supposed to adjudicate
directly against a closed-form 2D multifractal spectrum (the way max-vs-sum
was adjudicated at dim=1), and that part is NOT resolved -- the cascade
does not yet recover its closed form, so per explicit instruction the cone
weight is NOT re-adjudicated against it (that would be fixing one unknown
by tuning against another). See scale_range/leaders.py: the weight remains
supported only by fBm2D showing it is "not worse than alternatives" (weak)
plus the same dimension-scaling mechanism confirmed for bug #1 above.

ORIENTATION-COMBINATION CHECK (done first, per instruction, before touching
anything else): Wendt, Roux, Jaffard & Abry (2009), "Wavelet leaders and
bootstrap for multifractal analysis of images" -- define the 2D wavelet
leader as

    L(j,k) = sup_{lambda' subset 3*lambda_{j,k}} sup_{m in {1,2,3}} |d_lambda'^(m)|

i.e. the sup ranges jointly over the spatial neighborhood (3x3 dyadic
cubes), all finer scales in the cone, AND all three orientation subbands
(m = LH, HL, HH) -- no orientation is privileged over another; a max over
three genuinely independent index sets is order-independent (max is
associative), so "combine orientations, then neighborhood+cone" and
"combine everything jointly" give identical results. `combine_orientations`
(np.max(|c|, axis=0) over the 3 orientations) + `compute_leaders` implement
exactly this. CONFIRMED: no divergence from the reference, no implicit
HH-dominance in the code -- the combination is correct max-modulus per
Wendt/Roux/Jaffard/Abry, as written, before any of the investigation below.

Given the code matches the reference, the discrepancy must be in the
CLOSED-FORM ground truth, not the combination rule. Two hypotheses were
tested against the separable cascade's measured h(q):
  (a) "sum" (my original assumption): local exponent = hx + hy (the
      product measure's joint mass exponent; assumes HH asymptotically
      dominates LH/HL).
  (b) "min": local exponent = min(hx, hy) (the classical POINTWISE Holder
      exponent of a 2D function -- for F(x,y)=Fx(x)Fy(y), a Taylor-type
      expansion near (x0,y0) gives |Delta F| ~ max(r^hx, r^hy, r^(hx+hy)),
      and since r<1 as r->0, the term with the SMALLEST exponent dominates,
      i.e. r^min(hx,hy) -- matching the user's point that edges/filaments
      (a direction with locally low regularity) legitimately show up in
      LH/HL, not just HH).
Neither hypothesis fully explains the data: for the separable cascade, the
NEGATIVE-q (smooth) branch matches hypothesis (a) closely (measured h(-5)
~4.0 vs predicted 2*h_max_1D=4.0), but the POSITIVE-q (singular) branch is
SMALLER than even hypothesis (b) predicts. For the non-separable (4-weight)
cascade, NEITHER branch matches either hypothesis's "joint" closed form
(measured h ~1.2-1.3 at extreme negative q vs a joint-formula prediction of
~3.2-3.3).

RESOLVED (was "genuinely unresolved" above; kept for the historical record,
not rewritten): the pipeline is not the problem. An independent, external,
published toolbox (MULTIFRACS, MATLAB) was cross-checked against on the
separable p=0.25 cascade (handoff/plbmf_crosscheck/) and matches mfleaders
to machine precision (~1e-9), see tests/test_dwt2.py. Since MULTIFRACS
*also* diverges from cascade_zeta_true_2d/cascade_h_true_2d at the same
scale (confirmed numerically, see
test_cascade2d_measure_spectrum_bridge_is_a_confirmed_false_conjecture
below), the closed-form target -- not the wavelet-leader construction -- is
the false conjecture. This settles what was previously the strict xfail
(test_cascade2d_quantitative_match_is_unresolved, now retired/renamed to
that test) as a confirmed negative result, not an open question.

GATE 2 STATUS, recorded honestly (do not silently upgrade this): what is
now certified is REFERENCE PARITY with MULTIFRACS on the 2D cascade -- a
different, weaker claim than an independent ANALYTIC ground-truth match.
The 2D multifractal analytic validation rung (a wavelet-leader spectrum
matching a properly-derived, from-first-principles 2D closed form) remains
UNMET. fBm2D (monofractal) is still the only 2D construction with a
genuine analytic check passing.

Known limitations carried forward from 1D (not re-litigated, just
re-checked): the adaptive j1 search can lock onto a narrow-but-biased range
on one noisy realization (confirmed, same as 1D). The "positive-q/singular
branch resolves more loosely" limitation could not be cleanly re-isolated
in 2D given the closed-form question above (fBm2D is monofractal -- no
branch asymmetry to measure -- and the cascade's asymmetry is entangled
with the now-resolved-as-a-false-conjecture closed-form question, not a
clean, separate finding).
"""

import itertools

import numpy as np
import pytest

from mfleaders._dwt2 import wavelet_transform_2d, combine_orientations
from mfleaders.leaders import _neighborhood_max, compute_leaders
from mfleaders.structure import zeta_q
from mfleaders.spectrum import multifractal_spectrum, holder_width
from mfleaders.scale_range import select_j1, select_j2

from synthetic import (
    cascade2d_staircase, cascade2d_zeta_true, cascade2d_h_true,
    fbm2d_spectral,
)

Q = np.concatenate([np.arange(-5, 0), np.arange(1, 6)]).astype(float)


def _fold_children_weighted(window_max, finer_leader, dim, weight):
    """Copy of leaders._fold_children with an explicit weight override, for
    adjudicating the cone weight against alternatives. The real library
    (leaders.py) only ever uses weight=2**-dim; this exists in test code
    only, mirroring how the 1D sum-vs-max adjudication kept the "wrong"
    alternative out of the library's public surface.
    """
    parent_shape = window_max.shape
    child_shape = finer_leader.shape

    max_index = tuple(s // 2 for s in child_shape)
    valid_shape = tuple(min(p, m) for p, m in zip(parent_shape, max_index))

    stack = [window_max]
    for off in itertools.product((0, 1), repeat=dim):
        sl = tuple(slice(o, o + 2 * v, 2) for o, v in zip(off, valid_shape))
        child = finer_leader[sl]

        candidate = np.full(parent_shape, np.nan)
        crop = tuple(slice(0, v) for v in valid_shape)
        candidate[crop] = weight * child
        stack.append(candidate)

    stacked = np.stack(stack, axis=0)
    with np.errstate(invalid='ignore'):
        result = np.max(stacked, axis=0)
    return result


def _compute_leaders_with_weight(coeffs_by_scale, dim, weight):
    scales = sorted(coeffs_by_scale)
    leaders = {}
    prev = None
    for j in scales:
        window_max = _neighborhood_max(np.abs(coeffs_by_scale[j]), dim)
        if prev is None:
            leaders[j] = window_max
        else:
            leaders[j] = _fold_children_weighted(window_max, prev, dim, weight)
        prev = leaders[j]
    return leaders


def _leaders_2d(image, wt_name='db3'):
    coeffs = wavelet_transform_2d(image, wt_name=wt_name, normalization=1)
    mag = combine_orientations(coeffs)
    return compute_leaders(mag, dim=2)


def _mag_2d(image, wt_name='db3'):
    coeffs = wavelet_transform_2d(image, wt_name=wt_name, normalization=1)
    return combine_orientations(coeffs)


# ---------------------------------------------------------------------------
# Ground-truth generator self-check
# ---------------------------------------------------------------------------

def test_fbm2d_generator_matches_theoretical_structure_function_scaling():
    """Direct (non-wavelet) check: Var[F(x+r,y)-F(x,y)] ~ r^(2H)."""
    n = 512
    n_trials = 20
    rs = np.array([2, 4, 8, 16, 32, 64])

    for H in [0.5, 0.7]:
        rng = np.random.default_rng(0)
        logvars = np.zeros(len(rs))
        for _ in range(n_trials):
            field = fbm2d_spectral(n, H, rng)
            for i, r in enumerate(rs):
                d = field[:, r:] - field[:, :-r]
                logvars[i] += np.log2(np.var(d))
        logvars /= n_trials
        slope = np.polyfit(np.log2(rs), logvars, 1)[0]
        assert slope == pytest.approx(2 * H, abs=0.15)


# ---------------------------------------------------------------------------
# fBm2D: isotropic monofractal collapse (single H everywhere, both branches)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("H", [0.5, 0.7])
def test_fbm2d_zeta_is_linear_and_recovers_H(H):
    rng = np.random.default_rng(3000 + int(H * 100))
    image = fbm2d_spectral(256, H, rng)

    leaders = _leaders_2d(image)
    j2 = select_j2(leaders, min_nj=100)
    j1 = 2

    _, _, _, zeta, _ = zeta_q(leaders, Q, j1, j2)

    fit_slope = np.sum(Q * zeta) / np.sum(Q * Q)
    residual = zeta - fit_slope * Q
    assert np.max(np.abs(residual)) < 0.15 * (zeta.max() - zeta.min())

    H_est = zeta[np.argmin(np.abs(Q - 2))] / 2
    assert H_est == pytest.approx(H, abs=0.15)


@pytest.mark.parametrize("H", [0.5, 0.7])
def test_fbm2d_D_h_collapses_to_a_point(H):
    rng = np.random.default_rng(4000 + int(H * 100))
    image = fbm2d_spectral(256, H, rng)

    leaders = _leaders_2d(image)
    j2 = select_j2(leaders, min_nj=100)
    j1 = 2

    _, _, _, _, hq, Dq = multifractal_spectrum(leaders, Q, j1, j2, dim=2)

    assert (Q < 0).any() and (Q > 0).any()
    delta_h = holder_width(hq)
    assert delta_h < 0.3  # narrow: the cascade below is far wider
    assert np.nanmean(hq) == pytest.approx(H, abs=0.15)
    assert np.nanmean(Dq) == pytest.approx(2.0, abs=0.3)  # ambient dim = 2


# ---------------------------------------------------------------------------
# Cone weight: adjudicated against fBm2D (see module docstring for why the
# cascade-based adjudication originally planned here is not used)
# ---------------------------------------------------------------------------

def test_cone_weight_2_pow_neg_2_is_at_least_as_good_as_alternatives():
    """Monte Carlo H recovery at j1=1 (no cone depth at all -- where the
    weight first enters, same logic as the 1D adjudication) across
    candidate weights. Unlike the 1D sum-vs-max case, this does NOT sharply
    discriminate among weights <= 0.5 (the current-scale window usually
    already wins the max regardless), so treat this as "not worse than
    alternatives" rather than a decisive proof -- reported honestly, see
    module docstring.
    """
    H_true = 0.7
    n = 256
    n_trials = 25
    j1, j2 = 1, 5

    weights = [1.0, 0.5, 0.25, 0.125]
    results = {w: [] for w in weights}

    rng = np.random.default_rng(42)
    for _ in range(n_trials):
        image = fbm2d_spectral(n, H_true, rng)
        mag = _mag_2d(image)
        for w in weights:
            leaders = _compute_leaders_with_weight(mag, dim=2, weight=w)
            zeta = zeta_q(leaders, np.array([2.0]), j1, j2)[3]
            results[w].append(zeta[0] / 2)

    print(f"\ncone-weight adjudication (fBm2D, H={H_true}, n_trials={n_trials}):")
    biases = {}
    for w in weights:
        vals = np.array(results[w])
        bias = vals.mean() - H_true
        se = vals.std(ddof=1) / np.sqrt(n_trials)
        biases[w] = abs(bias)
        print(f"  weight={w}: bias={bias:+.4f} (se={se:.4f})")

    # the actual weight (0.25) must not be WORSE than the theoretically
    # wrong "no damping" choice (1.0)
    assert biases[0.25] <= biases[1.0] + 0.02


# ---------------------------------------------------------------------------
# 2D cascade: qualitative shape only (see module docstring -- the
# quantitative closed-form target is a confirmed false conjecture, not
# asserted here; see test_cascade2d_measure_spectrum_bridge_is_a_confirmed_
# false_conjecture below)
# ---------------------------------------------------------------------------

def test_cascade2d_qualitative_shape_only():
    """What IS supported: zeta(q) concave, h(q) monotonically decreasing
    (both branches present, correct sign), D(q) peaking near the ambient
    dimension close to q=0. Quantitative agreement with the closed form is
    NOT checked here -- see
    test_cascade2d_measure_spectrum_bridge_is_a_confirmed_false_conjecture
    (module docstring) for why that closed form is not a valid target.
    """
    weights = [0.4, 0.3, 0.2, 0.1]
    n_levels = 10
    F = cascade2d_staircase(n_levels, weights)

    leaders = _leaders_2d(F)
    max_scale = max(leaders)
    j2 = max_scale - 1
    j1 = 3

    _, _, _, _, hq, Dq = multifractal_spectrum(leaders, Q, j1, j2, dim=2)
    zeta = zeta_q(leaders, Q, j1, j2)[3]

    assert (Q < 0).any() and (Q > 0).any()

    neg = Q < 0
    slopes_neg = np.diff(zeta[neg]) / np.diff(Q[neg])
    assert np.all(np.diff(slopes_neg) < 1e-6), "zeta(q) not concave (q<0)"

    assert np.all(np.diff(hq) < 0.05), "h(q) not decreasing in q"
    assert Dq[np.argmin(np.abs(Q - 1))] < 2.0 + 1e-6


def test_cascade2d_measure_spectrum_bridge_is_a_confirmed_false_conjecture():
    """RETIRED from strict xfail (was
    test_cascade2d_quantitative_match_is_unresolved) -- this is a settled
    negative result, not an open question. NOT A PIPELINE BUG: the pipeline
    itself has now been independently corroborated (see
    tests/test_dwt2.py's test_multifracs_cone_matches_multifracs_cascade_to_
    machine_precision and test_leaders_2d_entry_point_matches_multifracs_
    cascade_with_pure_defaults) against MULTIFRACS, an external, published,
    separately-implemented toolbox, to machine precision (~1e-8/1e-9) on
    the SEPARABLE p=0.25 binomial cascade (handoff/plbmf_crosscheck/
    cascade_input.mat) -- the same field construction as this test's own
    (non-separable, 4-weight) cascade, just a different parameterization.

    What was actually wrong is the CLOSED-FORM TARGET this test compares
    against (`cascade2d_zeta_true`/`cascade2d_h_true`, tests/synthetic.py),
    not the wavelet-leader pipeline. That target computes the cascade
    MEASURE's own mass/partition-function spectrum under the "sum"
    hypothesis (local exponent = hx+hy, i.e. it implicitly assumes the HH
    subband alone determines the leader asymptotically) and bridges it to
    zeta(q) via `zeta = ambient_dim + tau_measure` -- a bridge that is
    empirically validated in 1D but was never validated in 2D, and is
    directly contradicted by measurement. The actual WRJA (2009) leader
    definition takes a max over LH/HL/HH jointly (independently confirmed
    correct against the reference, see the module docstring above) -- no
    orientation is privileged, so the "HH alone" assumption behind the
    closed form's bridge does not describe what the pipeline (correctly)
    computes.

    Decisive evidence (on the SEPARABLE p=0.25 field, where MULTIFRACS
    parity gives independent, external corroboration that the MEASURED
    value is correct, not just self-consistent): at q=+5,
        zeta_true_2d (closed form)  = 6.1385   vs   measured zeta = 1.4096
        h_true_2d (closed form)     = 0.8431   vs   measured h    = 0.1085
    MULTIFRACS's own reported values match mfleaders' measured numbers to
    ~1e-9 -- i.e. MULTIFRACS *also* fails this closed form, at the same
    tolerances, on the same field. Two independent implementations
    agreeing with each other and disagreeing with the closed form is
    exactly the signature of a wrong conjecture, not a shared pipeline bug.

    This test now PINS the confirmed divergence (inverted from "should
    eventually match" to "known and expected not to match") so that if the
    gap were to unexpectedly close in the future, that itself would be
    noteworthy (e.g. a leader-construction change) rather than silently
    accepted. `cascade2d_zeta_true`/`cascade2d_h_true`/`cascade2d_D_true`
    (tests/synthetic.py) are kept, labeled as what they are (the cascade
    MEASURE's spectrum, not a validated 2D leader-spectrum target) -- not
    deleted, since they remain a legitimate (if different) quantity.

    Gate 2 status (record honestly, do not silently upgrade): this
    establishes REFERENCE PARITY with MULTIFRACS on the 2D multifractal
    cascade -- not an independent analytic ground-truth match. The 2D
    multifractal ANALYTIC validation rung (a wavelet-leader spectrum
    matching a properly-derived, first-principles 2D closed form) remains
    UNMET. fBm2D (monofractal) is still the only 2D construction with a
    genuine analytic check (H recovery, D(h) collapse to a point) passing.
    """
    weights = [0.4, 0.3, 0.2, 0.1]
    n_levels = 10
    F = cascade2d_staircase(n_levels, weights)

    leaders = _leaders_2d(F)
    max_scale = max(leaders)
    j2 = max_scale - 1
    j1 = 3

    _, _, _, _, hq, Dq = multifractal_spectrum(leaders, Q, j1, j2, dim=2)
    zeta = zeta_q(leaders, Q, j1, j2)[3]

    zt = cascade2d_zeta_true(Q, weights)
    ht = cascade2d_h_true(Q, weights)

    # confirmed, substantial, stable divergence -- NOT a near-miss and NOT
    # the pipeline (see docstring): pin that it stays large, don't assert
    # it's small.
    assert np.max(np.abs(zeta - zt)) > 1.0, (
        "the measure-spectrum bridge's divergence from the pipeline has "
        "shrunk unexpectedly -- investigate before assuming this is good "
        "news; it may indicate a leader-construction change, not the "
        "conjecture becoming true"
    )
    assert np.max(np.abs(hq - ht)) > 0.5


# ---------------------------------------------------------------------------
# Known-limitation re-check (not re-litigated, just confirmed still true)
# ---------------------------------------------------------------------------
#
# Note on the "positive-q/singular branch resolves more loosely" limitation
# carried forward from 1D: fBm2D is a poor place to re-check it (being
# monofractal, there is no genuine branch asymmetry to measure -- an
# attempted realization-to-realization-spread check here found no reliable
# difference between q>0 and q<0, as expected). The 2D cascade IS where
# this shows up, but it is entangled with the open LH/HL question above: at
# modest |q| both branches are off by a similar factor there, and only
# diverge somewhat more at extreme |q| -- printed in
# test_cascade2d_qualitative_shape_only's output, not asserted as a clean,
# separate finding the way it could be in 1D.


def test_select_j1_can_still_misfire_on_a_single_2d_realization():
    """Re-check (not re-litigate) the known adaptive-j1 limitation in 2D:
    it can still lock onto a narrow-but-biased range on one noisy
    realization -- the analytic tests above rely on the simpler fixed
    floor instead, exactly as in 1D. This just confirms select_j1 runs and
    lands somewhere sane on 2D leaders.

    n=256 at this seed gives j2=3, which is too narrow for select_j1's
    default min_range=3/min_j1=2 to produce ANY valid range (narrowest
    possible j1 would be 1, below the floor) -- select_j1 now correctly
    raises ValueError for that input (see scale_range.py; previously it
    silently returned the floor-violating j1=1, a real bug found via this
    exact test). n=512 gives more scales (j2=4) and is the genuine "runs
    and lands somewhere sane" case this test is meant to check.
    """
    rng = np.random.default_rng(4050)
    image = fbm2d_spectral(512, 0.5, rng)
    leaders = _leaders_2d(image)
    j2 = select_j2(leaders, min_nj=100)

    j1, trace = select_j1(leaders, j2)
    assert j1 >= 2
    assert len(trace) >= 1

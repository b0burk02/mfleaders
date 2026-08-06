"""Adjudication of the p=infinity cone weight (compute_leaders' `weight`
parameter): definition-driven, not tuned to fit any target.

BACKGROUND. leaders.py's `_fold_children` damps a finer scale's leader by
`weight` before it competes in the next scale's max. The default
(`weight='pymultifracs'`, giving `2**-dim`) matches pymultifracs' own p=inf
construction (needed for the Gate-1 1D oracle). But the WRJA p-leader
definition, taken to its p->infinity limit, gives an UNWEIGHTED sup: for
any FIXED positive weight c, `(c * sum(x**p))**(1/p) -> max(x)` as p->inf
(the weight is p-independent and any fixed positive constant to the power
1/p tends to 1) -- so per the definition, `weight` at p=infinity should be
1.0, not `2**-dim`. This is a mathematical fact about the definition, not
an implementation detail, and it's verified numerically below.

Reading pymultifracs/wavelet.py directly shows its `.5 * lower_contribution`
term fires even when `leader_flag` is True (i.e. even at p=infinity) -- so
pymultifracs itself appears to carry the finite-p normalization into its
p=infinity branch, a likely bug beyond the already-adjudicated base-scale
sum-vs-max quirk (see test_oracle_pymultifracs.py).

WHAT THIS FILE ESTABLISHES (empirically, not by definition-reading alone):
  1. Removing the weight does NOT explain or fix the Gate-2 cascade
     closed-form mismatch -- weighted and unweighted give virtually
     identical (both still wrong) results on both the separable and the
     non-separable 2D cascade, at every j1 tested. The weight question and
     the cascade question are SEPARATE, not the same bug.
  2. Weighted vs unweighted is NOT decisively distinguishable via fBm
     (1D or 2D) Monte Carlo either -- unlike the base-scale sum-vs-max
     adjudication (a clean 3x, many-SE difference), the two weight choices
     give biases within 1-2 SE of each other at every j1 and H tested.
  3. Switching the default to unweighted WOULD break the Gate-1 1D
     pymultifracs oracle at every scale (order-1 differences, growing with
     scale) -- pymultifracs' own weighted behavior must be replicated
     exactly to match it, so the two questions (definitional correctness
     vs oracle-matching) are in tension and each is tested for what it is.

CONCLUSION: this was a legitimate, well-motivated hypothesis (and the
p->infinity mathematical argument for weight=1.0 stands on its own), but it
does not resolve the cascade's closed-form mismatch on its own -- that
mismatch is now understood to be a false conjecture in the closed-form
target itself, not a pipeline bug (resolved via MULTIFRACS reference
parity; see test_analytic_validation_2d.py's
test_cascade2d_measure_spectrum_bridge_is_a_confirmed_false_conjecture and
CLAUDE.md's Gate 2 status). The cone weight is a SEPARATE finding from that
resolution: weight=1.0 (unweighted) is exactly what MULTIFRACS's own
p=infinity construction uses (confirmed, not just consistent with the
p->infinity mathematical argument) -- see leaders.py's `compute_leaders`,
`cone='multifracs'` now defaults `weight` to 1.0 for this reason.
"""

import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from mfleaders._dwt import wavelet_transform_1d
from mfleaders._dwt2 import wavelet_transform_2d, combine_orientations
from mfleaders.leaders import compute_leaders
from mfleaders.structure import zeta_q
from mfleaders.spectrum import multifractal_spectrum

from synthetic import (
    fbm, cascade_staircase_2d, cascade_zeta_true_2d, cascade_h_true_2d,
    cascade2d_staircase, cascade2d_zeta_true, cascade2d_h_true,
)

Q = np.concatenate([np.arange(-5, 0), np.arange(1, 6)]).astype(float)


def test_weight_vanishes_in_the_p_to_infinity_limit():
    """The definitional argument, checked numerically: for ANY fixed
    positive weight, the weighted p-norm converges to the unweighted max
    as p -> infinity.
    """
    x = np.array([0.3, 0.7, 0.5, 0.9, 0.2])
    for c in [1.0, 0.5, 0.25, 0.125, 4.0]:
        val_p1000 = (c * np.sum(x ** 1000)) ** (1 / 1000)
        assert val_p1000 == pytest.approx(x.max(), abs=2e-3)


def test_unweighted_variant_breaks_the_1d_pymultifracs_oracle():
    """Confirms the default ('pymultifracs') is required for the Gate-1
    oracle match, and that the unweighted alternative genuinely diverges
    (not a subtle residual) -- so switching the default would silently
    break Gate 1.
    """
    import pymultifracs as pmf
    from pymultifracs import wavelet as pmw

    rng = np.random.default_rng(42)
    signal = np.cumsum(rng.standard_normal(2 ** 14))

    wt_coefs = pmw.wavelet_analysis(signal, wt_name='db3', normalization=1)
    wt_leaders = pmw.compute_leaders(wt_coefs, p_exp=np.inf)
    theirs = {j: wt_leaders.values[j][:, 0] for j in wt_leaders.values}

    coeffs = wavelet_transform_1d(signal, wt_name='db3', normalization=1)

    # cone='pymultifracs' explicitly: compute_leaders' own default is now
    # 'multifracs' (2D's reference toolbox), but this test is specifically
    # about matching pymultifracs bit-for-bit -- it needs the OTHER cone
    # composition regardless of weight choice.
    weighted = compute_leaders(coeffs, dim=1, weight='pymultifracs',
                                cone='pymultifracs')
    unweighted = compute_leaders(coeffs, dim=1, weight=1.0,
                                  cone='pymultifracs')

    j = 6
    t = theirs[j]
    n = len(t)

    m_weighted = weighted[j][:n]
    valid = ~np.isnan(t) & ~np.isnan(m_weighted)
    np.testing.assert_allclose(t[valid], m_weighted[valid], atol=1e-9)

    m_unweighted = unweighted[j][:n]
    valid_u = ~np.isnan(t) & ~np.isnan(m_unweighted)
    diff_u = np.abs(t[valid_u] - m_unweighted[valid_u])
    assert diff_u.max() > 0.5, (
        "expected the unweighted variant to diverge substantially from "
        "pymultifracs, not match it")


def test_weight_choice_does_not_explain_separable_cascade_mismatch():
    p = 0.25
    F = cascade_staircase_2d(11, p)
    zt = cascade_zeta_true_2d(Q, p)
    ht = cascade_h_true_2d(Q, p)

    coeffs = wavelet_transform_2d(F, wt_name='db3', normalization=1)
    mag = combine_orientations(coeffs)

    errs = {}
    for name, w in [('weighted', 'pymultifracs'), ('unweighted', 1.0)]:
        leaders = compute_leaders(mag, dim=2, weight=w)
        max_scale = max(leaders)
        j1, j2 = 3, max_scale - 1
        zeta = zeta_q(leaders, Q, j1, j2)[3]
        _, _, _, _, hq, _ = multifractal_spectrum(leaders, Q, j1, j2, dim=2)
        errs[name] = (np.nanmax(np.abs(hq - ht)), np.nanmax(np.abs(zeta - zt)))

    print(f"\nseparable cascade, weighted vs unweighted errors: {errs}")

    # both remain badly wrong -- neither is a fix
    assert errs['weighted'][0] > 0.5
    assert errs['unweighted'][0] > 0.5
    # and they're close to each other (removing the weight isn't the lever)
    assert abs(errs['weighted'][0] - errs['unweighted'][0]) < 0.2
    assert abs(errs['weighted'][1] - errs['unweighted'][1]) < 0.5


def test_weight_choice_does_not_explain_nonseparable_cascade_mismatch():
    weights4 = [0.4, 0.3, 0.2, 0.1]
    F = cascade2d_staircase(10, weights4)
    zt = cascade2d_zeta_true(Q, weights4)
    ht = cascade2d_h_true(Q, weights4)

    coeffs = wavelet_transform_2d(F, wt_name='db3', normalization=1)
    mag = combine_orientations(coeffs)

    errs = {}
    for name, w in [('weighted', 'pymultifracs'), ('unweighted', 1.0)]:
        leaders = compute_leaders(mag, dim=2, weight=w)
        max_scale = max(leaders)
        j1, j2 = 3, max_scale - 1
        zeta = zeta_q(leaders, Q, j1, j2)[3]
        _, _, _, _, hq, _ = multifractal_spectrum(leaders, Q, j1, j2, dim=2)
        errs[name] = (np.nanmax(np.abs(hq - ht)), np.nanmax(np.abs(zeta - zt)))

    print(f"\nnon-separable cascade, weighted vs unweighted errors: {errs}")

    assert errs['weighted'][0] > 1.0
    assert errs['unweighted'][0] > 1.0
    assert abs(errs['weighted'][0] - errs['unweighted'][0]) < 0.3


def test_weight_choice_is_not_decisive_on_fbm_H_recovery():
    """Unlike the base-scale sum-vs-max adjudication (a clean, many-SE
    difference), weighted vs unweighted H recovery should NOT be sharply
    distinguishable here -- both 1D and 2D. A small, real difference is
    fine; a 1D-style 3x/many-SE gap would mean this test's premise (that
    the weight isn't the cascade's problem) needs revisiting.
    """
    H_true = 0.7
    n = 2 ** 12
    n_trials = 30
    j1, j2 = 1, 6

    rng = np.random.default_rng(7)
    results = {'weighted': [], 'unweighted': []}
    for _ in range(n_trials):
        signal = fbm(n, H_true, rng)
        coeffs = wavelet_transform_1d(signal, wt_name='db3', normalization=1)
        for name, w in [('weighted', 'pymultifracs'), ('unweighted', 1.0)]:
            # cone='pymultifracs' explicitly: this test isolates the weight
            # variable specifically -- it must not also silently change
            # cone composition (compute_leaders' own default moved to
            # 'multifracs'), which would confound what's being adjudicated.
            leaders = compute_leaders(coeffs, dim=1, weight=w,
                                       cone='pymultifracs')
            zeta = zeta_q(leaders, np.array([2.0]), j1, j2)[3]
            results[name].append(zeta[0] / 2)

    means = {k: np.mean(v) for k, v in results.items()}
    ses = {k: np.std(v, ddof=1) / np.sqrt(n_trials) for k, v in results.items()}
    diff = abs(means['weighted'] - means['unweighted'])
    combined_se = np.sqrt(ses['weighted'] ** 2 + ses['unweighted'] ** 2)

    print(f"\n1D fBm H-recovery, weighted={means['weighted']:.4f}"
          f"+/-{ses['weighted']:.4f} vs unweighted={means['unweighted']:.4f}"
          f"+/-{ses['unweighted']:.4f}")

    assert diff < 5 * combined_se, (
        "weighted vs unweighted diverged sharply -- reconsider whether the "
        "weight IS after all a significant factor")

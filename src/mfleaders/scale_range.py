"""Principled selection of the fit range's lower bound, j1.

This does NOT compare against any other implementation or known answer --
it only uses a single fit's own residual scatter, which is exactly what's
available in a real application where the true exponent is unknown. That
makes it a defensible rule for a methods section (unlike "scan until this
matches a reference implementation").

Rule: fix j2 and start from the narrowest usable range [j2 - min_range + 1,
j2]. Widen the range one scale at a time (decreasing j1). Accept the wider
range as long as its slope estimate agrees with the previous (narrower)
range's estimate to within `n_se` standard errors of either fit. Stop -- and
report the last-accepted (widest still-consistent) j1 -- the first time
widening shifts the estimate beyond that band, or when j1 would drop below
`min_j1`. A real bias entering at small j (e.g. a base-scale construction
artifact, or generic discretization/finite-resolution effects) shows up
exactly as a shift the stability check can catch -- but with only one
realization, a small bias can still hide inside fit noise. `min_j1` is a
floor of last resort: at j1=1 the leader has no recursive cone depth at all
(it is just the current scale's own neighborhood max, with none of the
finer scales folded in yet), so leaders there behave like raw coefficients
rather than the cross-scale construction the whole formalism relies on.
Requiring at least one octave of cone depth (min_j1=2) is a standard,
generic convention, not tuned to any specific reference or dataset.

Known residual at the default floor (min_j1=2). Monte Carlo on fBm(H=0.7,
n=2**13, db3, j2=8, 200 trials): H recovered from zeta(2)/2 has bias -0.0094
(SE 0.0018, 5.1 SE from zero) at j1=2. This is NOT leftover base-scale
contamination -- that artifact (see leaders.py, test_analytic_validation.py)
decays sharply with j1 and is gone by j1=3 to float precision, whereas this
residual is essentially the same size at j1=2 (-0.0094), j1=3 (-0.0062,
2.5 SE), j1=4 (-0.0058, 1.8 SE) and j1=5 (-0.0092, 1.8 SE) -- flat rather
than decaying. The likely cause is generic: log2(mean(L^2)) is a downward-
biased (Jensen's inequality, log of a finite-sample mean) estimator of
log2(E[L^2]) at every scale, not specific to j1=2 or to the base case. Track
as a small (~1% of H at these settings), separate, known systematic -- do
not conflate it with the base-scale question the j1 rule above is guarding
against.
"""

import numpy as np

from .regression import linear_regression_se
from .structure import structure_functions


def select_j2(leaders_by_scale, min_nj=30, q_ref=2.0):
    """Largest scale with at least `min_nj` valid (non-NaN) leaders.

    Coarse scales are built from few dyadic positions; a fit range that
    includes scales with only a handful of leaders is dominated by their
    noise. Requiring a minimum count is a standard, generic reliability
    floor -- independent of wavelet, signal, or any reference implementation.
    `min_nj=30` is a conventional, unremarkable choice (not tuned per case).
    """
    scales, _, nj = structure_functions(leaders_by_scale, [q_ref])
    reliable = scales[nj >= min_nj]
    if len(reliable) == 0:
        raise ValueError(f"no scale has >= {min_nj} valid leaders")
    return int(reliable.max())


def select_j1(leaders_by_scale, j2, q_ref=2.0, weighted=None, n_se=1.0,
              min_range=3, min_j1=2):
    """Smallest j1 (widest range) whose zeta(q_ref) estimate is still
    consistent, within `n_se` standard errors, with EVERY narrower range
    already accepted -- anchored to the narrowest range, not just chained
    pairwise, so a single noisy-but-lucky adjacent pair can't drag the
    selection into a biased regime. Floored at `min_j1`.

    Returns
    -------
    j1 : int
    trace : list[dict]
        One entry per candidate range tried, in narrowing order, with keys
        j1, slope, se. Useful for diagnostics/plots.
    """
    scales, log2_s, nj = structure_functions(leaders_by_scale, [q_ref])
    y = log2_s[:, 0]

    narrowest_j1 = j2 - min_range + 1
    if narrowest_j1 < min_j1:
        # No range satisfies both `min_range` and `min_j1` for this `j2` --
        # `range(narrowest_j1, min_j1 - 1, -1)` would be empty, and falling
        # through used to silently return the pre-loop `accepted_j1 =
        # narrowest_j1`, BELOW the documented floor (e.g. j1=1 on a short
        # j2=3 2D map). That is a real bug (found via
        # test_select_j1_can_still_misfire_on_a_single_2d_realization), not
        # a tuning choice -- raise instead of returning an invalid j1.
        raise ValueError(
            f"no scaling range satisfies both min_range={min_range} and "
            f"min_j1={min_j1} for j2={j2} (narrowest possible j1 would be "
            f"{narrowest_j1}); use a larger j2 or relax min_range/min_j1")
    candidates_desc = list(range(narrowest_j1, min_j1 - 1, -1))

    w = nj.astype(float) if weighted == 'Nj' else None

    trace = []
    accepted_j1 = narrowest_j1
    accepted_slopes = []
    accepted_ses = []

    for j1 in candidates_desc:
        mask = (scales >= j1) & (scales <= j2)
        if mask.sum() < 3:
            break

        wj = w[mask] if w is not None else None
        slope, intercept, se = linear_regression_se(scales[mask], y[mask], wj)

        if np.isnan(slope):
            break

        if accepted_slopes:
            consistent_with_all = all(
                abs(slope - s) <= n_se * max(se, s_se)
                for s, s_se in zip(accepted_slopes, accepted_ses)
            )
            if not consistent_with_all:
                break

        trace.append({'j1': j1, 'slope': slope, 'se': se})
        accepted_j1 = j1
        accepted_slopes.append(slope)
        accepted_ses.append(se)

    # Defensive: every candidate in `candidates_desc` is >= min_j1 by
    # construction once the guard above passes, so this should already
    # hold -- clamped explicitly anyway so the floor is never silently
    # violated by some path this reasoning missed.
    return max(accepted_j1, min_j1), trace

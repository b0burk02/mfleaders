"""Dimension-generic wavelet-leader construction.

One pair of primitives (`_neighborhood_max`, `_fold_children`) builds leaders
at every dimension: the neighborhood is a ``3**dim`` block (3 in 1D, 3x3 in
2D, 3x3x3 in 3D), and the cone-recursive fold combines a scale with the
scales below it. There is no dimension-specific fork: `dim` only changes how
many neighbor offsets and child offsets get enumerated.

Cone composition -- TWO DIFFERENT WAYS TO COMPOSE THE SAME TWO PRIMITIVES.
`compute_leaders`'s `cone` parameter selects which reference toolbox's
recursion order is reproduced (see there for the full explanation):
``'multifracs'`` (the MULTIFRACS MATLAB toolbox's construction, DEFAULT) or
``'pymultifracs'`` (the pymultifracs Python package's construction, needed
for the Gate-1 1D oracle). Confirmed by exact index-set tracing (not just
value comparison) that the two conventions genuinely range over different
coefficient sets at scale j>=2 -- mfleaders' pymultifracs-style set is a
strict subset of the multifracs-style set, and the gap widens geometrically
with recursion depth. This is a real, load-bearing difference, not a
relabeling: switching a 2D cascade cross-check from one to the other closed
a persistent zeta(q) residual (previously ~0.1-0.5 across q, several ledger
items already ruled out) to machine precision (~1e-9 or better), across every
q and scale tested -- see `tests/test_dwt2.py`'s
`test_multifracs_cone_matches_multifracs_cascade_to_machine_precision`.

Cone weight -- UNDER ACTIVE INVESTIGATION, do not trust the default blindly.

The general p-leader (finite p) is defined (Wendt, Roux, Jaffard & Abry
2009) as

    L^(p)(j,k) = [ sum_{lambda' subset 3*lambda_{j,k}} 2**((j'-j)*dim) * |c_lambda'|**p ]**(1/p)

with the sum over all finer-or-equal scales j'<=j in the spatial
neighborhood. As p -> infinity, THIS EXPRESSION CONVERGES TO AN UNWEIGHTED
SUP, regardless of the weight: (c * sum(x_i**p))**(1/p) -> max(x_i) as
p->inf for any FIXED positive c (verified numerically -- the weight is a
p-independent constant, and any fixed positive constant to the power 1/p
tends to 1). So the plain (p=infinity) wavelet leader, per this definition,
is `sup |c_lambda'|` -- unweighted. mfleaders only implements p=infinity
(rule 3), so `weight` below should, per the definition, be 1.0 (no
damping) -- NOT 2**-dim.

`compute_leaders`'s `weight` parameter defaults to `2**-dim` for backward
compatibility with the already-certified 1D pymultifracs oracle (Gate 1),
whose own p=infinity leader construction ALSO applies this weight (verified
by direct source reading of pymultifracs/wavelet.py: the `.5 *
lower_contribution` term fires even when `leader_flag` is True, i.e. even
at p=infinity) -- so pymultifracs itself appears to carry the finite-p
normalization into its p=infinity branch, which the p->infinity limit
argument above says shouldn't happen. Whether pymultifracs' 1D oracle match
depended on replicating that specific (probably erroneous) behavior, and
whether the unweighted variant (`weight=1.0`) is actually correct, is
exactly what's being adjudicated -- see the report and
tests/test_cone_weight_adjudication.py. Do not change the default here
until that adjudication is resolved; both variants must stay runnable and
comparable via this one parameter (not a forked/duplicated implementation).

Base case. At the finest scale there is no finer cone to fold in, so the
leader there is always a plain neighborhood max with no damping, regardless
of `weight` (there is nothing to damp yet).

Finite-p (p-leaders). Reproduces MULTIFRACS's ``DxPx2d.m`` finite-p
construction exactly (``compute_leader_sans_voisin`` / ``compute_leader_
from_neigbourhood``, both literally SAME function for p=inf and finite p in
that source -- only the final reduction (max vs sum) differs), traced line
by line rather than re-derived from the textbook WRJA formula independently
(this project's own standard: match the reference toolbox's actual
algorithm, not a recalled formula that might use a different, also-valid
convention). Concretely, for finite p:

  own(j)   = 2**(dim*j) * |coefficient(j)|**p      (NOT |coefficient(j)|,
                                                      and NOT weighted --
                                                      the weight parameter
                                                      only applies to p=inf)
  fold     = SUM over (own, weight=1.0 * the 2**dim children), instead of
             MAX -- reusing `_fold_children` with `op=np.sum` and a fixed
             weight of 1.0 (DxPx2d.m's finite-p fold applies NO extra
             per-fold damping; all the scale-dependence lives in `own`'s
             2**(dim*j) factor, applied once, at birth)
  neighbor = SUM over the padded 3**dim neighborhood, instead of MAX --
             reusing `_neighborhood_max` with `op=np.sum`
  finish   = AFTER the full scale loop: leader(j) = (2**(-dim*j) *
             accumulated(j)) ** (1/p) -- DxPx2d.m's separate "p-Leaders
             finish off" loop, applied once per scale, not per fold step.

Orientation combination for finite p is proven equivalent to MULTIFRACS's
own per-orientation-then-sum-at-the-end structure by linearity: unlike
p=inf's max (which composes freely across ANY grouping of neighborhood/
cone/orientation because max is idempotent), summing is ALSO fully
associative/commutative/distributive across those same axes, and every
step in the finite-p pipeline up to the single final `**(1/p)` root is a
pure sum -- so pre-combining orientations via `sum_o |coef_o|**p` BEFORE
this recursion (done in `_dwt2.py`'s `combine_orientations_p`, not here --
this module stays orientation-agnostic per rule 2) gives bit-identical
results to combining them only at the end, as long as the abs/power
transform is applied to each raw orientation coefficient exactly once and
the root is taken exactly once, at the very end. See
`compute_leaders`'s ``pre_combined_power`` parameter.

p-validity. MULTIFRACS does not hard-reject an invalid p: `NLcorrPL2d.m`
computes `zp`, the regression slope of `log2(mean(|coefficient(j)|**p))`
against j (an estimate of the coefficients' own p-th-moment scaling
exponent), and treats `zp <= 0` as "the correction term for p-leaders is
undefined... final estimates can be biased... a smaller p (or larger
gamint) should be selected" -- a WARNING, not a refusal, with fractional
integration (gamint) as its suggested remedy. `gamint` is a coefficient-
level preprocessing step applied before leader construction (with its own
nonlinear post-hoc correction, `NLcorrPL2d.m`) -- implementing it is out of
this extension's stated scope (leader construction only). `_p_validity_zp`
below mirrors MULTIFRACS's OWN check (same regression-slope-of-log2-mean
quantity) and emits the same kind of warning, minus the gamint remedy this
package does not implement.
"""

import itertools
import warnings

import numpy as np

from .regression import linear_regression

_UNSET = object()


def _neighborhood_max(coefs, dim, boundary='discard', op=np.max):
    """Max over a (3,)*dim neighborhood.

    `op` (default ``np.max``, matching every existing caller): the
    reduction applied to the stacked neighborhood candidates. Passing
    ``np.sum`` (used only by `compute_leaders`'s finite-p path) reuses the
    exact same candidate-stacking geometry with MULTIFRACS's own finite-p
    reduction -- `DxPx2d.m`'s ``compute_leader_from_neigbourhood`` is
    literally one function, shared between p=inf and finite p, differing
    only in this reduction.

    boundary : 'discard' | 'pad'
        'discard' (default): border positions -> NaN, unconditionally,
        regardless of whether the underlying coefficients there were finite.
        A single missing (NaN) neighbor invalidates the whole position -- a
        leader is only well-defined where its full neighborhood is
        uncorrupted by the signal's boundary, so NaN propagates via plain
        ``max`` rather than being skipped via ``nanmax``. This matters
        beyond the immediate border: combined with the recursive fold below,
        it makes the invalid region grow (by one element per axis) at every
        coarser scale, rather than staying fixed at one element in from each
        edge. This is the convention pymultifracs' 1D leader construction
        also uses (verified by direct source read, Gate 1) -- do not change
        this default, 1D depends on it exactly.

        'pad': MULTIFRACS's convention (``compute_leader_from_neigbourhood``
        in ``DxPx2d.m``): pad the array with a phantom ring of ZERO (not
        NaN) and compute a value at every position, including the true
        outer edge. A position only comes out invalid if a REAL (non-pad)
        neighbor was already NaN -- the zero-pad itself never manufactures
        invalidity, since it can never win a max against a finite magnitude.
        Used for the 2D leader pipeline to match MULTIFRACS; not used for
        1D (would diverge from the Gate-1 pymultifracs oracle).
    """
    shape = coefs.shape

    if boundary == 'pad':
        padded = np.pad(coefs, 1, mode='constant', constant_values=0.0)
        offsets = itertools.product((-1, 0, 1), repeat=dim)
        stack = []
        for off in offsets:
            sl = tuple(slice(1 + o, 1 + o + s) for o, s in zip(off, shape))
            stack.append(padded[sl])
        stacked = np.stack(stack, axis=0)
        with np.errstate(invalid='ignore'):
            return op(stacked, axis=0)

    out = np.full(shape, np.nan)

    interior = tuple(slice(1, s - 1) for s in shape)
    if any(s <= 2 for s in shape):
        return out

    offsets = itertools.product((-1, 0, 1), repeat=dim)
    stack = []
    for off in offsets:
        sl = tuple(slice(1 + o, s - 1 + o) for o, s in zip(off, shape))
        stack.append(coefs[sl])

    stacked = np.stack(stack, axis=0)
    with np.errstate(invalid='ignore'):
        out[interior] = op(stacked, axis=0)

    return out


def _fold_children(window_max, finer_leader, dim, weight, op=np.max):
    """Combine this scale's neighborhood max with the finer scale's leaders.

    ``finer_leader`` has roughly twice ``window_max``'s extent per axis
    (dyadic subsampling): parent index k's children live at 2k, 2k+1 along
    each axis. Each child contributes ``weight * finer_leader[child]``; the
    result is the elementwise reduction (``op``, default ``np.max`` --
    matching every existing caller byte-for-byte) of ``window_max`` and all
    ``2**dim`` weighted children. As in `_neighborhood_max`, a single NaN
    among these ``1 + 2**dim`` candidates invalidates the position (plain
    ``max``/``sum``, not ``nanmax``/``nansum``) -- a leader needs its whole
    cone, not just what's available. ``op=np.sum`` (finite-p path) reuses
    this exact candidate geometry with MULTIFRACS's own finite-p reduction
    (`DxPx2d.m`'s ``compute_leader_sans_voisin`` -- one function, shared
    between p=inf and finite p, differing only in this reduction).
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
        result = op(stacked, axis=0)

    return result


def _p_validity_zp(base_by_scale, scales, p):
    """Mirror MULTIFRACS's own p-leader validity check (`NLcorrPL2d.m`):
    the regression slope of log2(mean(|coefficient(j)|**p)) against j (an
    estimate of the coefficients' own p-th-moment scaling exponent, called
    `zp` there). MULTIFRACS treats `zp <= 0` as "the correction term for
    p-leaders is undefined... final estimates can be biased" and warns
    rather than refusing to compute -- same here. `base_by_scale[j]` is
    already ``|coefficient(j)|**p`` (or, for the orientation-combined 2D
    case, ``sum_o |coef_o(j)|**p``) -- i.e. exactly the quantity this check
    needs, already computed by `compute_leaders` for the leader
    construction itself; passed in rather than recomputed.
    """
    zpj = np.full(len(scales), np.nan)
    for i, j in enumerate(scales):
        vals = base_by_scale[j]
        finite = vals[np.isfinite(vals)]
        if finite.size == 0:
            continue
        with np.errstate(divide='ignore'):
            mean_val = np.mean(finite)
            zpj[i] = np.log2(mean_val) if mean_val > 0 else -np.inf

    slope, _ = linear_regression(np.array(scales, dtype=float), zpj)
    if np.isnan(slope) or slope <= 0:
        warnings.warn(
            f"p-leader validity check failed for p={p} (mirroring "
            f"MULTIFRACS's NLcorrPL2d.m): the coefficients' own p-th-"
            f"moment scaling exponent eta(p) -- the regression slope of "
            f"log2(mean(|coefficient(j)|**p)) vs scale j -- is "
            f"{slope:.4g}, not > 0. MULTIFRACS treats this as 'p-leaders "
            f"undefined, estimates can be biased' and recommends a "
            f"smaller p (or fractional integration via a larger gamint, "
            f"which this extension does not implement -- out of scope, "
            f"see compute_leaders' docstring). Proceeding anyway; treat "
            f"the resulting zeta(q)/D(h) with caution.",
            RuntimeWarning, stacklevel=3,
        )
    return slope


def compute_leaders(coeffs_by_scale, dim=1, weight=_UNSET,
                    boundary=_UNSET, cone='multifracs', p=np.inf,
                    pre_combined_power=False):
    """Build wavelet leaders (p=infinity, the default) or finite-p leaders
    from per-scale coefficient arrays.

    Calling this with NO keyword arguments beyond ``dim`` reproduces
    MULTIFRACS's own leader construction exactly (audited requirement: the
    default configuration as a whole must reproduce one reference toolbox,
    not an untested hybrid of both). Concretely, ``weight`` and ``boundary``
    default to whichever value matches the ``cone`` in effect -- NOT to a
    fixed literal -- so leaving them unspecified while requesting
    ``cone='pymultifracs'`` gives the OTHER reference's values instead. This
    is why their defaults are an internal sentinel rather than a literal:
    the "right" default genuinely depends on which toolbox `cone` is
    reproducing, and previously (before this was fixed) ``weight`` and
    ``boundary`` had FIXED literal defaults tied only to the 1D/pymultifracs
    convention regardless of `cone`, so ``compute_leaders(coeffs, dim=2)``
    silently ran cone='multifracs' + weight=2**-dim + boundary='discard' --
    a hybrid matching NEITHER reference toolbox. That combination is no
    longer reachable by omission; pass both explicitly if you genuinely
    want it.

    NOTE ON THE DWT ITSELF: this function only builds leaders FROM already-
    computed coefficients. Full end-to-end MULTIFRACS reproduction on an
    image also needs `_dwt2.wavelet_transform_2d`'s own ``border`` argument
    set to ``'multifracs'`` (its OWN default stays ``'mfleaders'`` --
    changing that default is out of scope here). Use `_dwt2.leaders_2d` for
    a single call that reproduces MULTIFRACS end to end, DWT included, by
    default.

    Holding `wavelet_transform_2d`'s OWN default fixed protects the DWT
    LAYER ONLY, not fBm2D's certification config as a whole: fBm2D's own
    tests (`tests/test_analytic_validation_2d.py`) call
    ``compute_leaders(mag, dim=2)`` with no extra keywords, so THIS
    function's ``weight``/``boundary`` default change (above) moved fBm2D's
    leader-construction config too -- from `weight=2**-dim, boundary=
    'discard'` (original certification) to `weight=1.0, boundary='pad'`
    (current, via `cone='multifracs'`'s resolution). fBm2D was re-run and
    re-confirmed passing under the new config on 2026-08-06 (H recovery
    within ~0.02-0.04 of true H, D(h) width ~0.02-0.03, both at H=0.5 and
    H=0.7) -- but that is a re-verification, not something the DWT-layer
    decoupling above provided for free.

    Parameters
    ----------
    coeffs_by_scale : dict[int, ndarray]
        Maps scale (1 = finest) to the DWT detail-coefficient array at that
        scale, shape ``(n_1, ..., n_dim)``. Scales must be contiguous
        starting at 1.
    dim : int
        Spatial dimension: 1, 2, or 3 (higher falls out of the same
        primitive but is untested).
    weight : float | 'pymultifracs' | None
        Per-octave damping applied to a finer scale's leader before it
        competes in the next scale's max (see module docstring -- this is
        UNDER ACTIVE INVESTIGATION, not a settled default).
        Left unspecified (default): resolves to ``1.0`` (unweighted, no
        damping) when ``cone='multifracs'`` -- matching MULTIFRACS's own
        p=infinity construction, which applies NO weight at all -- or to
        ``'pymultifracs'`` (see below) when ``cone='pymultifracs'``.
        ``'pymultifracs'`` reproduces ``2**-dim``, matching pymultifracs'
        own p=infinity construction and the Gate-1 1D oracle. ``1.0`` or
        ``None`` gives the UNWEIGHTED sup that the p->infinity limit of the
        WRJA p-leader definition implies. Any other float is accepted for
        direct comparison/adjudication.
    boundary : 'discard' | 'pad'
        Passed to `_neighborhood_max` (see there). Left unspecified
        (default): resolves to ``'pad'`` when ``cone='multifracs'``
        (matching MULTIFRACS's ``compute_leader_from_neigbourhood``) or to
        ``'discard'`` when ``cone='pymultifracs'`` (matching pymultifracs/
        Gate 1, 1D).
    cone : 'multifracs' | 'pymultifracs'
        Which reference toolbox's cone-composition ORDER this reproduces --
        named by which reference each reproduces, not by dimension: both
        conventions run at any `dim`, they just happen to have been
        adjudicated using a 1D oracle (pymultifracs) and a 2D one
        (MULTIFRACS) respectively.

        ``'multifracs'`` (DEFAULT): reproduces the MULTIFRACS MATLAB
        toolbox's construction (``DxPx2d.m``'s ``compute_leader_sans_voisin``
        + ``compute_leader_from_neigbourhood``) and is recommended for all
        image work. At every scale, a "cone-only" quantity is carried
        forward with NO spatial neighborhood applied in between (only the
        direct 2**dim quadrant children, via `_fold_children` on the RAW
        ``|coefficient|``, not on a neighborhood-maxed value); the 3**dim
        spatial neighborhood (`_neighborhood_max`) is applied exactly ONCE,
        at the target scale, to that cone-only quantity. This means the set
        of finer-scale coefficients contributing to a given leader is
        LARGER than under ``'pymultifracs'`` (confirmed by exact index-set
        tracing, growing geometrically with scale) -- the neighborhood
        widens the effective footprint at every one of the scales it would
        otherwise be applied at, not just the last one.

        ``'pymultifracs'``: reproduces the 1D pymultifracs package's
        construction (verified by direct source read of its
        ``wavelet.py::compute_leaders``, the ``scale>1`` branch: ``max(this
        scale's fresh 3-window, weight * direct children of the PREVIOUS
        FULL LEADER)``) and exists mainly for the 1D regression test (Gate
        1's pymultifracs oracle match depends on this exact composition,
        not the other one -- do not switch it without re-deriving Gate 1).
        Here, `_neighborhood_max` runs fresh at EVERY scale, and
        `_fold_children` folds from the previous scale's already-
        neighborhood-maxed leader, not from a raw cone-only quantity.

        Confirmed (2D cascade cross-check, see module docstring): switching
        from ``'pymultifracs'`` to ``'multifracs'`` collapsed a persistent
        zeta(q) residual against the MULTIFRACS reference from ~0.1-0.5 to
        machine precision, across the full q range tested and both branches.
    p : float
        Default ``np.inf``: plain wavelet leaders, EXACTLY today's
        behavior (this code path is untouched by the finite-p addition --
        same lines, same operations, byte-identical output). Any finite
        p > 0: finite-p leaders (see module docstring for the exact
        construction, traced from MULTIFRACS's ``DxPx2d.m``). Only
        implemented for ``cone='multifracs'`` (raises ``NotImplementedError``
        for ``cone='pymultifracs'`` -- out of scope: pymultifracs' own
        finite-p support, if any, has not been read or validated here).
        p <= 0 raises ``ValueError`` (not meaningful for any p-leader).
    pre_combined_power : bool
        Default ``False``: ``coeffs_by_scale`` holds RAW coefficients (this
        function applies ``|.|**p`` itself), matching the self-contained
        contract p=infinity already has. ``True``: ``coeffs_by_scale``
        ALREADY holds ``|coefficient|**p`` (or, for multi-orientation 2D
        data, ``sum_orientations |coef_orientation|**p`` -- see
        `_dwt2.combine_orientations_p`) -- this function only applies the
        ``2**(dim*j)`` scale weight and the final root, without touching
        the input again. Needed because orientation-combination for
        finite p must happen INSIDE the ``|.|**p`` transform (summing raw
        coefficients across orientations first would be mathematically
        wrong), not before it as p=infinity's max-modulus combination
        does -- see module docstring. Ignored when ``p == np.inf``.

    Returns
    -------
    dict[int, ndarray]
        Leader arrays, same keys as ``coeffs_by_scale``.
    """
    is_p_inf = (p == np.inf)
    if not is_p_inf and p <= 0:
        raise ValueError(f"p must be positive (or np.inf); got {p!r}")

    weight_explicit = weight is not _UNSET
    if weight is _UNSET:
        weight = 1.0 if cone == 'multifracs' else 'pymultifracs'
    if boundary is _UNSET:
        boundary = 'pad' if cone == 'multifracs' else 'discard'

    if weight == 'pymultifracs':
        weight = 2.0 ** (-dim)
    elif weight is None:
        weight = 1.0

    scales = sorted(coeffs_by_scale)
    if scales != list(range(1, len(scales) + 1)):
        raise ValueError("scales must be contiguous starting at 1")

    if not is_p_inf and cone != 'multifracs':
        raise NotImplementedError(
            f"finite p is only implemented for cone='multifracs' "
            f"(MULTIFRACS's own p-leader construction); "
            f"cone={cone!r} with p={p} is out of scope for this extension."
        )
    if not is_p_inf and weight_explicit and weight != 1.0:
        warnings.warn(
            f"weight={weight!r} was passed explicitly together with "
            f"finite p={p}, but MULTIFRACS's finite-p fold applies no "
            f"per-fold weight (it is always 1.0 -- all scale-dependence "
            f"lives in the 2**(dim*j) factor applied to 'own', once, at "
            f"birth). weight is ignored for finite p.",
            RuntimeWarning,
        )

    leaders = {}

    if cone == 'multifracs' and is_p_inf:
        sans_voisin = None
        for j in scales:
            own = np.abs(coeffs_by_scale[j])
            if sans_voisin is None:
                sans_voisin = own
            else:
                sans_voisin = _fold_children(own, sans_voisin, dim, weight)
            leaders[j] = _neighborhood_max(sans_voisin, dim,
                                            boundary=boundary)
    elif cone == 'multifracs':  # finite p
        base_by_scale = {
            j: (coeffs_by_scale[j] if pre_combined_power
                else np.abs(coeffs_by_scale[j]) ** p)
            for j in scales
        }
        _p_validity_zp(base_by_scale, scales, p)

        sans_voisin = None
        for j in scales:
            own = (2.0 ** (dim * j)) * base_by_scale[j]
            if sans_voisin is None:
                sans_voisin = own
            else:
                sans_voisin = _fold_children(own, sans_voisin, dim, 1.0,
                                              op=np.sum)
            leaders[j] = _neighborhood_max(sans_voisin, dim,
                                            boundary=boundary, op=np.sum)

        for j in scales:
            leaders[j] = (2.0 ** (-dim * j) * leaders[j]) ** (1.0 / p)
    elif cone == 'pymultifracs':
        prev = None
        for j in scales:
            window_max = _neighborhood_max(np.abs(coeffs_by_scale[j]), dim,
                                            boundary=boundary)
            if prev is None:
                leaders[j] = window_max
            else:
                leaders[j] = _fold_children(window_max, prev, dim, weight)
            prev = leaders[j]
    else:
        raise ValueError(f"unknown cone composition: {cone!r}")

    return leaders

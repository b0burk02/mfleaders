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
"""

import itertools

import numpy as np

_UNSET = object()


def _neighborhood_max(coefs, dim, boundary='discard'):
    """Max over a (3,)*dim neighborhood.

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
            return np.max(stacked, axis=0)

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
        out[interior] = np.max(stacked, axis=0)

    return out


def _fold_children(window_max, finer_leader, dim, weight):
    """Combine this scale's neighborhood max with the finer scale's leaders.

    ``finer_leader`` has roughly twice ``window_max``'s extent per axis
    (dyadic subsampling): parent index k's children live at 2k, 2k+1 along
    each axis. Each child contributes ``weight * finer_leader[child]``; the
    result is the elementwise max of ``window_max`` and all ``2**dim``
    weighted children. As in `_neighborhood_max`, a single NaN among these
    ``1 + 2**dim`` candidates invalidates the position (plain ``max``, not
    ``nanmax``) -- a leader needs its whole cone, not just what's available.
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


def compute_leaders(coeffs_by_scale, dim=1, weight=_UNSET,
                    boundary=_UNSET, cone='multifracs'):
    """Build p=infinity wavelet leaders from per-scale coefficient arrays.

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

    Returns
    -------
    dict[int, ndarray]
        Leader arrays, same keys as ``coeffs_by_scale``.
    """
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

    leaders = {}

    if cone == 'multifracs':
        sans_voisin = None
        for j in scales:
            own = np.abs(coeffs_by_scale[j])
            if sans_voisin is None:
                sans_voisin = own
            else:
                sans_voisin = _fold_children(own, sans_voisin, dim, weight)
            leaders[j] = _neighborhood_max(sans_voisin, dim,
                                            boundary=boundary)
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

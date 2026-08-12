# CLAUDE.md — mfleaders

Dimension-generic wavelet-leader multifractal analysis (MFA) in Python. 1D and 2D
from one core, 3D falling out naturally. Intended for eventual public release.
Correctness is the point: the test suite is a primary deliverable, equal to the code.

## Why this package exists
No maintained, pip-installable 2D wavelet-leader MFA exists in Python. The existing
Python packages (pymultifracs, mfanalysis) are 1D only; the 2D reference lives in
Matlab (the PLBMF toolbox). This package fills that gap. Because there is no Python
oracle at 2D, 1D correctness against pymultifracs is what earns trust in the 2D code.

It also has a specific internal purpose: to provide an INDEPENDENT leader-based
estimator of multifractal width, to be compared against an existing wavelet pipeline
(in the separate `phase_dim` project) that uses raw wavelet-coefficient moments and
reports a q>=0 left-branch-only D-span width. That comparison is why independence
matters — see the freeze rule.

## Hard rules (do not violate without explicit sign-off)
1. **Standalone / freeze.** This is a self-contained public package. The existing
   certified wavelet pipeline lives in a SEPARATE project folder named `phase_dim`
   (its main pipeline file is mf_wavelet.py; its results record is
   mhd_certification_results.md), NOT in this repo. That entire folder is FROZEN:
   never locate, import, reproduce, read from, or modify anything inside it. It is the
   eventual comparison baseline, not a dependency. Build against the specs in this
   file only.
2. **Dimension-generic core.** The leader construction is one algorithm at all
   dimensions: neighborhood is 3 in 1D, 3x3 in 2D, 3x3x3 in 3D; the leader is a max
   over that neighborhood across the current scale and all finer scales in the cone.
   Write the neighborhood + cone-max as ONE dimension-parameterized primitive. Never
   fork into 1D-only and 2D-only implementations.
3. **p=infinity leaders first.** Plain wavelet leaders before p-leaders. p-leaders and
   their finite-resolution correction come only after p=infinity matches the oracle.
4. **Never loosen a tolerance to make a test pass.** An unexplained residual is a bug.
   Trace every discrepancy to a named cause (wavelet order, boundary mode, scale range,
   leader vs p-leader). Hiding a residual destroys the oracle, the most valuable asset
   in this project.
5. **Widths are distinct scalars.** Delta-h (Holder-spectrum width) and the log-cumulant
   c2 are the native leader widths. A generalized-dimension "D-span" (a range in D) is a
   DIFFERENT quantity — and it is what the phase_dim pipeline reports. Keep them distinct
   in names, code, and tests. Never conflate. Making these commensurable is the core of
   the eventual comparison, so the distinction must be explicit, not smoothed over.
6. **Tests are written alongside code, not bolted on at the end.**

## Reference conventions (match these, do not invent)
- Leader definition and conventions: Wendt, Roux, Jaffard & Abry (2009), "Wavelet
  leaders and bootstrap for multifractal analysis of images," and the PLBMF toolbox.
- For 1D conventions (scale indexing, boundary handling, finite-resolution correction):
  read pymultifracs source rather than guessing. That is where silent
  "looks-right-but-doesn't-match" discrepancies originate.

## Pipeline
DWT (pywt wavedec/wavedec2, Daubechies, order configurable ~db3-db4)
  -> leaders (dimension-generic core)
  -> structure functions S(q,j) = <L^q>
  -> zeta(q) via linear regression of log2 S(q,j) vs j
  -> Legendre transform -> D(h), BOTH branches
  -> log-cumulants c1, c2, c3

## Validation ladder (build and gate in this order)
1. 1D vs pymultifracs (numerical oracle). Diff leaders, zeta(q), c1/c2/c3, D(h) to
   tolerance on identical inputs. Match wavelet order + boundary mode; ensure BOTH
   sides compute p=infinity leaders. Write this test EARLY; it gates everything.
2. 1D vs analytic: fBm (monofractal -> point, zeta linear) and binomial cascade
   (known closed-form D(h)). Must recover BOTH branches — the right branch is the
   whole reason leaders beat raw coefficients.
3. 2D vs analytic: fBm2d and 2D cascade with known D(h). No Python oracle here; this
   stage relies on the 1D chain already being certified.
4. (optional, later) 2D vs Matlab PLBMF image toolbox, if access is available.

## Milestones / review gates
- **Gate 1 (stop and report):** dimension-generic core + full 1D pipeline + passing
  1D-vs-pymultifracs oracle test, with every non-zero residual explained. Do not start
  2D until this is reviewed.
- Gate 2: 2D pipeline passing analytic validation.
- Later: p-leaders + finite-resolution correction; bootstrap CIs; docs/packaging.

## Environment
- pip install pymultifracs into the env (needed importable by the 1D oracle test).
- Keep the package installable from early on: pyproject.toml, src/ layout, pytest.
- **pymultifracs is pinned to 0.3.1** (pyproject.toml, dev extras). This is
  NOT the project's certified reference -- that is the MATLAB MULTIFRACS
  toolbox (2D; see Gate 2 status below), which mfleaders reproduces to
  ~1e-9. pymultifracs is a non-reference, 1D-only regression check: it and
  mfleaders' `_dwt.py` disagree by exactly one boundary sample per array
  end (mfleaders marks one more position NaN at each end than
  pymultifracs 0.3.1's `_filtering2` does -- confirmed by direct source
  read, 2026-08-06; wherever both sides agree a position is valid, values
  match to 0.0 exactly). The resulting 28 comparison tests
  (test_dwt.py::test_matches_pymultifracs_wavelet_analysis,
  test_oracle_pymultifracs.py's coefficient/zeta/log_cumulants/spectrum
  matches) are marked `xfail` against this specific pinned version, not
  deleted or loosened -- see their xfail reasons. Do not "fix" this by
  changing DWT/leader boundary logic to chase pymultifracs; that would be
  altering MATLAB-certified code to match a non-reference package. If
  bumping the pin, re-run these tests first -- they may start passing (if
  pymultifracs' behavior changes back) or need their xfail reason updated.

## Status
<!-- Keep this current as the build progresses. -->
- [x] Gate 1 (closed 2026-07-09): dimension-generic core + 1D pipeline +
      pymultifracs oracle match + 1D analytic validation (fBm, binomial
      cascade). Oracle match reported 2026-07-09; held open pending
      ground-truth adjudication of the base-scale (j=1) max-vs-sum dispute
      the oracle test surfaced. Adjudicated via fBm/cascade closed-form
      ground truth (tests/test_analytic_validation.py): max-base (this
      implementation) recovers H with ~3x smaller bias than pymultifracs'
      sum-base at j1=1 (9 vs 32 SE from truth, Monte Carlo), converging by
      j1=3 -- confirms max-base is correct and pymultifracs' base case is a
      bug. j1/j2 selection for real use is a principled, data-only rule
      (mfleaders.select_j1/select_j2), not a scan for pymultifracs
      agreement. Known residual at the default floor j1=2: H recovery has a
      small (~1%, -0.0094 +/- 0.0018, 200-trial MC) bias, flat (not
      decaying) across j1=2..5 -- distinguished from the (already-excluded)
      base-scale artifact and attributed to a generic finite-sample
      log-averaging (Jensen) effect; documented in scale_range.py and
      test_j1_2_residual_bias_is_small_and_not_base_scale_contamination.
      Approved closed 2026-07-09.
- [ ] Gate 2: 2D analytic validation. IN PROGRESS, reported 2026-07-09, NOT
      closed. Monofractal (fBm2D) path certified; multifractal (cascade)
      path is not -- per explicit instruction, no 2D science or
      projection-angle test until the cascade recovers its closed form.
      - 2D MONOFRACTAL PATH CERTIFIED. Two real dimension-scaling bugs found
        and fixed: (1) `_dwt2.py`'s L1-normalization exponent needed an
        extra factor of `dim` (was inflating recovered H for fBm2D by ~2x);
        (2) `spectrum.py`'s `Dq = Dq + 1` was hardcoded to the 1D ambient
        dimension instead of `+ dim` (was capping 2D D(q) near 1 instead of
        2). Both confirmed by fBm2D, which passes cleanly on both branches
        (H recovery, D(h) collapse to h=H, D=2).
      - 2D MULTIFRACTAL PATH NOT CERTIFIED. Orientation-combination rule
        checked FIRST against Wendt/Roux/Jaffard/Abry (2009): the reference
        defines the leader as a sup jointly over the spatial neighborhood,
        the scale cone, AND all three orientation subbands (no orientation
        privileged). `combine_orientations` (plain max-modulus over
        LH/HL/HH) + `compute_leaders` implement exactly this -- CONFIRMED,
        no divergence, no implicit HH-dominance bug in the code. The
        discrepancy is therefore in the closed-form ground truth, not the
        combination rule: two hypotheses for the cascade's local exponent
        (hx+hy, the joint measure exponent; min(hx,hy), the classical
        pointwise Holder exponent -- the latter motivated by the user's
        point that edges/filaments legitimately live in LH/HL) were tested
        against measured h(q)/zeta(q) for both a separable and a
        genuinely non-separable 2D cascade. NEITHER hypothesis matches:
        the separable cascade's negative-q branch matches the "joint"
        hypothesis well but its positive-q branch is smaller than either
        hypothesis predicts; the non-separable cascade matches neither
        branch. Genuinely unresolved -- kept as an explicit, strict xfail
        (test_cascade2d_quantitative_match_is_unresolved), not deleted or
        loosened. Because the cascade doesn't recover its closed form, the
        cone weight was NOT re-adjudicated against it (would be fixing one
        unknown by tuning against another, per explicit instruction); it
        remains only weakly supported (fBm2D "not worse than alternatives"
        + the confirmed normalization dimension-scaling mechanism).
      - Known limitations re-checked: adaptive j1 can still misfire on a
        single 2D realization (confirmed, same as 1D). "Positive-q branch
        resolves more loosely" could not be cleanly re-isolated in 2D
        (fBm2D is monofractal, no asymmetry to measure; the cascade's
        asymmetry is entangled with the unresolved closed-form question).
      - fBm2D H-sweep (H=0.2/0.3/0.5/0.7/0.9, 25-trial MC each): recovered
        h(q) tracks true H cleanly across the WHOLE range at every q,
        including q=5 (the most singular-emphasizing moment) -- biases are
        small (~0.01-0.06, no monotonic sag toward small H) and nothing
        like the cascade's positive-q branch, which undershoots its
        closed-form target by a factor of ~2-4x. This rules out generic
        finite-size estimator bias as the explanation: the estimator is
        essentially unbiased on a genuine monofractal, so the cascade
        discrepancy is a closed-form derivation problem, not an estimator
        problem. The min-vs-sum question (or some other structural gap)
        remains live.
      - PLBMF black-box cross-check hand-off prepared at
        `handoff/plbmf_crosscheck/` (input fields as .mat, exact config
        spec, mfleaders' own reference zeta(q)/h(q)/D(q) on those fields).
        mfleaders was not modified to match anything; no PLBMF/MATLAB
        source was read. Awaiting an independent run in a separate session.
      - Cone-weight-vs-WRJA-definition investigation (MULTIFRACS finding):
        confirmed mfleaders' p=infinity leader unconditionally applies
        `weight=2**-dim` (leaders.py `_fold_children`); confirmed, via the
        p->infinity limit of the WRJA p-leader formula (any fixed positive
        weight vanishes as p->inf: `(c*sum(x**p))**(1/p) -> max(x)`), that
        the definition's p=infinity leader is a true UNWEIGHTED sup, and
        that the weight belongs only to the finite-p normalization.
        pymultifracs' own 1D code was independently verified (direct source
        read, Gate 1) to carry this weight into its p=infinity branch too
        -- a likely bug beyond the already-adjudicated base-scale quirk.
        Added `compute_leaders(..., weight=...)` (default
        `'pymultifracs'` = `2**-dim`, unchanged; `1.0` = unweighted) so
        both are directly comparable via one parameter, no forked
        implementation (tests/test_cone_weight_adjudication.py). RESULT:
        weighted vs unweighted give VIRTUALLY IDENTICAL (both still wrong)
        results on both 2D cascades at every j1 tested, and are not
        decisively distinguishable via fBm Monte Carlo in either 1D or 2D
        (unlike the sharp, many-SE base-scale sum-vs-max difference).
        Switching the default to unweighted would break the Gate-1 1D
        pymultifracs oracle outright (order-1 divergence, verified). 
        CONCLUSION: the weight question, while a legitimate and now-
        documented definitional deviation, is NOT the Gate-2 blocker --
        the cascade closed-form mismatch is unchanged either way. Default
        left at `'pymultifracs'`; the strict xfail cascade test remains in
        place and unresolved.
- [x] Gate 2, REFERENCE PARITY WITH MULTIFRACS (closed 2026-08-06, distinct
      from and NOT an upgrade of the multifractal analytic rung below,
      which remains unmet). The 2D cross-check hand-off above (PLBMF
      black-box, `handoff/plbmf_crosscheck/`) was completed against
      MULTIFRACS (Wendt/Abry MATLAB toolbox, DxPx2d.m read directly for
      this purpose -- reading a reference implementation's source to
      determine a boundary/normalization convention was judged to not
      compromise the numerical cross-check's independence, unlike modifying
      mfleaders to match it blindly).
      - THREE further, real, source-confirmed divergences from MULTIFRACS
        were found and fixed beyond the two dimension-scaling bugs above,
        each isolated and confirmed by execution (border/taps/eps-trim/
        dim-normalization candidates were checked and ruled OUT along the
        way, not silently skipped -- see `_dwt2.py`'s docstrings for what
        was tried and reverted): (a) the DWT-level border/boundary
        handling was restructured to mark-and-convolve
        (`_dwt2_step_multifracs`, `border='multifracs'` on
        `wavelet_transform_2d`) instead of length-trimming, matching
        `DxPx2d.m`'s convention of keeping full-length arrays and letting
        a NaN sentinel propagate through subsequent convolutions; (b) the
        cone/neighborhood COMPOSITION ORDER (`compute_leaders`'s `cone`
        parameter) -- confirmed by exact index-set tracing that mfleaders'
        prior composition (fresh 3x3 neighborhood at every scale) ranges
        over a STRICT SUBSET of MULTIFRACS's own construction (cone-only
        fold with the spatial neighborhood applied exactly once, at the
        target scale), the gap widening geometrically with depth -- this
        was the decisive fix, collapsing a persistent zeta(q) residual to
        machine precision; (c) `compute_leaders`'s own `weight`/`boundary`
        defaults were made to depend on `cone` (previously fixed literals
        tied only to the pymultifracs/1D convention), so that calling
        `compute_leaders(coeffs, dim=2)` with NO extra keywords genuinely
        reproduces MULTIFRACS's leader-construction convention -- not an
        untested hybrid of both references, which is what it silently was
        before this was caught in audit. `_dwt2.leaders_2d(image)` is the
        one-call entry point that reproduces MULTIFRACS end to end
        (DWT included) with pure defaults.
      - RESULT: mfleaders' zeta(q), the per-scale log2 S(q,j) it is fit
        from, and the regression intercept all match MULTIFRACS's own
        reported values on the separable p=0.25 binomial cascade
        (`cascade_input.mat`) to machine precision (~1e-8/1e-9) at every
        q in [-5,5] tested, both branches, j=3-6 -- pinned by
        `tests/test_dwt2.py`'s
        `test_multifracs_cone_matches_multifracs_cascade_to_machine_precision`
        and `test_leaders_2d_entry_point_matches_multifracs_cascade_with_
        pure_defaults`.
      - THE PRIOR "UNRESOLVED" CASCADE CLOSED-FORM MISMATCH IS NOW
        UNDERSTOOD, NOT FIXED: MULTIFRACS's own output ALSO diverges from
        `cascade2d_zeta_true`/`cascade2d_h_true` (the "sum"/hx+hy
        measure-spectrum bridge) by the same large margin mfleaders did
        (e.g. at q=+5 on the separable cascade: closed form
        zeta=6.14/h=0.843 vs the now cross-confirmed measured
        zeta=1.41/h=0.108). Two independent implementations agreeing with
        each other and disagreeing with the closed form is the signature
        of a wrong conjecture in the target, not a shared pipeline bug.
        `test_cascade2d_quantitative_match_is_unresolved` (strict xfail)
        was RETIRED -- renamed to
        `test_cascade2d_measure_spectrum_bridge_is_a_confirmed_false_
        conjecture` in `tests/test_analytic_validation_2d.py`, now a
        normal (non-xfail) test that PINS the confirmed divergence rather
        than waiting on it to close. `cascade2d_zeta_true`/`h_true`/
        `D_true` and their separable-cascade counterparts in
        `tests/synthetic.py` are KEPT, labeled as what they are (the
        cascade measure's own spectrum, not a validated 2D leader-spectrum
        target) -- not deleted.
      - WHAT THIS DOES NOT CLAIM: this is reference parity (two
        independent implementations of the same definition agree), not an
        independent analytic ground-truth match. The 2D MULTIFRACTAL
        ANALYTIC validation rung (validation-ladder step 3's "2D vs
        analytic... 2D cascade with known D(h)") REMAINS UNMET -- no
        properly-derived, first-principles 2D leader-spectrum closed form
        exists or has been validated. fBm2D (monofractal) is still the
        ONLY 2D construction with a genuine analytic check passing (H
        recovery, D(h) collapse to a point, both branches). Do not read
        this entry as closing that rung; it closes a different,
        narrower, but real claim.
- [x] p-leaders LEADER CONSTRUCTION (closed 2026-08-11) -- finite p added to
      `leaders.compute_leaders` (new `p` parameter, default `np.inf`,
      byte-identical regression-guarded: `test_p_inf_default_matches_
      explicit_p_inf`, `test_p_inf_leaders_2d_unchanged_by_finite_p_
      addition`) and `_dwt2.leaders_2d` (new `p` parameter). Localized
      extension per explicit instruction: reuses `_neighborhood_max`/
      `_fold_children` unchanged for p=inf, and (for finite p) via a new
      `op` parameter on both (default `np.max`, preserving every existing
      caller; finite p passes `op=np.sum`) -- the DWT and border handling
      were not touched. Traced from MULTIFRACS's `DxPx2d.m` line by line
      (`compute_leader_sans_voisin` / `compute_leader_from_neigbourhood`,
      literally one function each for p=inf and finite p, differing only
      in the max-vs-sum reduction), not re-derived from the textbook WRJA
      formula independently -- see leaders.py's module docstring for the
      exact construction (own(j) = 2**(dim*j)*|coef(j)|**p; fold via SUM
      with weight=1.0, not the p=inf `weight` parameter; neighborhood via
      SUM; finish off AFTER the full scale loop:
      leader(j) = (2**(-dim*j)*accumulated(j))**(1/p)).
      Orientation combination for finite p (`_dwt2.combine_orientations_p`)
      sums `|coef_o|**p` over the 3 orientations BEFORE the recursion --
      PROVEN equivalent to MULTIFRACS's own per-orientation-then-sum-at-
      the-end structure, unlike p=inf's max: every step of the finite-p
      pipeline up to the single final root is a pure sum, and sum (unlike
      max, which relies on idempotence) is associative/commutative/
      distributive across ANY grouping of neighborhood/cone/orientation
      axes, so pre- vs post-combining orientations is provably identical
      as long as `|.|**p` is applied to each raw coefficient exactly once
      and the root is taken exactly once, at the very end.
      VALIDATED against MULTIFRACS's actual `DxPx2d.m` (run directly in
      MATLAB R2026a, not re-derived/assumed): leader values match to
      max relative diff ~4e-11 to ~1e-9 (same tolerance class as the
      already-certified p=inf cross-check) at p=1 and p=2, confirmed on
      TWO independent synthetic fields (128x128 H=0.6 and 96x96 H=0.35)
      plus a spot-check at p=0.5, p=1.5, p=3 on a third field -- all
      matching to the same precision. Comparison is against MULTIFRACS's
      OWN construction with `param_est.plead_corr=0` (its NLcorrPL2d.m
      nonlinearity-correction post-step DISABLED) -- MULTIFRACS's default
      output has that correction baked in, and comparing against it
      uncorrected initially produced a large, confusing residual (traced
      to this, not a bug in the leader recursion -- confirmed by an
      independent literal Python transliteration of DxPx2d.m matching
      mfleaders' abstracted implementation to ~1e-15, and by directly
      diffing the intermediate cone-only `sans_voisin` accumulator against
      a MATLAB debug dump, which already matched to ~1e-9 BEFORE the
      correction was disabled -- see tests/test_pleaders.py's docstring).
      Regression guard: `tests/test_pleaders.py`'s MULTIFRACS cross-check
      test reads a compact JSON extract (scales 2-3 only, the base case at
      scale 1 omitted) of this same validated MATLAB run from a local, NOT
      committed path (`handoff/plbmf_crosscheck/
      multifracs_pleader_reference.json`) -- guarded by
      `pytest.mark.skipif`, so it runs when that file is present locally
      and skips gracefully (not an error) on a fresh clone, where it is
      not shipped. The MATLAB validation itself is real and is reported
      here and in README's Validation section as a cited result, not as a
      shipped, re-runnable artifact.
      p-VALIDITY: MULTIFRACS does not hard-reject an invalid p; `NLcorrPL2d.m`
      computes `zp` (regression slope of `log2(mean(|coefficient(j)|**p))`
      vs j) and warns ("p-leaders undefined... estimates can be biased...
      a smaller p or larger gamint should be selected") when `zp <= 0`,
      rather than refusing to compute. Mirrored here (`leaders.py`'s
      `_p_validity_zp`) as the same kind of warning, minus the gamint
      remedy (see below) -- tested by `test_p_validity_warning_fires_on_
      field_with_many_exact_zeros` / `..._does_not_fire_on_well_behaved_
      field`.
      NOT IMPLEMENTED, explicitly out of THIS extension's stated scope
      ("only the leader-construction step changes"): (a) `gamint`
      (fractional integration applied to coefficients BEFORE leader
      construction, MULTIFRACS's mechanism for giving a low-regularity
      process enough headroom for a chosen p) and its paired `NLcorrPL2d.m`
      nonlinearity correction (multiplies the raw leader by a scale-
      dependent factor after the fact) -- a coefficient-level
      preprocessing feature, not a leader-construction one; (b) finite p
      under `cone='pymultifracs'` (raises `NotImplementedError` --
      pymultifracs' own finite-p support, if any, has not been read or
      validated). The "finite-resolution correction" this project's own
      roadmap referred to is this NLcorrPL2d/gamint machinery -- still
      open, tracked separately from leader construction now that the
      latter is validated.
- [x] p-leaders VALIDATED ON REAL DATA + plead_corr=2 IMPACT QUANTIFIED
      (closed 2026-08-12). Re-ran the MATLAB cross-check above on actual
      Eos HI column-density data (GALFA, the 397x180 and 389x429 rectangles
      used throughout the Eos investigation), not just synthetic fields:
      mfleaders reproduces MATLAB MULTIFRACS's uncorrected (`plead_corr=0`)
      leader construction to machine precision, verified on synthetic
      fields and on real Eos HI data (max rel diff ~1e-10). The
      `plead_corr=2` correction (MATLAB default) is NOT implemented, but
      was verified to be EXACTLY ZERO for c2, Delta-h, and D(q) shape at
      short fit range (j1=2, j2=4, the 3-scale worst case this correction
      would matter most for); it produces only a ~0.06 rigid shift in the
      h(q) location. Therefore width/multifractality results are identical
      to MATLAB-default; only absolute h/H point estimates carry the small
      shift. Mechanistically understood, not just observed: `NLcorrPL2d.m`
      multiplies every leader at a scale by the SAME (pixel-independent)
      constant, which cancels exactly out of the leader-ratio R(j,q,k)
      D(q) is built from, and shifts log|L(j,.)| by a q-independent
      additive constant per scale -- shifting h(q) by the same amount for
      every q (so Delta-h, a difference, cancels it exactly) while leaving
      the variance-based c2 untouched (adding a constant to a set doesn't
      change its variance). This MATLAB cross-check (script, both
      `plead_corr` settings' raw leader output, and the comparison
      summary), plus the earlier synthetic-field MATLAB comparisons, were
      run and kept locally under `handoff/plbmf_crosscheck/` but are NOT
      shipped in this repository -- reported here and in README's
      Validation section as a cited statement rather than as re-runnable
      artifacts.
- [x] lacunarity module ADDED (2026-08-12), explicitly a diagnostic, NOT
      cross-validated -- distinct in kind from every entry above.
      `src/mfleaders/lacunarity.py`: gray-level gliding-box lacunarity
      Lambda(r), a complementary structure diagnostic (gappiness/void
      structure). NOT part of the validated public API and NOT exported
      from `__init__.py` (same convention as the internal 1D pipeline, for
      the same reason: it has not been checked against an external
      reference implementation the way the leader/zeta/spectrum/cumulant
      pipeline has). Compare Lambda(r) CURVES (height at small r, decay
      shape), never a single fitted slope -- fitting one slope was shown,
      in the exploratory work this was extracted from
      (`examples/discriminant_calibration.py`, `examples/eos_lacunarity_
      check.py`), to wash out real shape differences between otherwise
      visibly-different curves. On a full rectangular field the
      boundary handling is trivial (`valid_mask` defaults to all-True); on
      an irregular/masked footprint, pass `valid_mask` explicitly and the
      exterior is excluded by eroding the valid region by the box radius
      -- never by zero-filling it (zero-fill was separately shown, same
      exploratory work, to make an irregular mask numerically
      indistinguishable from a full rectangle).
      A REAL BUG WAS FOUND AND FIXED while writing this module's own test
      suite (`tests/test_lacunarity.py`), not assumed safe by inspection:
      `scipy.ndimage.uniform_filter` propagates a SINGLE NaN across its
      ENTIRE output array, not just positions within a box's reach
      (verified directly: one NaN pixel in a 10x10 test field NaN'd out
      all 100 `uniform_filter` outputs, at every box size tried) -- unlike
      the DWT-level convolution elsewhere in mfleaders (`leaders.py`,
      `_dwt2.py`), which only corrupts positions its kernel actually
      touches. Unhandled, this would have silently defeated the
      erosion-based exclusion: erosion picks out which OUTPUT POSITIONS to
      trust, but the INPUT array itself would already have gone
      entirely-NaN inside `uniform_filter`, leaving nothing valid to read
      regardless of erosion. Fixed by replacing invalid positions with a
      finite placeholder (value irrelevant -- erosion guarantees a kept
      position's box never actually touches one) before filtering, and by
      computing the `binarize='median'`/float-threshold statistics over
      the valid subset only (the same whole-array-NaN exposure applied to
      `np.median`/`.mean()` too). This was never triggered during this
      session's own actual use of the pre-fix version (every real call
      passed already-fully-valid rectangles, never a genuinely masked
      field with the erosion path actually exercised) -- caught only
      because the committed test suite deliberately exercises a real NaN
      region with an explicit `valid_mask`, not the trivial all-valid case.
      Smoke-tested only (`tests/test_lacunarity.py`: smooth field near
      Lambda=1, sparse/structured field well above 1, both curves
      non-increasing with box size, the mask-erosion path free of NaN
      contamination, and the `binarize='median'`-gives-50%-coverage
      caveat) -- NOT an external cross-validation, and not claimed as one.
- [ ] bootstrap CIs
- [ ] docs + packaging
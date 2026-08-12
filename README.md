# mfleaders

Wavelet-leader multifractal analysis for 2D images. Computes the multifractal
structure functions zeta(q), the singularity spectrum D(h), and the
log-cumulants c1, c2, c3.

Where a power-spectrum slope gives a single average roughness, wavelet leaders
give the full distribution of local roughness across an image -- the singularity
spectrum D(h). A smooth, self-similar field yields a narrow, near-monofractal
spectrum; an intermittent field, with sharp structure concentrated on a small
set (filaments, shocks, folded surfaces), yields a broad, skewed one. That width
and skew is the quantity of interest.

## Validation status

- 2D fBm (monofractal) -- passes its own analytic check (recovers H and D=2).
- 2D multifractal -- reproduces the MULTIFRACS MATLAB toolbox (Wendt/Abry,
  https://www.irit.fr/~Herwig.Wendt/software.html) to ~1e-9 in zeta(q) by
  default. This is reference parity with another implementation, not an
  independent analytic ground-truth match.
- Finite-p leaders -- see "Finite-p (p-leaders)" below.
- Lacunarity module -- see "Lacunarity (diagnostic, not cross-validated)"
  below. Explicitly NOT part of the validation status above.

Open: there is no analytic ground-truth check for a genuine 2D multifractal yet
-- only reference parity plus the monofractal (fBm2D) analytic case. Deriving a
correct 2D leader-spectrum closed form for a known cascade would close this.

The leader construction is dimension-generic internally, and a 1D pipeline is
retained (not part of the public interface) purely as a regression guard,
matching the pymultifracs package to float precision on identical inputs.

## Install

    git clone <repo-url>
    cd mfleaders
    pip install -e .

## Usage

The default configuration reproduces the MULTIFRACS toolbox. For a 2D image:

    import numpy as np
    from mfleaders import (
        leaders_2d, zeta_q, multifractal_spectrum, holder_width,
        log_cumulants, select_j2,
    )

    image = ...  # 2D ndarray

    # Moments to sweep: q<0 recovers the large-h ("regular") branch of
    # D(h), q>0 the small-h ("singular") branch.
    Q = np.concatenate([np.arange(-5, 0), np.arange(1, 6)]).astype(float)

    # Reproduces MULTIFRACS's leader construction by default.
    leaders = leaders_2d(image)

    j1 = 2                              # standard cone-depth floor
    j2 = select_j2(leaders, min_nj=100)  # largest scale with enough valid leaders

    _, _, _, zeta, _ = zeta_q(leaders, Q, j1, j2)
    _, _, _, _, hq, Dq = multifractal_spectrum(leaders, Q, j1, j2)
    _, _, _, c = log_cumulants(leaders, n_cumul=3, j1=j1, j2=j2)

    delta_h = holder_width(hq)          # D(h) width
    c1, c2, c3 = c

`multifractal_spectrum`'s `dim` defaults to 2 (the ambient dimension of an
image) -- getting this wrong makes D(q) come out capped near 1 instead of 2.

See `examples/fbm2d_example.py` for a complete, runnable version of this that
generates its own test field and prints zeta(q), D(h), and c1/c2/c3.

## Finite-p (p-leaders)

`leaders_2d` and `leaders.compute_leaders` both take a `p` parameter: default
`p=inf` (plain wavelet leaders, unchanged behavior). Any finite `p > 0` selects
finite-p leaders instead:

    leaders = leaders_2d(image, p=2.0)   # opt-in; default remains p=inf

`p=2` is opt-in and recommended for real or scale-limited data (a small,
masked, or otherwise short-fit-range field), where p=infinity's plain max-based
leader can starve or be more outlier-sensitive than an L^p-averaged one.

**p-validity/regularity caveat**: finite-p leaders are only well-defined for a
process with enough regularity to support the chosen `p`. mfleaders does not
implement MULTIFRACS's `gamint` (fractional integration) escape hatch for
insufficiently regular data; instead it mirrors MULTIFRACS's own diagnostic
(`NLcorrPL2d.m`'s `eta(p)` check) and emits a `RuntimeWarning` when the data's
own coefficient scaling suggests `p` may be too large for it -- proceeding
anyway, not refusing, exactly as MULTIFRACS does. Treat that warning as a
signal to try a smaller `p`, not as a hard failure.

**Validation**: The 2D wavelet-leader construction reproduces the MATLAB
MULTIFRACS toolbox (PLBMF; Wendt & Abry,
https://www.irit.fr/~Herwig.Wendt/software.html) to ~1e-10 in zeta(q),
verified on synthetic and real fields. MULTIFRACS's default `plead_corr=2`
correction (Leonarduzzi et al. 2017, IEEE Trans. Signal Process.
65:3359-3368) is not implemented; it was verified exactly zero for c2,
Delta-h, and D(q) shape at short fit range (~0.06 rigid shift in h(q)
location only), so width results are unaffected.

See `tests/test_pleaders.py` for the regression suite: byte-identical p=inf
behavior and the p-validity warnings always run. The MULTIFRACS finite-p
cross-check itself reads a local JSON reference and is `skipif`-guarded --
it runs when that reference is present and skips gracefully otherwise, since
the MATLAB cross-check artifacts described above are not shipped in this
repository.

## Lacunarity (diagnostic, not cross-validated)

`mfleaders.lacunarity` provides gray-level gliding-box lacunarity Lambda(r), a
complementary structure diagnostic (gappiness/void structure). **It is NOT
cross-validated against an external reference implementation** -- unlike the
wavelet-leader pipeline above, use it as a diagnostic, not a certified
measurement. It is deliberately not exported from `mfleaders/__init__.py`
(the same convention the internal 1D pipeline uses for the same reason):

    from mfleaders.lacunarity import gliding_box_lacunarity_curve

    lam, _ = gliding_box_lacunarity_curve(field, box_sizes=[2, 4, 8, 16, 32])

Compare Lambda(r) **curves** (shape: height at small r, and decay pattern)
across fields, not single fitted power-law slopes -- a slope fit was shown
(in the exploratory work this module was extracted from) to wash out real
shape differences between fields that otherwise look identical when
summarized as one number.

On full rectangular fields (e.g. an MHD simulation cube with no missing
data), the boundary handling is trivial (`valid_mask` defaults to all-True,
so only a border half a box-width wide is trimmed). On irregular/masked
footprints (e.g. a real observational map), pass `valid_mask` explicitly
(e.g. `np.isfinite(field)`): the exterior is excluded by *eroding* the valid
region by the box radius -- a box only contributes if its entire footprint
lies within `valid_mask` -- never by zero-filling the exterior. See the
module's own docstring for a real, non-obvious pitfall this required
handling explicitly (`scipy.ndimage.uniform_filter` propagates a single NaN
across its *entire* output, not just nearby positions).

Smoke-tested (not externally validated) in `tests/test_lacunarity.py`.

## Tests

    pytest

Three tests depend on MULTIFRACS cross-check input fields (*.mat) not included
in the repository; they skip gracefully when absent. The reference outputs they
compare against are included. `tests/test_pleaders.py`'s MULTIFRACS finite-p
cross-check similarly depends on a JSON reference not included in this
repository -- it does not require MATLAB, but skips gracefully (rather than
erroring) when that reference is absent, e.g. on a fresh clone (see "Finite-p
(p-leaders)" above). `tests/test_lacunarity.py` is a smoke test only (see
"Lacunarity" above) -- it is not an external-reference cross-check.

## References

- H. Wendt, S. Roux, S. Jaffard, P. Abry, "Wavelet leaders and bootstrap for
  multifractal analysis of images," Signal Processing 89 (2009) 1100-1114.
- MULTIFRACS / PLBMF toolbox: https://www.irit.fr/~Herwig.Wendt/software.html

# PLBMF cross-check hand-off

Black-box comparison artifacts for an independent (e.g. PLBMF/MATLAB) 2D
wavelet-leader MFA implementation. mfleaders was NOT modified to match
anything here, and no PLBMF/MATLAB source was read or imported while
producing this -- the point of this hand-off is that the two
implementations stay independent.

## Why this exists

mfleaders' own 2D cascade validation (against a closed-form target derived
by hand) is currently unresolved -- see `../../CLAUDE.md`'s Gate 2 status
and `tests/test_analytic_validation_2d.py`. The orientation-combination
rule (max-modulus over LH/HL/HH per Wendt/Roux/Jaffard/Abry 2009) has been
verified correct; the discrepancy is in the derived closed-form ground
truth, not the pipeline. This hand-off lets an independent implementation
(PLBMF) adjudicate the same fields directly, sidestepping the need to get
the closed-form derivation right at all.

## Input fields

### `cascade_input.mat` -- separable 2D binomial cascade

- Variable `field`: 2049 x 2049 array, `F(x,y) = Fx(x) * Fy(y)`, the outer
  product of two identical 1D binomial devil's-staircase CDFs.
- Each 1D staircase: canonical binomial (Bernoulli) multiplicative measure,
  weight `p = 0.25` (and `1-p = 0.75`), `n_levels = 11` splits
  (`2**11 = 2048` intervals, staircase sampled at `m/2048` for
  `m = 0..2048`, giving 2049 points per axis).
- Fully deterministic -- no random seed needed to reproduce (`p` and
  `n_levels` are also stored as scalar variables in the .mat file).
- Construction: `mu(cell) = p**N0 * (1-p)**N1` where N0/N1 count 0/1 bits
  in the cell index's `n_levels`-bit binary expansion (per axis); staircase
  is the cumulative sum, padded with a leading 0.

### `fbm2d_input.mat` -- isotropic 2D fractional Brownian field

- Variable `field`: 512 x 512 array.
- `H = 0.7` (stored as scalar variable `H`).
- Synthesis: Fourier filtering (white noise, 2D FFT, multiply by
  `|f|**-(H+1)` with the DC term zeroed, inverse FFT, real part) --
  approximate, not exact, but self-checked against the direct
  (non-wavelet) structure-function scaling law `Var[F(x+r,y)-F(x,y)] ~
  r**(2H)` before use (see `tests/test_analytic_validation_2d.py`).
- Random seed `12345` (stored as scalar variable `seed`), NumPy
  `default_rng(12345)` -- reproducible only if regenerated with the same
  RNG algorithm (NumPy's PCG64); provided primarily so the exact field
  used here is preserved (`field`), not so it can be regenerated
  byte-for-byte from a different RNG.

## Exact configuration to match

| Parameter | Value |
|---|---|
| Wavelet | Daubechies, 3 vanishing moments (`db3` in PyWavelets / MATLAB Wavelet Toolbox naming) |
| Boundary handling | Zero-padding at each decomposition level (PyWavelets `mode='zero'`; MATLAB Wavelet Toolbox equivalent is zero-padding, NOT symmetric or periodic extension) |
| Decomposition | Separable 2D DWT, one level at a time (LL fed forward), giving 3 detail subbands per level: LH, HL, HH |
| Leader type | **p = infinity (plain wavelet leaders)** -- NOT p-leaders, no finite-resolution correction |
| Neighborhood | 3x3 spatial neighborhood per scale |
| Orientation combination | max modulus over LH, HL, HH (Wendt, Roux, Jaffard & Abry 2009) -- no single orientation privileged |
| Cone weight | mfleaders currently uses `2**-2` per octave when folding a finer scale's leader into the next scale's max (`2**-dim` for dimension `dim=2`) -- **this is the parameter under investigation; do not assume it's right, report what your implementation uses too** |
| Coefficient normalization | L1 convention (`normalization=1` in mfleaders' terms): coefficients scaled by `2**(scale * dim * (0.5 - 1/normalization))` per level, `dim=2` |
| Scale range (cascade) | j1=3, j2=6 (fixed, not adaptively selected -- max decomposable scale for this field is 8; j2 stops 2 short of that to avoid the thinnest, noisiest coarse scale) |
| Scale range (fBm2D) | j1=2, j2=5 (max decomposable scale is 6) |
| q-grid | `[-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]` (10 values; q=0 skipped, trivial) |
| Both branches | q<0 and q>0 both included -- q>0 selects the small-h/singular branch, q<0 the large-h/regular branch (zeta(q) is concave, so h(q)=zeta'(q) is monotonically decreasing in q -- this sign convention was verified, not assumed, against the 1D binomial cascade's closed form) |

## mfleaders reference output

`mfleaders_reference_output.json` contains, for both fields, at the exact
scale range above:
- `zeta`: the structure-function scaling exponents zeta(q)
- `h`: the Hölder exponents h(q) (both branches)
- `D`: the multifractal spectrum D(q) = D(h(q)), ambient-dimension-shifted
  (so it should peak near 2, not 1, near q=0)
- `nj`: number of valid (non-NaN) leaders at every scale (all scales, not
  just the fit range) -- for assessing whether either implementation's
  coarse-scale estimates are being computed from too few samples
- `max_scale`: the deepest scale either implementation should be able to
  reach for that field size and wavelet

## What to compare

Run the same two fields (`cascade_input.mat`, `fbm2d_input.mat`) through
the independent implementation with the configuration above, and compare
its zeta(q)/h(q)/D(q) against `mfleaders_reference_output.json`'s values.

- If they agree: mfleaders' pipeline (including the current cone weight)
  is corroborated, and the problem was specifically in the closed-form
  derivation used for the earlier internal cascade test.
- If they disagree: report exactly where (which q, which branch, which
  field) and by how much -- do not average away directional disagreement
  the way an aggregate error metric would.

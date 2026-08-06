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

## Tests

    pytest

Three tests depend on MULTIFRACS cross-check input fields (*.mat) not included
in the repository; they skip gracefully when absent. The reference outputs they
compare against are included.

## References

- H. Wendt, S. Roux, S. Jaffard, P. Abry, "Wavelet leaders and bootstrap for
  multifractal analysis of images," Signal Processing 89 (2009) 1100-1114.
- MULTIFRACS / PLBMF toolbox: https://www.irit.fr/~Herwig.Wendt/software.html

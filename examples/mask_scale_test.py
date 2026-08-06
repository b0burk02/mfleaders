"""Diagnostic: does the wavelet-leader scale range survive an irregular
(masked) footprint, like a cloud that doesn't fill the whole image?

Every coarser scale needs a full dyadic block of finer-scale data underneath
it (the "cone"); a masked-out region acts like an extra, INTERIOR border, and
border-corrupted leaders are invalidated the same way an image edge is (see
mfleaders.leaders._neighborhood_max). So a ragged real-world footprint can
starve the coarse scales you need for a zeta(q) fit long before it looks like
a problem by eye.

This script runs the SAME synthetic fBm2D field two ways -- full rectangle vs.
masked -- and reports, per scale: the valid-leader count (nj), whether that
scale is usable (nj >= 30), and whether zeta(q) is fittable at all. No real
data values are needed: only a mask (True/1 = valid pixel) is external input,
and the valid region is filled with synthetic fBm so the comparison is
apples-to-apples.

To use with a real mask (e.g. an Eos cloud footprint): set MASK_PATH below to
a .npy or .fits boolean array shaped (N, N). If MASK_PATH doesn't exist, this
script generates a synthetic placeholder mask instead, so it still runs
standalone on a fresh clone.
"""

import os

import numpy as np

from mfleaders import leaders_2d, zeta_q, select_j1, select_j2

# --- set this to a real mask file to test an actual footprint -------------
MASK_PATH = "eos_mask.npy"   # boolean/0-1 array, shape (N, N), True = valid
# ----------------------------------------------------------------------------

N = 256
H_TRUE = 0.7
Q = np.concatenate([np.arange(-5, 0), np.arange(1, 6)]).astype(float)
MIN_NJ = 30  # "usable scale" threshold requested for this diagnostic


def load_mask(path):
    """Load a boolean mask from .npy or .fits/.fit. Extend here for other
    formats (e.g. a FITS extension name, or inverting a NaN-flagged map).
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        return np.load(path).astype(bool)
    if ext in (".fits", ".fit"):
        from astropy.io import fits  # optional dependency, only needed here
        with fits.open(path) as hdul:
            return hdul[0].data.astype(bool)
    raise ValueError(f"unrecognized mask format: {path!r}")


def fbm2d_spectral(n, H, rng):
    """Same generator as examples/fbm2d_example.py -- see there for details."""
    white = rng.standard_normal((n, n))
    W = np.fft.fft2(white)
    fx = np.fft.fftfreq(n)
    fy = np.fft.fftfreq(n)
    FX, FY = np.meshgrid(fx, fy, indexing="ij")
    freq = np.sqrt(FX ** 2 + FY ** 2)
    freq[0, 0] = 1.0
    filt = freq ** (-(H + 1.0))
    filt[0, 0] = 0.0
    return np.fft.ifft2(W * filt).real


def make_placeholder_mask(n, rng):
    """An irregular, cloud-like blob covering ~60% of the frame: threshold
    a smooth random field. Stands in for a real footprint mask when
    MASK_PATH isn't present -- NOT a substitute for testing your actual
    mask before trusting this diagnostic on real data.
    """
    blob = fbm2d_spectral(n, 1.5, rng)
    return blob > np.percentile(blob, 40)


def run_case(name, leaders):
    """Print nj per scale, usable scales, and the zeta(q) fit outcome.
    Returns a small summary dict for the rectangle-vs-mask comparison.
    """
    max_j = max(leaders)
    # zeta_q's `scales`/`nj` cover every scale in `leaders`, independent of
    # the j1/j2 given here -- j1=1, j2=max_j is just a convenient way to
    # pull them all out; only the regression itself needs a real range.
    scales, _, nj, _, _ = zeta_q(leaders, Q, 1, max_j)

    print(f"\n--- {name} ---")
    print(f"scales present: {scales.tolist()}")
    print(f"nj per scale:   {dict(zip(scales.tolist(), nj.tolist()))}")

    usable = scales[nj >= MIN_NJ]
    print(f"usable scales (nj >= {MIN_NJ}): {usable.tolist()}")

    try:
        j2 = select_j2(leaders, min_nj=MIN_NJ)
        j1, _ = select_j1(leaders, j2)
        _, _, _, zeta, _ = zeta_q(leaders, Q, j1, j2)
        if np.all(np.isfinite(zeta)):
            print(f"zeta(q) fit: OK  (j1={j1}, j2={j2})")
            fit_ok, fit_j1, fit_j2 = True, j1, j2
        else:
            print(f"zeta(q) fit: FAILED  (j1={j1}, j2={j2}, non-finite zeta)")
            fit_ok, fit_j1, fit_j2 = False, j1, j2
    except ValueError as e:
        print(f"zeta(q) fit: FAILED  ({e})")
        print(
            "  next options: (1) relax select_j1's min_range/min_j1 -- a "
            "genuine but narrower range may still exist; (2) this footprint "
            "may simply be too small, at this resolution, for wavelet "
            "leaders to say anything -- a lower-resolution map, or a "
            "different diagnostic entirely (e.g. lacunarity, which doesn't "
            "need a cone of finer scales under every point) may be the "
            "right tool for this shape/size instead of forcing leaders to "
            "work here."
        )
        fit_ok, fit_j1, fit_j2 = False, None, None

    return {
        "name": name,
        "max_scale": int(max_j),
        "max_usable_scale": int(usable.max()) if usable.size else None,
        "fit_ok": fit_ok,
        "fit_range": (fit_j1, fit_j2),
    }


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    image = fbm2d_spectral(N, H_TRUE, rng)

    if os.path.exists(MASK_PATH):
        mask = load_mask(MASK_PATH)
        print(f"loaded mask from {MASK_PATH}")
    else:
        mask = make_placeholder_mask(N, np.random.default_rng(1))
        print(f"{MASK_PATH!r} not found -- using a synthetic placeholder "
              f"mask instead (valid fraction: {mask.mean():.2f})")

    masked_image = image.copy()
    masked_image[~mask] = np.nan

    rect_summary = run_case("full rectangle", leaders_2d(image))
    mask_summary = run_case("masked footprint", leaders_2d(masked_image))

    print("\n--- rectangle vs. mask ---")
    print(f"max scale reached:   rect={rect_summary['max_scale']}  "
          f"masked={mask_summary['max_scale']}")
    print(f"max usable scale:    rect={rect_summary['max_usable_scale']}  "
          f"masked={mask_summary['max_usable_scale']}")
    print(f"zeta(q) fit OK:      rect={rect_summary['fit_ok']}  "
          f"masked={mask_summary['fit_ok']}")
    if rect_summary["max_usable_scale"] and mask_summary["max_usable_scale"]:
        lost = rect_summary["max_usable_scale"] - mask_summary["max_usable_scale"]
        print(f"coarse scales lost to the mask: {lost}")

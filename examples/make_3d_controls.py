"""Generate two 3D synthetic volumes at a shared target fractal dimension,
project each to 2D by column-density integration, and check that the
projections separate under a gliding-box lacunarity test.

This is the 3D generalization of examples/make_control_fields.py -- same two
space-filling mechanisms and same "avoid a hard threshold, it corrupts the
power spectrum" engineering choice, extended from a 2D field to a 3D volume:

  SMOOTH-FILL: an isotropic 3D fractional Brownian volume. Mass fills the
  volume by being everywhere at once, smoothly. Statistically homogeneous.

  FOLDED SURFACE ("lung"): mass concentrated in a thin shell wrapping an
  isocontour (level set) of an underlying 3D fractal field -- a folded
  membrane winding through the volume and wrapping voids, the way a lung
  fills a volume with surface area instead of bulk. The shell is a smooth
  Gaussian band around the level set rather than a hard threshold, for the
  same reason as the 2D script: a hard threshold adds sharp edges that bias
  the power-spectrum-slope fit used to measure D.

DIMENSION CONVENTION -- deliberately DIFFERENT from the 2D script's, for a
reason explained here rather than left implicit. make_control_fields.py
measures the GRAPH dimension of a 2D field: embed the field's graph one
dimension up (a surface in R^3) and its dimension is D = (n+1) - H, n=2,
range (2,3). That embedding-dimension-up convention generalizes to a 3D
volume as D = (n+1) - H, n=3, range (3,4) -- but this script's folded-surface
mechanism only ever ADDS roughness on top of its underlying field (thinning
the shell increases D), so it can only push D up from that field's own
baseline; it can reach the CEILING of whatever range is in play, never the
floor. In the 2D script's own range (2,3), the shared default D_TARGET=3.0
sits at that ceiling and is reachable this way (confirmed by running it:
lung converges cleanly to D=3.01). Carrying the same n+1-H convention
straight into n=3 would move that same numeral to the FLOOR of the new
range (3,4) instead -- unreachable by a mechanism that only adds roughness.

Rather than silently reinterpreting "D_TARGET=3.0" against a range where
it is no longer reachable, this script uses the OTHER standard fBm-derived
fractal dimension: the dimension of the field's LEVEL SETS (isosurfaces)
themselves, D = n - H (the isosurface/coastline generalization of
Mandelbrot's 2-H contour-line result to n dimensions), range (n-1, n) =
(2,3) for n=3. This is arguably the more apt quantity here regardless: the
folded-surface field IS constructed as a band around a level set, so its
own isosurface dimension is the natural knob, and D_TARGET=3.0 lands back
at the ceiling of (2,3) -- the same reachable position, by the same
mechanism, as in the 2D script -- rather than at an unreachable floor.
The H-recovery step is unchanged (power spectrum still gives P(k) ~
k^-(2H+n), n=3); only the final H -> D mapping differs from the 2D script.
`measure_D_3d` applies this identically to both fields, so the comparison
is apples-to-apples even though the folded field is a derived
(shell-weighted), not a raw fBm, volume.
"""

import os
from functools import lru_cache

import numpy as np

# --- tunables ---------------------------------------------------------------
N = 256                  # volume size (N x N x N)
D_TARGET = 3.0           # shared target fractal dimension for both volumes
BASE_H_LUNG = 0.5        # roughness of the underlying field the folded
                         # surface's isocontour is drawn from (not the folded
                         # field's own D -- shell thickness, tuned below,
                         # controls that)
MATCH_TOLERANCE = 0.15   # max allowed |D_smooth - D_lung| before this is
                         # reported as a failed comparison, not a pass
PROJ_AXIS = 0            # axis to sum along for the column-density projection
SEED_SMOOTH = 0
SEED_LUNG = 1
OUT_DIR = os.path.dirname(__file__)
# -----------------------------------------------------------------------------


def fbm3d_spectral(n, H, rng):
    """3D generalization of make_control_fields.fbm2d_spectral: isotropic
    fractional Brownian volume via spectral synthesis. The amplitude filter
    exponent is H + ndim/2 = H + 1.5 for this n=3 domain, giving a power
    spectrum P(k) ~ k^-(2H+3) (the n=3 case of the P(k) ~ k^-(2H+n) relation
    described in the module docstring). This step is convention-independent
    -- it is the final H -> D mapping in `measure_D_3d` that differs from
    the 2D script.
    """
    white = rng.standard_normal((n, n, n))
    W = np.fft.fftn(white)
    freqs = np.fft.fftfreq(n)
    fx = freqs[:, None, None]
    fy = freqs[None, :, None]
    fz = freqs[None, None, :]
    freq = np.sqrt(fx ** 2 + fy ** 2 + fz ** 2)
    freq[0, 0, 0] = 1.0
    filt = freq ** (-(H + 1.5))
    filt[0, 0, 0] = 0.0
    return np.fft.ifftn(W * filt).real


@lru_cache(maxsize=2)
def _radial_k_index(n):
    """Integer radial-frequency bin index for an n x n x n grid. Depends
    only on the grid size, not on field values, so it is built once and
    reused across every power-spectrum call in the bisection loop below
    (at n=256 that's dozens of calls that would otherwise each rebuild a
    256^3 frequency-magnitude array).
    """
    freqs = np.fft.fftshift(np.fft.fftfreq(n)) * n
    fx = freqs[:, None, None]
    fy = freqs[None, :, None]
    fz = freqs[None, None, :]
    K = np.sqrt(fx ** 2 + fy ** 2 + fz ** 2)
    return np.rint(K).astype(np.int32)


def radial_power_spectrum_3d(field):
    """Spherically-averaged power spectrum P(k), k in integer bins. Uses
    np.bincount rather than make_control_fields.radial_power_spectrum's
    per-bin boolean mask: at 256^3 a masked loop over ~128 bins would touch
    the grid 128 times; bincount touches it once.
    """
    n = field.shape[0]
    k_idx = _radial_k_index(n)
    F = np.fft.fftn(field - field.mean())
    P = np.fft.fftshift(np.abs(F) ** 2)
    kmax_bin = n // 2
    valid = (k_idx >= 1) & (k_idx < kmax_bin)
    sums = np.bincount(k_idx[valid], weights=P[valid], minlength=kmax_bin)
    counts = np.bincount(k_idx[valid], minlength=kmax_bin)
    Pr = np.full(kmax_bin, np.nan)
    nz = counts > 0
    Pr[nz] = sums[nz] / counts[nz]
    return np.arange(kmax_bin), Pr


def measure_D_3d(field, kfrac=(0.02, 0.25)):
    """Measure the level-set (isosurface) dimension D = 3 - H of a 3D scalar
    field from its own power spectrum (P(k) ~ k^-(2H+3); see module
    docstring for why this convention, not the 2D script's graph-dimension
    one, is used here), fit over the middle of the available k-range --
    same fractional cuts as make_control_fields.measure_D, excluding the
    noisiest large scales and the highest, discretization-affected
    wavenumbers.
    """
    n = field.shape[0]
    kbins, Pr = radial_power_spectrum_3d(field)
    kmin, kmax = kfrac[0] * n / 2, kfrac[1] * n / 2
    mask = (kbins >= kmin) & (kbins <= kmax) & np.isfinite(Pr) & (Pr > 0)
    slope, _ = np.polyfit(np.log(kbins[mask]), np.log(Pr[mask]), 1)
    H = (-slope - 3) / 2
    return 3 - H, H


def make_shell(field, level, thickness):
    """Mass concentrated in a smooth Gaussian shell around `level` -- a
    continuous stand-in for a thin isocontour membrane (avoids the sharp
    edges a hard threshold would add to the power spectrum). Dimension-
    agnostic: works directly on the 3D field.
    """
    return np.exp(-0.5 * ((field - level) / thickness) ** 2)


def tune_shell_thickness(base, level, D_target, lo=0.002, hi=1.0,
                          tol=0.02, max_iter=30):
    """Bisect on shell thickness (log-spaced) until the SHELL FIELD's own
    measured D matches D_target -- thickness is a strong, monotonic (thinner
    -> higher D) handle on this, same as in make_control_fields.py. Returns
    (thickness, D_achieved, converged).
    """
    def D_of(t):
        return measure_D_3d(make_shell(base, level, t * base.std()))[0]

    D_lo, D_hi = D_of(lo), D_of(hi)
    if not (D_lo >= D_target >= D_hi):
        print(f"  WARNING: target D={D_target} is outside this bracket's "
              f"achievable range [{D_hi:.3f}, {D_lo:.3f}] for "
              f"BASE_H_LUNG={BASE_H_LUNG} -- lower BASE_H_LUNG if D_target "
              f"is below this range, raise it if above.")

    mid, D_mid = lo, D_lo
    for _ in range(max_iter):
        mid = np.sqrt(lo * hi)
        D_mid = D_of(mid)
        if abs(D_mid - D_target) < tol:
            return mid, D_mid, True
        if D_mid > D_target:
            lo = mid
        else:
            hi = mid
    return mid, D_mid, False


def gliding_box_lacunarity(mass, box_size):
    """Lambda(r) = mean(M(r)^2) / mean(M(r))^2 over gliding boxes of size
    `box_size`, run here on the 2D column-density projections. Same
    box-mean-for-box-sum substitution as make_control_fields.py (equivalent
    up to a constant factor that cancels in this ratio, and cheaper).
    """
    from scipy.ndimage import uniform_filter
    box_mean = uniform_filter(mass, size=box_size, mode="wrap")
    return np.mean(box_mean ** 2) / np.mean(box_mean) ** 2


if __name__ == "__main__":
    H_target = 3 - D_TARGET

    print(f"generating {N}^3 volumes (this can take a minute or two)...\n")

    # --- SMOOTH-FILL (3D fBm volume) ----------------------------------------
    rng_smooth = np.random.default_rng(SEED_SMOOTH)
    smooth = fbm3d_spectral(N, H_target, rng_smooth)
    smooth = smooth - smooth.min()  # nonnegative "mass"; shifting by a
                                     # constant only touches the DC term,
                                     # excluded from the D fit -- doesn't
                                     # change the measured dimension.
    D_smooth, H_smooth = measure_D_3d(smooth)

    # --- FOLDED SURFACE ("lung") --------------------------------------------
    rng_lung = np.random.default_rng(SEED_LUNG)
    base = fbm3d_spectral(N, BASE_H_LUNG, rng_lung)
    level = np.median(base)
    thickness, D_lung, converged = tune_shell_thickness(base, level, D_TARGET)
    lung = make_shell(base, level, thickness * base.std())

    print(f"D_TARGET = {D_TARGET}\n")
    print(f"smooth-fill: H_input={H_target:.4f}  D_measured={D_smooth:.4f}")
    print(f"folded surface: shell_thickness={thickness:.4f}*std "
          f"(bisection {'converged' if converged else 'did NOT converge'})"
          f"  D_measured={D_lung:.4f}")

    # --- HARD REQUIREMENT: both volumes must share D ------------------------
    mismatch = abs(D_smooth - D_lung)
    print(f"\n|D_smooth - D_lung| = {mismatch:.4f} "
          f"(tolerance {MATCH_TOLERANCE})")
    if mismatch <= MATCH_TOLERANCE:
        print("MATCH -- the two 3D volumes share a fractal dimension; this "
              "is a fair smooth-fill-vs-folded-surface comparison.")
    else:
        print("MISMATCH -- these volumes do NOT share a fractal dimension. "
              "This is a FAILURE to fix (retune BASE_H_LUNG / thickness "
              "bracket / D_TARGET), not a comparison to hand off as-is.")

    # --- project to 2D: column-density integral along one axis -------------
    proj_smooth = smooth.sum(axis=PROJ_AXIS)
    proj_lung = lung.sum(axis=PROJ_AXIS)

    # --- sanity check: does the folded-surface projection read as gappier? --
    print("\ngliding-box lacunarity on the 2D column-density projections "
          "(rough internal check, not a pipeline match):")
    print(f"{'box':>5}  {'smooth':>8}  {'lung':>8}  {'lung/smooth':>12}")
    box_sizes = [4, 8, 16, 32, 64]
    ratios = []
    for box in box_sizes:
        lam_smooth = gliding_box_lacunarity(proj_smooth, box)
        lam_lung = gliding_box_lacunarity(proj_lung, box)
        ratios.append(lam_lung / lam_smooth)
        print(f"{box:5d}  {lam_smooth:8.4f}  {lam_lung:8.4f}  "
              f"{ratios[-1]:12.2f}")
    if all(r > 1 for r in ratios):
        print("SEPARATE -- the folded-surface projection is gappier/more "
              "structured than the smooth-fill projection at every box "
              "size, despite the two source volumes sharing a 3D fractal "
              "dimension.")
    else:
        print("WARNING: the projections do NOT consistently separate -- "
              "something is off, investigate before handing these off.")

    # --- save projected 2D fields -------------------------------------------
    smooth_path = os.path.join(OUT_DIR, "control3d_smooth_proj.npy")
    lung_path = os.path.join(OUT_DIR, "control3d_lung_proj.npy")
    np.save(smooth_path, proj_smooth)
    np.save(lung_path, proj_lung)
    print(f"\nsaved: {smooth_path}")
    print(f"saved: {lung_path}")

    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
        axes[0].imshow(proj_smooth, cmap="viridis")
        axes[0].set_title(f"smooth-fill projection (D_3d={D_smooth:.2f})")
        axes[1].imshow(proj_lung, cmap="viridis")
        axes[1].set_title(f"folded-surface projection (D_3d={D_lung:.2f})")
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
        fig.tight_layout()
        preview_path = os.path.join(OUT_DIR, "control3d_projections_preview.png")
        fig.savefig(preview_path, dpi=150)
        print(f"saved: {preview_path}")
    except ImportError:
        print("(matplotlib not installed -- skipping preview PNG)")

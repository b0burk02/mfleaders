"""Minimal 2D wavelet-leader multifractal analysis example.

Generates its own 2D fractional Brownian field (fBm) at a known Hurst
exponent H, runs it through mfleaders' default (MULTIFRACS-parity) 2D
pipeline, and prints the multifractal structure functions zeta(q), the
singularity spectrum D(h), and the log-cumulants c1/c2/c3.

fBm is monofractal: every point has the same local roughness H, so this is
a sanity check, not a demonstration of genuine multifractality (a narrow,
near-point D(h) IS the expected/correct result here -- see README.md).

No external data needed; runs standalone on a fresh clone.
"""

import numpy as np
from mfleaders import (
    leaders_2d, zeta_q, multifractal_spectrum, holder_width,
    log_cumulants, select_j2,
)


def fbm2d_spectral(n, H, rng):
    """Approximate isotropic 2D fBm via Fourier filtering: white noise,
    multiplied in the frequency domain by |f|^-(H+1) (d=2), inverse
    transformed. Same construction mfleaders' own fBm2D tests use
    (tests/synthetic.py:fbm2d_spectral), inlined here to keep this example
    self-contained.
    """
    white = rng.standard_normal((n, n))
    W = np.fft.fft2(white)

    fx = np.fft.fftfreq(n)
    fy = np.fft.fftfreq(n)
    FX, FY = np.meshgrid(fx, fy, indexing='ij')
    freq = np.sqrt(FX ** 2 + FY ** 2)
    freq[0, 0] = 1.0  # avoid division by zero; DC term filtered out below

    filt = freq ** (-(H + 1.0))
    filt[0, 0] = 0.0

    return np.fft.ifft2(W * filt).real


H_TRUE = 0.7
N = 256

rng = np.random.default_rng(0)
image = fbm2d_spectral(N, H_TRUE, rng)

# Moments to sweep: both q<0 (large-h, "regular" branch) and q>0 (small-h,
# "singular" branch) are needed to recover both sides of D(h).
Q = np.concatenate([np.arange(-5, 0), np.arange(1, 6)]).astype(float)

# leaders_2d(image) reproduces the MULTIFRACS toolbox's leader construction
# by default -- no extra keywords needed for that.
leaders = leaders_2d(image)

# Scaling range [j1, j2]: j1=2 is the standard cone-depth floor (a leader
# needs at least one octave of recursion to mean anything); j2 comes from
# select_j2, which picks the largest scale with enough valid leaders to
# trust (min_nj=100 here) -- both are the same choices mfleaders' own fBm2D
# tests use.
j1 = 2
j2 = select_j2(leaders, min_nj=100)

_, _, _, zeta, _ = zeta_q(leaders, Q, j1, j2)
_, _, _, _, hq, Dq = multifractal_spectrum(leaders, Q, j1, j2)  # dim=2 default
_, _, _, c = log_cumulants(leaders, n_cumul=3, j1=j1, j2=j2)

delta_h = holder_width(hq)
h_peak = hq[np.argmin(np.abs(Q - 1))]  # h(q=1), close to the typical/peak exponent
H_est = zeta[np.argmin(np.abs(Q - 2))] / 2  # H = zeta(2) / 2

print(f"scaling range used: j1={j1}, j2={j2}")
print()
print("zeta(q):")
for qq, z in zip(Q, zeta):
    print(f"  q={qq:+.0f}: {z:.4f}")
print()
print(f"D(h) width (delta_h) = {delta_h:.4f}")
print(f"D(h) peak location, h(q=1)  = {h_peak:.4f}")
print()
print(f"log-cumulants: c1={c[0]:.4f}, c2={c[1]:.4f}, c3={c[2]:.4f}")
print()
print(f"input H = {H_TRUE}, recovered H (zeta(2)/2) = {H_est:.4f}")

try:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot(hq, Dq, 'o-')
    ax.set_xlabel("h")
    ax.set_ylabel("D(h)")
    ax.set_title(f"fBm2D, H={H_TRUE}: D(h), width={delta_h:.3f}")
    fig.savefig("fbm2d_Dh.png", dpi=150)
    print("\nsaved plot to fbm2d_Dh.png")
except ImportError:
    print("\n(matplotlib not installed -- skipping plot)")

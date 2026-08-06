"""Ground-truth signal generators for the analytic validation ladder.

Both generators are deliberately independent of wavelets: fGn/fBm uses
Davies-Harte circulant-embedding FFT synthesis from the exact autocovariance
function; the binomial cascade is a purely combinatorial construction. Using
a wavelet-domain synthesis (e.g. assigning coefficient variances directly)
would make any subsequent wavelet-leader validation circular -- it would
just read back what was put in.

fgn_davies_harte is self-checked against the theoretical autocovariance via
Monte Carlo in test_analytic_validation.py before being trusted as ground
truth.
"""

import numpy as np


def fgn_acf(k, H):
    """Theoretical autocovariance of fractional Gaussian noise, Var=1."""
    k = np.abs(np.asarray(k, dtype=float))
    return 0.5 * (np.abs(k + 1) ** (2 * H) - 2 * k ** (2 * H)
                  + np.abs(k - 1) ** (2 * H))


def fgn_davies_harte(n, H, rng):
    """Exact fGn synthesis via circulant embedding (Davies & Harte, 1987)."""
    m = 2 * n
    row = fgn_acf(np.minimum(np.arange(m), m - np.arange(m)), H)

    lam = np.fft.fft(row).real
    if lam.min() < -1e-8 * lam.max():
        raise ValueError(
            f"circulant embedding not PSD: min eig={lam.min()}, "
            f"max eig={lam.max()} (try larger n)")
    lam = np.maximum(lam, 0)

    Zr = rng.standard_normal(m)
    Zi = rng.standard_normal(m)

    W = np.zeros(m, dtype=complex)
    W[0] = np.sqrt(lam[0] / m) * Zr[0]
    W[m // 2] = np.sqrt(lam[m // 2] / m) * Zr[m // 2]

    k = np.arange(1, m // 2)
    W[k] = np.sqrt(lam[k] / (2 * m)) * (Zr[k] + 1j * Zi[k])
    W[m - k] = np.conj(W[k])

    x = np.fft.ifft(W) * m
    return x.real[:n]


def fbm(n, H, rng):
    """fBm sample path of length n: cumulative sum of fGn."""
    return np.cumsum(fgn_davies_harte(n, H, rng))


def cascade_measure(n_levels, p):
    """Canonical binomial (Bernoulli) multiplicative measure.

    mu(I_k) = p**N0(k) * (1-p)**N1(k) for the 2**n_levels dyadic intervals
    I_k, where N0(k)/N1(k) count 0/1 bits in k's n_levels-bit expansion.
    Deterministic given p -- no randomness in the construction.
    """
    n = n_levels
    N = 2 ** n
    k = np.arange(N)
    N1 = np.zeros(N, dtype=int)
    kk = k.copy()
    for _ in range(n):
        N1 += kk & 1
        kk >>= 1
    N0 = n - N1
    s = 1 - p
    log_mu = N0 * np.log(p) + N1 * np.log(s)
    return np.exp(log_mu)


def cascade_staircase(n_levels, p):
    """Devil's staircase F(x) = mu([0, x]), sampled at x = m / 2**n_levels."""
    mu = cascade_measure(n_levels, p)
    return np.concatenate([[0.0], np.cumsum(mu)])


def cascade_zeta_true(q, p):
    """Closed-form zeta(q) of the cascade MEASURE's own partition function,
    bridged to a wavelet-leader structure-function exponent -- NOT a
    from-first-principles derivation of the leader construction itself.

    sum_k mu(I_k)**q = (p**q + s**q)**n_levels (binomial expansion) gives
    tau(q) = -log2(p**q + s**q), the measure's own mass-exponent (pure
    partition-function/box-counting multifractal formalism -- no wavelets,
    no leaders, no neighborhood, no orientations anywhere in this
    derivation). zeta(q) = 1 + tau(q) is an ASSUMED bridge (leader
    structure-function exponent = ambient dimension + measure mass
    exponent), not derived from the actual leader construction.

    This bridge is EMPIRICALLY VALIDATED in 1D: the real pipeline's
    leader-based zeta(q)/h(q)/D(q) matches this formula closely in
    test_analytic_validation.py, so in 1D it is legitimate ground truth.
    The 2D versions of this same bridge (cascade_zeta_true_2d,
    cascade2d_zeta_true below) are NOT validated the same way -- see their
    docstrings; do not treat them as verified 2D leader-spectrum targets.
    """
    s = 1 - p
    return 1 - np.log2(p ** q + s ** q)


def cascade_t_of_q(q, p):
    s = 1 - p
    return p ** q / (p ** q + s ** q)


def cascade_h_true(q, p):
    """Closed-form h(q) of the cascade MEASURE (tau'(q), the local Holder
    exponent of the measure's own mass distribution) -- see
    cascade_zeta_true's docstring: bridged to the leader-based h(q) by the
    same assumed relationship, validated in 1D, not independently
    re-derived here from the leader construction itself.
    """
    s = 1 - p
    t = cascade_t_of_q(q, p)
    return -(t * np.log2(p) + (1 - t) * np.log2(s))


def cascade_D_true(q, p):
    """Closed-form D(q) = f(h(q)) of the cascade MEASURE: the binary-entropy
    multifractal (box-counting) spectrum of mu itself, already offset to
    range up to 1 (matching the pymultifracs/leader convention), with the
    0*log2(0) = 0 convention at t=0 or t=1. See cascade_zeta_true's
    docstring for the measure-vs-leader distinction; this 1D bridge is
    empirically validated, the 2D ones (below) are not.
    """
    t = np.atleast_1d(cascade_t_of_q(q, p)).astype(float)
    out = np.zeros_like(t)
    for i, ti in enumerate(t):
        term1 = 0.0 if ti <= 0 else ti * np.log2(ti)
        term2 = 0.0 if ti >= 1 else (1 - ti) * np.log2(1 - ti)
        out[i] = -(term1 + term2)
    return out


# ---------------------------------------------------------------------------
# 2D "ground truth" candidates (Gate 2) -- NOT VERIFIED LEADER-SPECTRUM
# TARGETS. Read this before using either cascade_*_2d family below.
# ---------------------------------------------------------------------------
#
# Both 2D closed forms below (this separable one and the non-separable
# cascade2d_* one further down) compute a spectrum of the cascade MEASURE
# itself (its mass/partition-function exponents), generalizing the 1D
# measure formula above, and then apply the SAME "zeta = ambient_dim +
# tau_measure" bridge used in 1D. That bridge is empirically validated in
# 1D (the real pipeline matches it). It is NOT independently validated in
# 2D, and direct measurement CONTRADICTS it: the actual wavelet-leader
# pipeline (3x3 neighborhood, cone, max over LH/HL/HH orientations) does
# not recover either of these formulas on the realized field, for either
# the separable or the non-separable construction (see
# test_analytic_validation_2d.py's
# test_cascade2d_measure_spectrum_bridge_is_a_confirmed_false_conjecture and
# CLAUDE.md's Gate 2 status).
# So: these are measure-theoretic reference values, kept for the record and
# for possible future use if the measure-to-2D-leader bridge is ever
# properly derived or independently confirmed -- they are NOT currently
# usable as 2D ground truth, and must not be treated as such.
#
# Separable construction: F2D(x, y) = Fx(x) * Fy(y), the outer product of
# two independent copies of the SAME 1D binomial staircase. The MEASURE's
# closed form follows from a Jensen/concavity argument: for a product
# measure, the joint local (mass) exponent at (x, y) is the SUM of the two
# marginal exponents, alpha(x,y) = alpha_x(x) + alpha_y(y), and the joint
# multifractal (box-counting) spectrum is the sup-convolution of the two
# (identical) 1D spectra, f_2D(a) = sup_{a1+a2=a} [f(a1) + f(a2)]. Because
# the 1D binary-entropy spectrum f is concave, Jensen's inequality makes
# that sup exact at a1=a2=a/2: f_2D(a) = 2 f(a/2). The same doubling is
# applied to h(q) and, via the (in 2D, unvalidated) zeta=d+tau bridge, to
# zeta(q). Self-consistency check only (not a leader-pipeline check): at
# p=0.5 (F2D(x,y)=xy, the H=1 monofractal case), zeta2D(q)=2q, matching the
# expected box structure function of a bilinear function.


def cascade_staircase_2d(n_levels, p):
    """Separable 2D devil's staircase: outer product of two 1D staircases
    with the same p (isotropic).

    CONFIRMED (see report; corrects an earlier, stale version of this
    docstring): the actual wavelet-leader pipeline (max over orientation
    subbands LH/HL/HH, per Wendt/Roux/Jaffard/Abry 2009) does NOT recover
    cascade_zeta_true_2d/cascade_h_true_2d/cascade_D_true_2d on the
    positive-q (singular) branch at achievable image sizes, and the
    negative-q branch's apparent match is not reliable evidence either --
    see cascade2d_measure below (a genuinely non-separable construction),
    which was ONCE (incorrectly) documented here as matching its own
    closed form; it does not -- it matches NEITHER branch. Neither 2D
    cascade construction currently has a confirmed leader-spectrum closed
    form. This field is kept for the record and because its negative-q
    zeta/h numbers happen to track the measure formula reasonably closely
    (an observation, not a validated target); it is not used for the
    cone-weight adjudication or as a certified 2D ground truth.
    """
    F = cascade_staircase(n_levels, p)
    return np.outer(F, F)


def cascade_zeta_true_2d(q, p):
    """Measure-spectrum zeta(q) for the separable 2D cascade -- NOT a
    verified 2D leader-spectrum target. See the module comment above
    cascade_staircase_2d and that function's docstring.
    """
    return 2 * cascade_zeta_true(q, p)


def cascade_h_true_2d(q, p):
    """Measure-spectrum h(q) for the separable 2D cascade -- NOT a verified
    2D leader-spectrum target. See cascade_staircase_2d's docstring.
    """
    return 2 * cascade_h_true(q, p)


def cascade_D_true_2d(q, p):
    """Measure-spectrum D(q) for the separable 2D cascade -- NOT a verified
    2D leader-spectrum target. See cascade_staircase_2d's docstring.
    """
    return 2 * cascade_D_true(q, p)


# Genuinely non-separable 2D cascade: 4 independent weights (one per
# quadrant), not a product of two 1D cascades. At each of n_levels splits,
# a cell is divided into 4 quadrants indexed by (row_bit, col_bit) in
# {0,1}x{0,1} -> weights[2*row_bit + col_bit]; mu(cell) = product of the
# weights picked at each level along the path to that cell. This avoids the
# separable construction's degeneracy (no axis-aligned smooth/detail
# factorization to give any orientation subband an artificial head start).
#
# cascade2d_zeta_true/h_true/D_true below compute this construction's
# MEASURE spectrum (its mass/partition-function exponents), via the SAME
# method as the 1D case, now with 4 terms instead of 2 (sum_cells
# mu(cell)^q = (sum_i w_i^q)^n_levels, a direct multinomial-expansion
# generalization of the 1D binomial one):
#   tau(q) = -log2(sum_i w_i^q)
#   t_i(q) = w_i^q / sum_i w_i^q
#   h(q)   = -sum_i t_i(q) log2(w_i)                  [= tau'(q)]
#   D(q)   = -sum_i t_i(q) log2(t_i(q))               [= q h(q) - tau(q)]
#   zeta(q) = 2 + tau(q)                               [ambient dimension 2,
#                                                        via the same
#                                                        zeta=dim+tau bridge]
# Self-checked at w = [.25,.25,.25,.25] (uniform -> F2D(x,y)=xy, H=1):
# zeta(q) = 2 + tau(q) = 2 - log2(4 * 0.25**q) = 2q, matching the expected
# bilinear-function box structure function -- a self-consistency check of
# the MEASURE formula, not a check against the actual leader pipeline.
#
# NOT A VERIFIED 2D LEADER-SPECTRUM TARGET, same caveat as the separable
# construction above: the zeta=dim+tau bridge is unvalidated in 2D and is
# directly contradicted by measurement -- the actual wavelet-leader
# pipeline matches NEITHER branch of this formula (confirmed; an earlier,
# now-corrected docstring on cascade_staircase_2d incorrectly claimed this
# construction "matches its own closed form" -- it does not). Kept for the
# record; not currently usable as 2D ground truth. See
# test_analytic_validation_2d.py's
# test_cascade2d_measure_spectrum_bridge_is_a_confirmed_false_conjecture and
# CLAUDE.md's Gate 2 status.


def cascade2d_measure(n_levels, weights):
    """mu(cell) for the 2**n_levels x 2**n_levels grid of cells, via the
    4-weight, non-separable construction described above.
    """
    weights = np.asarray(weights, dtype=float)
    assert weights.shape == (4,)
    n = n_levels
    N = 2 ** n
    log_w = np.log(weights).reshape(2, 2)  # index [row_bit, col_bit]

    row_bits = np.zeros((N, n), dtype=int)
    col_bits = np.zeros((N, n), dtype=int)
    rr = np.arange(N)
    cc = np.arange(N)
    for k in range(n):
        row_bits[:, k] = rr & 1
        col_bits[:, k] = cc & 1
        rr = rr >> 1
        cc = cc >> 1

    log_mu = np.zeros((N, N))
    for k in range(n):
        rb = row_bits[:, k]
        cb = col_bits[:, k]
        level_log_w = log_w[rb][:, cb]  # shape (N, N), broadcasting row x col
        log_mu += level_log_w
    return np.exp(log_mu)


def cascade2d_staircase(n_levels, weights):
    """2D CDF F(x,y) = mu2D([0,x] x [0,y]) via a 2D cumulative sum,
    sampled at x,y = m/2**n_levels.
    """
    mu = cascade2d_measure(n_levels, weights)
    F = np.cumsum(np.cumsum(mu, axis=0), axis=1)
    F = np.pad(F, ((1, 0), (1, 0)))
    return F


def cascade2d_t_of_q(q, weights):
    weights = np.asarray(weights, dtype=float)
    q = np.atleast_1d(q).astype(float)
    wq = weights[None, :] ** q[:, None]  # shape (n_q, 4)
    return wq / wq.sum(axis=1, keepdims=True)


def cascade2d_zeta_true(q, weights):
    """Measure-spectrum zeta(q) for the non-separable 2D cascade -- NOT a
    verified 2D leader-spectrum target. See the module comment above and
    cascade_staircase_2d's docstring.
    """
    weights = np.asarray(weights, dtype=float)
    q = np.atleast_1d(q).astype(float)
    wq_sum = (weights[None, :] ** q[:, None]).sum(axis=1)
    return 2 - np.log2(wq_sum)


def cascade2d_h_true(q, weights):
    """Measure-spectrum h(q) for the non-separable 2D cascade -- NOT a
    verified 2D leader-spectrum target. See the module comment above and
    cascade_staircase_2d's docstring.
    """
    t = cascade2d_t_of_q(q, weights)
    log2_w = np.log2(weights)
    return -(t * log2_w[None, :]).sum(axis=1)


def cascade2d_D_true(q, weights):
    """Measure-spectrum D(q) for the non-separable 2D cascade -- NOT a
    verified 2D leader-spectrum target. See the module comment above and
    cascade_staircase_2d's docstring.
    """
    t = cascade2d_t_of_q(q, weights)
    out = np.zeros(t.shape[0])
    for i in range(t.shape[0]):
        ti = t[i]
        terms = np.where(ti > 0, ti * np.log2(np.where(ti > 0, ti, 1)), 0.0)
        out[i] = -terms.sum()
    return out


def fbm2d_spectral(n, H, rng):
    """Approximate isotropic 2D fBm via Fourier filtering: white noise,
    multiplied in the frequency domain by |f|^-(H + d/2) with d=2, inverse
    transformed. Not exact (unlike the 1D Davies-Harte generator) -- self-
    checked against the theoretical structure-function scaling law (Var of
    increments ~ r^(2H)) via plain finite differences (no wavelets) in
    test_analytic_validation_2d.py before being trusted as ground truth.
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

    field = np.fft.ifft2(W * filt).real
    return field

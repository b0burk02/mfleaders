"""Gliding-box lacunarity: a complementary structure diagnostic (gappiness/
void structure), gray-level Lambda(r).

NOT PART OF mfleaders' VALIDATED PUBLIC API and NOT EXPORTED from
``mfleaders/__init__.py`` -- deliberately, matching the same convention the
internal 1D pipeline uses for the same reason (see ``__init__.py``'s own
docstring). Unlike the wavelet-leader pipeline (``leaders.py``,
``structure.py``, ``spectrum.py``, ``cumulants.py``), THIS MODULE HAS NOT
BEEN CROSS-VALIDATED AGAINST ANY EXTERNAL REFERENCE IMPLEMENTATION. Treat
Lambda(r) as a diagnostic signal -- useful for comparing curve SHAPE
(height at small r, decay pattern) between fields -- not as a certified
measurement. Do not fit a single power-law slope to Lambda(r) and report it
as a definitive statistic on its own; compare curves (a slope fit averages
away exactly the shape differences this diagnostic is useful for -- see
this module's own test suite and the exploratory work it was extracted
from, ``examples/discriminant_calibration.py`` / ``examples/eos_lacunarity_
check.py``, for worked examples of curves that a single slope would have
hidden).

Boundary handling: on a full rectangular field (e.g. an MHD simulation
cube with no missing data), ``valid_mask`` defaults to all-True, so the
erosion below is a no-op beyond trimming a border half a box-width wide --
standard non-periodic gliding-box practice, NOT periodic wraparound (an
earlier draft of this diagnostic used ``scipy.ndimage.uniform_filter``'s
``mode='wrap'``, appropriate only for a periodic synthetic field, and is
not what this module does). On an irregular or masked footprint (a real
observational map with NaN/blank pixels), pass ``valid_mask`` explicitly
(e.g. ``np.isfinite(field)``): the exterior is excluded by ERODING the
valid region by the box radius -- a box only contributes to Lambda(r) if
its ENTIRE footprint lies within ``valid_mask`` -- not by zero-filling the
exterior. Zero-filling a masked exterior fabricates a false edge and was
directly shown (same exploratory work referenced above) to make an
irregular mask numerically indistinguishable from a full rectangle,
inflating the apparent usable footprint; this module never does that.

A real correctness pitfall this module handles explicitly, found while
writing its own test suite (``tests/test_lacunarity.py``) rather than
assumed safe by inspection: ``scipy.ndimage.uniform_filter`` propagates a
SINGLE NaN across its ENTIRE output array, not just positions within a
box's reach (verified directly -- a lone NaN in a 10x10 field NaNs out all
100 uniform_filter outputs, at every box size). This is unlike the
DWT-level convolution elsewhere in mfleaders (``leaders.py``, ``_dwt2.py``),
which only corrupts positions its kernel actually touches. Left unhandled,
this would have silently defeated the erosion-based exclusion above: the
erosion correctly picks out which OUTPUT POSITIONS to trust, but if the
INPUT array itself already went entirely-NaN inside ``uniform_filter``,
there would be nothing valid left to read. This module replaces invalid
positions with a finite placeholder (value irrelevant -- see
``gliding_box_lacunarity_curve``'s implementation) before filtering, so
only genuinely invalid (excluded) positions are ever affected.
"""

import numpy as np
from scipy.ndimage import binary_erosion, uniform_filter


def gliding_box_lacunarity_curve(field, box_sizes, binarize=None, valid_mask=None):
    """Lambda(r) = <M(r)^2> / <M(r)>^2 over gliding boxes, one value per
    entry of ``box_sizes``, computed only at positions where the box lies
    FULLY inside ``valid_mask`` (erosion) -- see module docstring.

    Parameters
    ----------
    field : ndarray, shape (n_rows, n_cols)
        The mass/intensity field. May contain NaN if ``valid_mask`` marks
        those positions invalid; NaN positions are never read once
        excluded by erosion (a box that never touches them is unaffected
        by their value).
    box_sizes : sequence of int
        Gliding-box side lengths to evaluate Lambda(r) at.
    binarize : None | 'median' | float
        None (default): gray-level, no thresholding. ``'median'``:
        threshold at the field's own median -- ALWAYS gives exactly 50%
        coverage for any field by construction, which artificially
        equalizes coverage across fields regardless of their true
        void/mass split; kept only as a labeled cautionary comparison, not
        a recommended default. A float (e.g. 0.1): threshold at that
        fraction of the field's own mean -- a low value acts as a
        physically-motivated "is this a void pixel" floor that respects
        each field's own true coverage fraction instead of forcing 50%.
    valid_mask : ndarray of bool, shape (n_rows, n_cols), optional
        True = valid data. Defaults to all-True (no exterior to exclude).

    Returns
    -------
    lambdas : ndarray, shape (len(box_sizes),)
        Lambda(r) per box size, NaN where no fully-valid box exists at
        that size or where the mean of the valid box values is zero.
    threshold : float | None
        The threshold actually used, if ``binarize`` was not None.
    """
    data = field.astype(float)
    if valid_mask is None:
        valid_mask = np.ones(data.shape, dtype=bool)
    valid_mask = np.asarray(valid_mask, dtype=bool)

    threshold = None
    if binarize is not None:
        # Computed over the VALID subset only -- data may contain NaN at
        # invalid positions, and np.median/.mean over the whole array
        # would themselves come back NaN otherwise.
        valid_vals = data[valid_mask]
        threshold = float(np.median(valid_vals)) if binarize == "median" else float(binarize * valid_vals.mean())
        data = (data > threshold).astype(float)  # NaN > threshold is False -> 0.0, harmless here

    # scipy.ndimage.uniform_filter propagates a SINGLE NaN across its
    # ENTIRE output array, not just positions within the box's reach
    # (verified directly: a lone NaN in a 10x10 field NaNs out all 100
    # uniform_filter outputs, at every box size tested) -- unlike a direct
    # convolution (e.g. leaders.py's DWT step), which only corrupts
    # positions the kernel actually touches. So invalid positions must be
    # replaced with a finite placeholder before filtering, or erosion below
    # would be discarding positions from an already globally-corrupted
    # array. The placeholder's VALUE is irrelevant: erosion discards every
    # output position whose box touches an invalid cell, so a fully-valid
    # (kept) position's box never actually includes a placeholder cell in
    # its average -- only invalid (discarded) positions could, and those
    # are never read.
    safe_data = np.where(valid_mask, data, 0.0)

    lambdas = []
    for box in box_sizes:
        box_mean = uniform_filter(safe_data, size=box, mode="constant", cval=0.0)
        struct = np.ones((box, box), dtype=bool)
        eroded = binary_erosion(valid_mask, structure=struct, border_value=0)
        vals = box_mean[eroded]
        if vals.size == 0:
            lambdas.append(np.nan)
            continue
        m = vals.mean()
        lambdas.append(np.nan if m == 0 else float(np.mean(vals ** 2) / m ** 2))

    return np.array(lambdas), threshold

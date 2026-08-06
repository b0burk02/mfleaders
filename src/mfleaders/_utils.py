"""Small numeric helpers shared across the pipeline."""

import numpy as np


def power(array, exponent):
    """``array ** exponent``, keeping NaN as NaN even when exponent == 0.

    ``np.nan ** 0 == 1.0`` in numpy, which would silently turn a masked
    (missing) coefficient into a valid data point for q=0. This preserves the
    mask instead.
    """
    if exponent == 0:
        nan_mask = np.isnan(array)
        with np.errstate(invalid='ignore'):
            result = array ** exponent
        result = np.where(nan_mask, np.nan, result)
        return result

    return array ** exponent


def leaders_to_array(leaders_by_scale):
    """Sorted list of scales and the (possibly ragged) per-scale arrays."""
    scales = sorted(leaders_by_scale)
    return scales, [leaders_by_scale[j] for j in scales]

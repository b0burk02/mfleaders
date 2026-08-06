"""mfleaders: wavelet-leader multifractal analysis for 2D images.

Public API is 2D-only: `leaders_2d` plus the structure-function/spectrum/
cumulant analysis functions it feeds into. The underlying leader
construction is dimension-generic (works at dim=1 too) and the 1D pipeline
is retained internally as the pymultifracs-validated regression/
correctness guard (see tests/test_oracle_pymultifracs.py) -- it is not
part of the supported public interface and is deliberately not exported
here. Import from the internal modules directly (e.g.
`mfleaders._dwt.wavelet_transform_1d`, `mfleaders.pipeline.analyze_1d`,
`mfleaders.leaders.compute_leaders`) if you need it; those paths may
change without notice.
"""

from ._version import __version__  # noqa: F401
from ._dwt2 import leaders_2d  # noqa: F401
from .structure import zeta_q  # noqa: F401
from .spectrum import multifractal_spectrum, holder_width  # noqa: F401
from .cumulants import log_cumulants  # noqa: F401
from .scale_range import select_j1, select_j2  # noqa: F401

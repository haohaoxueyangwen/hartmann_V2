from __future__ import annotations

"""Shared Matplotlib font styling helpers for repository-wide plots."""

from numbers import Real

from matplotlib.font_manager import FontProperties
from matplotlib.text import Text

PLOT_FONT_DELTA_PT = 2.0

_STYLE_PATCHED = False
_ORIGINAL_TEXT_SET_FONTSIZE = None


def _resolve_fontsize_points(fontsize):
    """Resolve Matplotlib font size specifiers to points when possible."""
    if fontsize is None:
        return None
    if isinstance(fontsize, Real):
        return float(fontsize)
    try:
        return float(FontProperties(size=fontsize).get_size_in_points())
    except Exception:
        try:
            return float(fontsize)
        except Exception:
            return None


def bump_fontsize(fontsize, delta_pt: float = PLOT_FONT_DELTA_PT):
    """Return a font size increased by the requested delta in points."""
    if fontsize is None:
        return None
    resolved = _resolve_fontsize_points(fontsize)
    if resolved is None:
        return fontsize
    return resolved + float(delta_pt)


def apply_plot_font_bump(delta_pt: float = PLOT_FONT_DELTA_PT) -> None:
    """Patch Matplotlib text sizing so repository plots render with larger fonts."""
    global PLOT_FONT_DELTA_PT
    global _STYLE_PATCHED
    global _ORIGINAL_TEXT_SET_FONTSIZE

    PLOT_FONT_DELTA_PT = float(delta_pt)
    if _STYLE_PATCHED:
        return

    _ORIGINAL_TEXT_SET_FONTSIZE = Text.set_fontsize

    def _patched_set_fontsize(self, fontsize):
        return _ORIGINAL_TEXT_SET_FONTSIZE(self, bump_fontsize(fontsize, PLOT_FONT_DELTA_PT))

    Text.set_fontsize = _patched_set_fontsize
    Text.set_size = _patched_set_fontsize
    _STYLE_PATCHED = True


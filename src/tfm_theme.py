from __future__ import annotations

import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Palette
# Okabe-Ito subset: perceptually uniform and colour-blind safe.
# ---------------------------------------------------------------------------

PALETTE: list[str] = [ # NO SIRVE PARA PERSONAS DALTÓNICAS
    '#722F37',  # wine red   – primary series / main bars
    '#4A7C6F',  # glass green – secondary series / highlights
    '#E8722A',  # warm orange – tertiary series / accents
]

"""
PALETTE: list[str] = [
    '#0072B2',  # blue     – primary series / main bars
    '#E69F00',  # amber    – secondary series / highlights
    '#009E73',  # green    – tertiary series / accents
]
"""

COLOR_PRIMARY: str   = PALETTE[0]
COLOR_SECONDARY: str = PALETTE[1]
COLOR_ACCENT: str    = PALETTE[2]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    'PALETTE',
    'COLOR_PRIMARY',
    'COLOR_SECONDARY',
    'COLOR_ACCENT',
    'apply_theme',
]


def apply_theme() -> None:
    '''
    Apply the project-wide Matplotlib theme.

    Sets the default colour cycle and a minimal set of style parameters
    so that every figure produced by tfm_plots (and any notebook) shares
    the same visual identity without touching individual plot functions.

    Call once at the top of each notebook or at module import time.

    Examples
    --------
    >>> from tfm_theme import apply_theme
    >>> apply_theme()
    '''
    plt.rcParams.update(
        {
            # Colour cycle -------------------------------------------------------
            'axes.prop_cycle': plt.cycler(color=PALETTE),

            # Grid ---------------------------------------------------------------
            'axes.grid':       True,
            'grid.alpha':      0.25,
            'grid.linestyle':  '--',

            # Spines -------------------------------------------------------------
            'axes.spines.top':   False,
            'axes.spines.right': False,

            # Font ---------------------------------------------------------------
            'font.size':        11,
            'axes.titlesize':   13,
            'axes.titleweight': 'bold',
            'axes.labelsize':   11,

            # Figure -------------------------------------------------------------
            'figure.dpi':       100,
            'savefig.dpi':      300,
            'savefig.bbox':     'tight',
        }
    )
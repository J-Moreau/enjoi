import io

import numpy as np
import PIL.Image
from matplotlib import pyplot as plt
from numpy.typing import NDArray


def convert_fig_to_array(fig: plt.Figure) -> NDArray:
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    img = PIL.Image.open(buf)
    return np.array(img).transpose(2, 0, 1)


def rcparams(fraction=1, subplots=(1, 1), beamer=False, columns=1, ieee=False):
    width_pt = 396
    if ieee:
        if columns == 2:
            width_pt = 515.52
        elif columns == 1:
            width_pt = 216
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27
    golden_ratio = (5**0.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])
    if not beamer:
        return {
            "text.usetex": True,
            "font.family": "serif",
            "text.latex.preamble": "\\usepackage{times} ",
            "figure.figsize": (fig_width_in, fig_height_in),
            "figure.constrained_layout.use": True,
            "figure.autolayout": False,
            "savefig.pad_inches": 0.015,
            "font.size": 8,
            "axes.labelsize": 8,
            "legend.fontsize": 6,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "axes.titlesize": 8,
        }
    else:
        width_pt = 307
        fig_width_pt = width_pt * fraction
        # Convert from pt to inches
        inches_per_pt = 1 / 72.27
        golden_ratio = (5**0.5 - 1) / 2

        # Figure width in inches
        fig_width_in = fig_width_pt * inches_per_pt
        # Figure height in inches
        fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])
        return {
            "text.usetex": True,
            "font.family": "sans-serif",
            "text.latex.preamble": r"\usepackage[T1]{fontenc}\usepackage{lmodern}",
            "figure.figsize": (fig_width_in, fig_height_in),
            "figure.constrained_layout.use": True,
            "figure.autolayout": False,
            "savefig.pad_inches": 0.015,
            "font.size": 8,
            "axes.labelsize": 8,
            "legend.fontsize": 6,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "axes.titlesize": 8,
            "mathtext.fontset": "custom",
            "mathtext.rm": "DejaVu Sans",
            "mathtext.it": "DejaVu Sans:italic",
            "mathtext.bf": "DejaVu Sans:bold",
        }

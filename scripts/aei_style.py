"""AEI 论文配图：house style（按 scientific-figure-making 规范实现）。"""
from dataclasses import dataclass
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PALETTE = {
    "blue_main": "#0F4D92", "blue_secondary": "#3775BA",
    "green_1": "#DDF3DE", "green_2": "#AADCA9", "green_3": "#8BCF8B",
    "red_1": "#F6CFCB", "red_2": "#E9A6A1", "red_strong": "#B64342",
    "neutral": "#CFCECE", "highlight": "#FFD700",
    "teal": "#42949E", "violet": "#9A4D8E",
}
DEFAULT_COLORS = [PALETTE["blue_main"], PALETTE["green_3"], PALETTE["red_strong"],
                  PALETTE["teal"], PALETTE["violet"], PALETTE["neutral"]]

@dataclass(frozen=True)
class FigureStyle:
    font_size: int = 16
    axes_linewidth: float = 2.5
    use_tex: bool = False
    font_family: tuple = ("DejaVu Sans", "Helvetica", "Arial", "sans-serif")

def apply_publication_style(style: FigureStyle = None):
    s = style or FigureStyle()
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": list(s.font_family),
        "font.size": s.font_size, "axes.linewidth": s.axes_linewidth,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.labelsize": s.font_size, "xtick.labelsize": s.font_size - 2,
        "ytick.labelsize": s.font_size - 2, "legend.fontsize": s.font_size - 3,
        "legend.frameon": False, "xtick.major.width": s.axes_linewidth * 0.8,
        "ytick.major.width": s.axes_linewidth * 0.8, "text.usetex": s.use_tex,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "figure.dpi": 120, "savefig.bbox": "tight",
    })

def create_subplots(nrows=1, ncols=1, figsize=None, **kw):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize or (6*ncols, 4.2*nrows), **kw)
    return fig, np.atleast_1d(np.asarray(axes)).ravel()

def finalize_figure(fig, out_path, formats=("pdf", "png"), dpi=300, close=True, pad=0.05):
    out = Path(out_path); out.parent.mkdir(parents=True, exist_ok=True)
    saved = []
    for f in formats:
        p = out.with_suffix("." + f)
        fig.savefig(p, dpi=dpi, bbox_inches="tight", pad_inches=pad)
        saved.append(p)
    if close: plt.close(fig)
    return saved

def make_grouped_bar(ax, categories, series, labels, ylabel="Value",
                     colors=None, errs=None, annotate=False, fmt="{:.3f}"):
    n, m = len(series), len(categories)
    colors = colors or DEFAULT_COLORS
    x = np.arange(m); w = 0.8 / n
    last = None
    for i, (vals, lab) in enumerate(zip(series, labels)):
        e = errs[i] if errs is not None else None
        last = ax.bar(x + (i - (n-1)/2)*w, vals, w, label=lab,
                      color=colors[i % len(colors)], yerr=e, capsize=4,
                      error_kw=dict(lw=1.6, ecolor="#444444"), zorder=3)
        if annotate: annotate_bars(ax, last, fmt=fmt)
    ax.set_xticks(x); ax.set_xticklabels(categories)
    ax.set_ylabel(ylabel); ax.grid(axis="y", alpha=0.25, zorder=0)
    return last

def annotate_bars(ax, bars, fmt="{:.2f}", fontsize=10, padding=3):
    for b in bars:
        ax.annotate(fmt.format(b.get_height()), (b.get_x()+b.get_width()/2, b.get_height()),
                    textcoords="offset points", xytext=(0, padding),
                    ha="center", fontsize=fontsize)

def make_trend(ax, x, y_series, labels, colors=None, ylabel=None, xlabel=None,
               bands=None, show_shadow=True):
    colors = colors or DEFAULT_COLORS
    for i, (y, lab) in enumerate(zip(y_series, labels)):
        c = colors[i % len(colors)]
        ax.plot(x, y, lw=2.6, label=lab, color=c, zorder=3)
        if show_shadow and bands is not None and bands[i] is not None:
            lo, hi = bands[i]
            ax.fill_between(x, lo, hi, color=c, alpha=0.18, lw=0, zorder=2)
    if ylabel: ax.set_ylabel(ylabel)
    if xlabel: ax.set_xlabel(xlabel)
    ax.grid(alpha=0.25, zorder=0)

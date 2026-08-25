"""论文② 出图的统一样式（IEEE 双栏）。

语义配色（全篇一致，别在单张图里临时换）：
  蓝  = ours / 提出的方法
  绿  = 上界或正向变体（oracle d̂）
  红  = baseline / 对照（PPO、MPC-only）
  中性 = 消融格
"""
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PALETTE = {
    "blue_main": "#0F4D92", "blue_secondary": "#3775BA",
    "green_1": "#DDF3DE", "green_2": "#AADCA9", "green_3": "#8BCF8B",
    "red_1": "#F6CFCB", "red_2": "#E9A6A1", "red_strong": "#B64342",
    "neutral": "#CFCECE", "highlight": "#FFD700",
    "teal": "#42949E", "violet": "#9A4D8E",
}

# 每个消融格固定一个颜色 + 花纹。花纹是为了灰度打印仍可区分 —— IEEE 纸版是灰度。
CELL_STYLE = {
    "ppo":          (PALETTE["red_strong"],   "",   "PPO"),
    "mpc_only":     (PALETTE["red_2"],        "xx", "MPC-only ($\\lambda\\equiv1$)"),
    "react_binary": (PALETTE["neutral"],      "..", "Reactive + binary"),
    "react_soft":   (PALETTE["neutral"],      "//", "Reactive + soft"),
    "pred_binary":  (PALETTE["blue_secondary"], "\\\\", "Predictive + binary"),
    "ours":         (PALETTE["blue_main"],    "",   "Ours (pred. + soft)"),
    "dhat_zero":    (PALETTE["violet"],       "//", "Ours, $\\hat{d}=0$"),
    "dhat_oracle":  (PALETTE["green_3"],      "",   "Ours, $\\hat{d}$ oracle"),
}
CELL_ORDER = ["ppo", "mpc_only", "react_binary", "react_soft",
              "pred_binary", "ours", "dhat_zero", "dhat_oracle"]

SCENE_LABEL = {"calm": "No gust", "nominal": "Nominal\n(train dist.)",
               "strong": "Strong gust\n(OOD amplitude)",
               "fast": "Strong + fast onset\n(OOD amp. + ramp)"}
SCENE_ORDER = ["calm", "nominal", "strong", "fast"]

# 指标的显示名与方向（↓ 越小越好）
METRIC = {
    "stats.wall_violation":   ("Wall-violation rate", "↓"),
    "stats.wall_viol_frac":   ("Violating-step fraction", "↓"),
    "stats.wall_depth":       ("Cumulative intrusion (m·s)", "↓"),
    "stats.min_wall_dist":    ("Min wall clearance (m)", "↑"),
    "stats.tracking_error_ema": ("Tracking RMSE (m)", "↓"),
    "stats.filter_lambda":    ("Mean takeover $\\lambda$", "-"),
    "stats.episode_len":      ("Episode length (steps)", "-"),
}


@dataclass(frozen=True)
class FigureStyle:
    font_size: int = 15
    axes_linewidth: float = 2.0
    font_family: tuple = ("DejaVu Sans", "Helvetica", "Arial", "sans-serif")


def apply_publication_style(style: FigureStyle = FigureStyle()):
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": list(style.font_family),
        "font.size": style.font_size,
        "axes.linewidth": style.axes_linewidth,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.8,
        "legend.frameon": False,
        "xtick.direction": "out",
        "ytick.direction": "out",
        # 矢量导出：字体存成 TeX 可编辑的 Type 42，投稿系统不会把字形栅格化
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
    })


def finalize(fig, out_path, formats=("pdf", "png"), dpi=300, pad=0.05):
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    saved = []
    for f in formats:
        p = out.with_suffix("." + f)
        fig.savefig(p, format=f, dpi=dpi, bbox_inches="tight", pad_inches=pad)
        saved.append(p)
    plt.close(fig)
    return saved

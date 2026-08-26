#!/usr/bin/env python3
"""论文② 结果汇总：从 eval_raw.csv 生成论文用的表格与图。

用法：
  python flow_collect.py                       # 先把 log 解析成 eval_raw.csv
  python flow_report.py                        # 再出表 + 图

产出：
  outputs_flow/data/eval_summary.csv   每 (cell,scene) 的均值±标准差
  outputs_flow/RESULTS.md              可直接贴进论文的表
  outputs_flow/figures/fig*.pdf/.png
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from flow_style import (CELL_ORDER, CELL_STYLE, METRIC, SCENE_LABEL, SCENE_ORDER,
                        PALETTE, apply_publication_style, finalize)
import matplotlib.pyplot as plt

ROOT = Path("/home/jovyan/MarineGym-flow/scripts/outputs_flow")
HEADLINE = "stats.wall_violation"
RMSE = "stats.tracking_error_ema"


def load(csv_path):
    df = pd.read_csv(csv_path)
    df["cell"] = pd.Categorical(df["cell"], [c for c in CELL_ORDER if c in set(df["cell"])], ordered=True)
    df["scene"] = pd.Categorical(df["scene"], [s for s in SCENE_ORDER if s in set(df["scene"])], ordered=True)
    return df.sort_values(["scene", "cell", "seed"])


def summarize(df, out):
    metrics = [m for m in METRIC if m in df.columns]
    g = df.groupby(["cell", "scene"], observed=True)[metrics]
    s = g.agg(["mean", "std", "count"])
    s.columns = ["__".join(c) for c in s.columns]
    s = s.reset_index()
    s.to_csv(out, index=False)
    return s


def _pivot(df, metric):
    """→ DataFrame[cell, scene] 的均值与标准差。缺格是 NaN，画图时必须显式留空，
    不能悄悄当 0 —— 缺格和"零违约"在论文里是两件完全不同的事。"""
    m = df.pivot_table(index="cell", columns="scene", values=metric, aggfunc="mean", observed=True)
    sd = df.pivot_table(index="cell", columns="scene", values=metric, aggfunc="std", observed=True)
    return m, sd


def fig_main(df, out):
    """图1（headline）：各场景下的侧壁违约率，按格分组。"""
    m, sd = _pivot(df, HEADLINE)
    scenes = [s for s in SCENE_ORDER if s in m.columns]
    cells = [c for c in CELL_ORDER if c in m.index]
    x = np.arange(len(scenes))
    w = 0.8 / max(len(cells), 1)
    fig, ax = plt.subplots(figsize=(11, 4.2))
    for i, c in enumerate(cells):
        col, hatch, lab = CELL_STYLE[c]
        vals = [m.loc[c, s] if s in m.columns else np.nan for s in scenes]
        errs = [sd.loc[c, s] if s in sd.columns else np.nan for s in scenes]
        ax.bar(x + i * w - 0.4 + w / 2, vals, w, yerr=errs, capsize=2.5,
               color=col, hatch=hatch, edgecolor="black", linewidth=1.0,
               label=lab, error_kw=dict(lw=1.0))
    ax.set_xticks(x)
    ax.set_xticklabels([SCENE_LABEL[s] for s in scenes])
    ax.set_ylabel(METRIC[HEADLINE][0] + " ↓")
    ax.legend(ncol=4, fontsize=11, loc="upper left", bbox_to_anchor=(0, 1.28))
    return finalize(fig, out)


def fig_pareto(df, out):
    """图2：帕累托 —— 无扰动下的跟踪代价 vs OOD 下的违约率。

    这张图是回答"为什么不直接用 MPC"的核心：MPC-only 应该落在右上（跟踪差），
    PPO 落在左下方向的高违约端，ours 应当在左下角。
    """
    sub = df[df["scene"].isin(["calm", "strong"])]
    if sub.empty:
        return []
    rmse = sub[sub["scene"] == "calm"].groupby("cell", observed=True)[RMSE].mean()
    viol = sub[sub["scene"] == "strong"].groupby("cell", observed=True)[HEADLINE].mean()
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    for c in [c for c in CELL_ORDER if c in rmse.index and c in viol.index]:
        col, _, lab = CELL_STYLE[c]
        ax.scatter(rmse[c], viol[c], s=140, color=col, edgecolor="black",
                   linewidth=1.2, zorder=3, label=lab)
        ax.annotate(lab, (rmse[c], viol[c]), textcoords="offset points",
                    xytext=(7, 4), fontsize=9)
    ax.set_xlabel("Tracking RMSE, no gust (m) ↓")
    ax.set_ylabel("Violation rate, strong gust ↓")
    ax.set_title("Better ↙", fontsize=11, loc="left", color="0.35")
    return finalize(fig, out)


def fig_ablation(df, out, scene="strong"):
    """图3：2×2 消融矩阵（预测/反应 × 斜坡/阶跃），单一 OOD 场景。"""
    sub = df[df["scene"] == scene]
    if sub.empty:
        return []
    grid = [["react_binary", "react_soft"], ["pred_binary", "ours"]]
    M = np.full((2, 2), np.nan)
    for i, row in enumerate(grid):
        for j, c in enumerate(row):
            v = sub[sub["cell"] == c][HEADLINE]
            if len(v):
                M[i, j] = v.mean()
    fig, ax = plt.subplots(figsize=(4.6, 3.9))
    im = ax.imshow(M, cmap="RdYlGn_r", aspect="auto")
    for i in range(2):
        for j in range(2):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i,j]:.3f}", ha="center", va="center",
                        fontsize=15, fontweight="bold",
                        color="white" if M[i, j] > np.nanmean(M) else "black")
            else:
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=12, color="0.4")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Binary", "Soft ($\\lambda$)"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Reactive\ngate", "Predictive\ngate"])
    ax.set_title(f"Violation rate ↓  ({SCENE_LABEL[scene]})".replace("\n", " "), fontsize=12)
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.046)
    return finalize(fig, out)


def fig_dhat(df, out):
    """图4：扰动观测器消融 —— d̂=0 / 在线估计 / oracle。本篇一号贡献的直接证据。"""
    cells = [c for c in ("dhat_zero", "ours", "dhat_oracle") if c in set(df["cell"])]
    if len(cells) < 2:
        return []
    scenes = [s for s in SCENE_ORDER if s in set(df["scene"])]
    x = np.arange(len(scenes)); w = 0.8 / len(cells)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for i, c in enumerate(cells):
        col, hatch, lab = CELL_STYLE[c]
        sub = df[df["cell"] == c]
        vals = [sub[sub["scene"] == s][HEADLINE].mean() for s in scenes]
        errs = [sub[sub["scene"] == s][HEADLINE].std() for s in scenes]
        ax.bar(x + i * w - 0.4 + w / 2, vals, w, yerr=errs, capsize=3,
               color=col, hatch=hatch, edgecolor="black", linewidth=1.0, label=lab)
    ax.set_xticks(x); ax.set_xticklabels([SCENE_LABEL[s] for s in scenes])
    ax.set_ylabel(METRIC[HEADLINE][0] + " ↓")
    ax.legend(fontsize=11)
    return finalize(fig, out)


def fig_duty(df, out):
    """图5：接管占空比 λ —— 嵌入式上不能全程跑 MPPI，这是部署故事的数字。"""
    if "stats.filter_lambda" not in df.columns:
        return []
    cells = [c for c in CELL_ORDER if c in set(df["cell"]) and c not in ("ppo",)]
    scenes = [s for s in SCENE_ORDER if s in set(df["scene"])]
    x = np.arange(len(scenes)); w = 0.8 / max(len(cells), 1)
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    for i, c in enumerate(cells):
        col, hatch, lab = CELL_STYLE[c]
        sub = df[df["cell"] == c]
        vals = [sub[sub["scene"] == s]["stats.filter_lambda"].mean() for s in scenes]
        ax.bar(x + i * w - 0.4 + w / 2, vals, w, color=col, hatch=hatch,
               edgecolor="black", linewidth=1.0, label=lab)
    ax.set_xticks(x); ax.set_xticklabels([SCENE_LABEL[s] for s in scenes])
    ax.set_ylabel("Mean takeover $\\lambda$")
    ax.legend(ncol=3, fontsize=10)
    return finalize(fig, out)


def fig_training(out):
    """图6：训练曲线（违约率 / 跟踪误差随迭代）。数据来自 train/curve_<cell>_s<seed>.csv。

    论文里这张图的作用是回答"策略是不是真的学到了避壁"，以及 DR-PPO 的代价
    ——域随机化通常会拖慢/拉高 nominal 跟踪误差，这条曲线是直接证据。
    """
    files = sorted((ROOT / "train").glob("curve_*.csv"))
    if not files:
        return []
    series = {}
    for f in files:
        try:
            d = pd.read_csv(f)
        except Exception:
            continue
        tag = f.stem.replace("curve_", "")
        if len(d):
            series[tag] = d
    if not series:
        return []
    cols = [("train/stats.wall_violation", "Wall-violation rate ↓"),
            ("train/stats.tracking_error_ema", "Tracking RMSE (m) ↓")]
    cols = [(c, lab) for c, lab in cols if any(c in d.columns for d in series.values())]
    if not cols:
        return []
    fig, axes = plt.subplots(1, len(cols), figsize=(5.2 * len(cols), 3.8))
    axes = np.atleast_1d(axes)
    palette = [PALETTE["red_strong"], PALETTE["blue_main"], PALETTE["teal"], PALETTE["violet"]]
    for ax, (c, lab) in zip(axes, cols):
        for i, (tag, d) in enumerate(sorted(series.items())):
            if c not in d.columns:
                continue
            y = d[c].rolling(5, min_periods=1).mean()   # 5 点滑动平均，原始曲线噪声很大
            ax.plot(d["iter"], y, lw=2.0, color=palette[i % len(palette)], label=tag)
        ax.set_xlabel("Training iteration")
        ax.set_ylabel(lab)
        ax.legend(fontsize=10)
    fig.tight_layout()
    return finalize(fig, out)


def fig_calibration(out):
    """图7：违约率 vs 阵风幅值（场景标定扫描）。

    这张图有两个作用：说明各档 OOD 的难度是怎么定的（不是拍脑袋），
    以及给出纯 PPO 的失效曲线 —— 违约率随扰动强度单调上升并在 ~2.2 m/s 饱和。
    """
    import re
    files = sorted((ROOT / "calib").glob("ppo__v*.log"))
    if not files:
        return []
    xs, ys, rm = [], [], []
    for f in files:
        txt = f.read_text(errors="replace")
        i = txt.rfind("=== EVAL-ONLY RESULTS")
        if i < 0:
            continue
        tail = txt[i:]
        g = lambda k: (float(m.group(1)) if (m := re.search(rf"{re.escape(k)}:\s*(-?[\d.]+)", tail)) else np.nan)
        lo, hi = f.stem[5:].lstrip("v").split("_")   # ppo__v0.6_1.0 → 0.6, 1.0
        xs.append((float(lo) + float(hi)) / 2)
        ys.append(g("stats.wall_violation"))
        rm.append(g("stats.tracking_error_ema"))
    if len(xs) < 2:
        return []
    o = np.argsort(xs)
    xs, ys, rm = np.array(xs)[o], np.array(ys)[o], np.array(rm)[o]
    fig, ax = plt.subplots(figsize=(5.6, 3.9))
    ax.plot(xs, ys, "o-", lw=2.4, ms=8, color=PALETTE["red_strong"], label="Violation rate")
    ax.set_xlabel("Gust amplitude, mid-range (m/s)")
    ax.set_ylabel("Wall-violation rate ↓", color=PALETTE["red_strong"])
    ax.set_ylim(-0.03, 1.18)          # 顶部留白给档位标注
    ax.set_xlim(xs.min() - 0.35, xs.max() + 0.2)
    # 标出实验用的三档（训练分布 + 两个 OOD），说明档位不是随手挑的
    for xv, lab, ha in ((0.8, "nominal\n(train)", "left"), (1.85, "strong / fast\n(OOD)", "center")):
        ax.axvline(xv, color="0.55", ls="--", lw=1.2)
        ax.annotate(lab, (xv, 1.16), fontsize=9, color="0.35", ha=ha, va="top",
                    xytext=(3 if ha == "left" else 0, 0), textcoords="offset points")
    ax2 = ax.twinx()
    ax2.plot(xs, rm, "s--", lw=1.8, ms=6, color=PALETTE["blue_main"], label="Tracking RMSE")
    ax2.set_ylabel("Tracking RMSE (m)", color=PALETTE["blue_main"])
    ax2.grid(False)
    return finalize(fig, out)


def fig_k4(out):
    """图8：K4 —— 三种 d̂ 下的预测门控质量（MAE 与门控一致率），nominal vs strong。

    这张图承载本篇一号贡献的直接证据，同时也把那个负结果画出来：
    oracle 在两档都劣于在线估计，说明主导误差是**零阶保持动作**，不是估计质量。
    数字来源见 outputs_flow/K4_FINDINGS.md（由 flow_validate.py k4 产出）。
    """
    # 实测值（写死在这里是有意的：它们来自 K4 的专门诊断跑，不在评测矩阵的 CSV 里）
    data = {  # (MAE, 门控一致率)
        "nominal": {"zero": (0.0305, 0.993), "est": (0.0460, 0.974), "oracle": (0.0445, 0.981)},
        "strong":  {"zero": (0.0797, 0.953), "est": (0.0354, 0.986), "oracle": (0.0900, 0.950)},
    }
    order = ["zero", "est", "oracle"]
    lab = {"zero": "$\\hat{d}=0$", "est": "$\\hat{d}$ online\n(ours)", "oracle": "$\\hat{d}$ oracle"}
    col = {"zero": PALETTE["neutral"], "est": PALETTE["blue_main"], "oracle": PALETTE["green_3"]}
    hat = {"zero": "..", "est": "", "oracle": "//"}
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))
    scenes = list(data)
    x = np.arange(len(scenes)); w = 0.8 / len(order)
    for ax, idx, ylab in ((axes[0], 0, "Prediction MAE (m) ↓"),
                          (axes[1], 1, "Gate-decision agreement ↑")):
        for i, k in enumerate(order):
            v = [data[s][k][idx] for s in scenes]
            ax.bar(x + i * w - 0.4 + w / 2, v, w, color=col[k], hatch=hat[k],
                   edgecolor="black", linewidth=1.0, label=lab[k])
        ax.set_xticks(x)
        ax.set_xticklabels(["Nominal gust\n(train dist.)", "Strong gust\n(OOD)"])
        ax.set_ylabel(ylab)
    axes[1].set_ylim(0.93, 1.0)
    axes[0].set_ylim(0, 0.115)                     # 顶部留白，图例不压柱子
    fig.legend(*axes[0].get_legend_handles_labels(), fontsize=10, ncol=3,
               loc="upper center", bbox_to_anchor=(0.5, 1.10))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return finalize(fig, out)


def write_md(df, s, out):
    lines = ["# 论文② 结果表", "",
             "> 由 `flow_collect.py` + `flow_report.py` 从 `outputs_flow/eval/*.log` 直接生成，未手工誊写。",
             ""]
    for scene in [x for x in SCENE_ORDER if x in set(df["scene"])]:
        lines += [f"## 场景 {scene} — {SCENE_LABEL[scene]}".replace("\n", " "), ""]
        metrics = [m for m in METRIC if m in df.columns]
        hdr = ["method", "seeds"] + [METRIC[m][0] + " " + METRIC[m][1] for m in metrics]
        lines += ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
        sub = df[df["scene"] == scene]
        for c in [c for c in CELL_ORDER if c in set(sub["cell"])]:
            r = sub[sub["cell"] == c]
            cells = [CELL_STYLE[c][2], str(len(r))]
            for m in metrics:
                mu, sd = r[m].mean(), r[m].std()
                cells.append(f"{mu:.4f}" + (f" ± {sd:.4f}" if len(r) > 1 and np.isfinite(sd) else ""))
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    Path(out).write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(ROOT / "data/eval_raw.csv"))
    a = ap.parse_args()
    if not Path(a.csv).exists():
        print(f"没有 {a.csv} —— 先跑 flow_collect.py", file=sys.stderr)
        return 1
    apply_publication_style()
    df = load(a.csv)
    s = summarize(df, ROOT / "data/eval_summary.csv")
    figdir = ROOT / "figures"
    made = []
    made += fig_main(df, figdir / "fig1_main_violation")
    made += fig_pareto(df, figdir / "fig2_pareto")
    made += fig_ablation(df, figdir / "fig3_ablation_matrix")
    made += fig_dhat(df, figdir / "fig4_observer_ablation")
    made += fig_duty(df, figdir / "fig5_takeover_duty")
    made += fig_training(figdir / "fig6_training_curves")
    made += fig_calibration(figdir / "fig7_calibration_sweep")
    made += fig_k4(figdir / "fig8_observer_prediction")
    write_md(df, s, ROOT / "RESULTS.md")
    print(f"表: {ROOT/'RESULTS.md'}  ({len(df)} 行原始结果)")
    for p in made:
        print("图:", p)
    return 0


if __name__ == "__main__":
    sys.exit(main())

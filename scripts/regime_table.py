#!/usr/bin/env python3
"""扰动强度扫描的汇总表：找出安全滤波器真正能起作用的区间。

关键的两列一起看：
  · 违约率     —— ours 相对 ppo 有没有下降
  · sat_frac   —— 推力是否已经饱和（饱和 = 没有多余控制权限可供重新分配）
"""
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path("/home/jovyan/MarineGym-flow/scripts/outputs_flow/regime")
KEYS = ["stats.wall_violation", "stats.wall_viol_frac", "stats.sat_frac",
        "stats.min_wall_dist", "stats.tracking_error_ema", "stats.filter_lambda",
        "stats.engage_frac", "stats.effort", "stats.power_W", "stats.action_jerk"]


def parse(p: Path):
    txt = p.read_text(errors="replace")
    i = txt.rfind("=== EVAL-ONLY RESULTS")
    if i < 0:
        return None
    tail = txt[i:]
    out = {}
    for k in KEYS:
        m = re.search(rf"{re.escape(k)}:\s*(-?[\d.]+)", tail)
        out[k.replace("stats.", "")] = float(m.group(1)) if m else float("nan")
    return out


def main():
    rows = []
    for p in sorted(ROOT.glob("*__v*.log")):
        cell, sp = p.stem.split("__v")
        d = parse(p)
        if d is None:
            print(f"⚠ 未完成: {p.name}", file=sys.stderr)
            continue
        lo, hi = sp.split("_")
        d.update(cell=cell, gust=f"[{lo},{hi}]", mid=(float(lo) + float(hi)) / 2)
        rows.append(d)
    if not rows:
        print("没有可汇总的结果", file=sys.stderr)
        return 1
    df = pd.DataFrame(rows).sort_values(["mid", "cell"])
    ROOT.parent.joinpath("data").mkdir(exist_ok=True)
    df.to_csv(ROOT.parent / "data/regime.csv", index=False)

    print(f"{'gust':>12}{'cell':>10}{'viol':>8}{'viol_frac':>11}{'sat':>7}"
          f"{'min_wall':>10}{'rmse':>7}{'lambda':>8}{'effort':>8}{'power':>8}")
    for _, r in df.iterrows():
        print(f"{r.gust:>12}{r.cell:>10}{r.wall_violation:>8.3f}{r.wall_viol_frac:>11.3f}"
              f"{r.sat_frac:>7.3f}{r.min_wall_dist:>10.3f}{r.tracking_error_ema:>7.3f}"
              f"{r.filter_lambda:>8.3f}{r.effort:>8.2f}{r.power_W:>8.1f}")

    print("\n== ours 相对 ppo 的违约率变化（负 = 改善）==")
    piv = df.pivot_table(index="gust", columns="cell", values="wall_violation")
    sat = df.pivot_table(index="gust", columns="cell", values="sat_frac")
    if {"ppo", "ours"} <= set(piv.columns):
        for g in piv.index:
            p_, o_ = piv.loc[g, "ppo"], piv.loc[g, "ours"]
            s_ = sat.loc[g, "ppo"] if "ppo" in sat.columns else float("nan")
            rel = (o_ - p_) / p_ * 100 if p_ else float("nan")
            flag = "← 有效区间" if rel < -10 else ("饱和" if s_ > 0.5 else "")
            print(f"{g:>12}  ppo {p_:.3f} → ours {o_:.3f}   {rel:+6.1f}%   "
                  f"推力饱和 {s_:.2f}  {flag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

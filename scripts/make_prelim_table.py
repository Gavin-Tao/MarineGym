#!/usr/bin/env python3
"""生成一份"方向可行性"对比表：现有格子的全部指标 + 各格 vs ours 的显著性。

刻意包含不利结果 —— 这张表的用途是判断方向能不能继续，不是展示。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/jovyan/MarineGym-flow/scripts/outputs_flow")
NAME = {"ppo": "PPO", "mpc_only": "MPC-only (λ≡1)", "react_soft": "Reactive+soft",
        "ours": "Ours (pred.+soft)", "pred_binary": "Pred.+binary",
        "react_binary": "React.+binary", "dhat_zero": "Ours, d̂=0", "dhat_oracle": "Ours, d̂ oracle"}
ORDER = ["ppo", "mpc_only", "react_soft", "ours", "pred_binary", "react_binary",
         "dhat_zero", "dhat_oracle"]
SCENES = ["calm", "nominal", "strong", "fast"]
COLS = [("stats.wall_violation", "违约率↓", "{:.4f}"),
        ("stats.wall_viol_frac", "触壁步占比↓", "{:.4f}"),
        ("stats.min_wall_dist", "min裕度↑", "{:.3f}"),
        ("stats.wall_depth", "累计侵入↓", "{:.4f}"),
        ("stats.tracking_error_ema", "跟踪RMSE↓", "{:.3f}"),
        ("stats.filter_lambda", "平均λ", "{:.3f}"),
        ("stats.wall_correction", "修正量", "{:.3f}"),
        ("stats.episode_len", "ep_len", "{:.0f}"),
        ("episodes", "n", "{:.0f}")]


def main():
    raw = pd.read_csv(ROOT / "data/eval_raw.csv")
    sig = pd.read_csv(ROOT / "data/significance.csv") if (ROOT / "data/significance.csv").exists() \
        else pd.DataFrame()
    L = ["# 论文② 方向可行性对比表（初步）", "",
         "> 由 `flow_collect.py` + `flow_significance.py` 从评测日志直接生成。",
         "> **这是初步数据，不是终版结果**，用途是判断方向能否继续。",
         ""]

    L += ["## ⚠️ 读这张表之前必须知道的三件事", "",
          "1. **绝对水平待定。** 同一 checkpoint 训练时报跟踪 RMSE 0.158、评测报 1.183，",
          "   差 7 倍，原因尚未定位（诊断进行中）。但**格子之间的相对比较仍然有效** ——",
          "   所有格子共用同一条策略、同一条评测路径，该问题对各格同等作用。",
          "2. **方法真正针对的场景(strong/fast)还没跑到 ours。** 目前只有 calm 与 nominal，",
          "   而 K4 的结果显示观测器的收益集中在强扰动区。现在这张表缺的正是关键一档。",
          "3. **新指标（抖振/饱和/功率/生存）是在这批跑完之后才加进环境的**，本表没有。",
          ""]

    for sc in SCENES:
        sub = raw[raw.scene == sc]
        if sub.empty:
            continue
        cells = [c for c in ORDER if c in set(sub.cell)]
        L += [f"## 场景 {sc}", "",
              "| 方法 | " + " | ".join(h for _, h, _ in COLS) + " |",
              "|" + "---|" * (len(COLS) + 1)]
        for c in cells:
            r = sub[sub.cell == c].iloc[0]
            vals = []
            for k, _, f in COLS:
                vals.append(f.format(r[k]) if k in r and pd.notna(r[k]) else "—")
            L.append(f"| {NAME.get(c, c)} | " + " | ".join(vals) + " |")
        L.append("")

    if len(sig):
        k = sig[sig.metric == "stats.wall_violation"]
        L += ["## 违约率显著性：各格 − ours（差 > 0 表示该格更差）", "",
              "| 场景 | 方法 | ours | 该格 | 差 | 95% CI | p | 显著 |",
              "|---|---|---|---|---|---|---|---|"]
        for _, r in k.sort_values(["scene", "cell"]).iterrows():
            p = f"{r.p:.3g}" if "p" in r and np.isfinite(r.get("p", np.nan)) else "—"
            L.append(f"| {r.scene} | {NAME.get(r.cell, r.cell)} | {r.ours_mean:.4f} | "
                     f"{r.cell_mean:.4f} | {r['diff']:+.4f} | "
                     f"[{r.ci_lo:+.3f}, {r.ci_hi:+.3f}] | {p} | "
                     f"{'**是**' if r.significant else '否'} |")
        L.append("")

    out = ROOT / "PRELIM_TABLE.md"
    out.write_text("\n".join(L))
    print(f"→ {out}")
    print("\n".join(L[:6]))
    return 0


if __name__ == "__main__":
    sys.exit(main())

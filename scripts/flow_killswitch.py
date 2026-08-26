#!/usr/bin/env python3
"""K1/K2/K3 判决：从 eval_raw.csv 直接给出"方向能不能继续"的结论。

写成脚本而不是我肉眼看表，是为了让判据在跑之前就固定下来 —— 事后挑口径把
不利结果解释过去，是这类工作最常见的失败模式。

K1  纯 MPPI(λ≡1) 的无扰动跟踪必须显著**差于** PPO。
    不成立 → 结论会变成"直接用 MPC 别用 RL"，λ 混合失去意义，整篇要改方向。
K2  纯 PPO 在阵风场景下的违约率要落在可区分区间（默认 10%–60%）。
    太低 → 场景太易（静态障碍的翻版）；太高 → 没有改进空间。
K3  反应式门控不应该与 ours 打平 —— 尤其在 fast（陡上升沿）档。
    打平 → "必须预测"这个核心论点没有实验支撑。
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HEADLINE = "stats.wall_violation"
RMSE = "stats.tracking_error_ema"


def _m(df, cell, scene, col):
    r = df[(df["cell"] == cell) & (df["scene"] == scene)][col]
    return (float(r.mean()), float(r.std()), len(r)) if len(r) else (np.nan, np.nan, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="/home/jovyan/MarineGym-flow/scripts/outputs_flow/data/eval_raw.csv")
    ap.add_argument("--k2-lo", type=float, default=0.10)
    ap.add_argument("--k2-hi", type=float, default=0.60)
    a = ap.parse_args()
    if not Path(a.csv).exists():
        print(f"没有 {a.csv}", file=sys.stderr)
        return 1
    df = pd.read_csv(a.csv)
    out, verdicts = [], {}

    # K1
    mpc, _, n1 = _m(df, "mpc_only", "calm", RMSE)
    ppo, _, n2 = _m(df, "ppo", "calm", RMSE)
    if n1 and n2:
        ok = mpc > ppo * 1.15                      # MPC 至少差 15% 才算"显著差"
        verdicts["K1"] = ok
        out.append(f"K1  MPC-only 跟踪 RMSE(calm) = {mpc:.4f}  vs  PPO = {ppo:.4f}  "
                   f"→ {'PASS' if ok else 'FAIL'}  (需 MPC 比 PPO 差 >15%)")
        if not ok:
            out.append("    ⚠ MPPI 的跟踪不比 PPO 差 → λ 越大越好，方法自我否定。必须改方向。")
    else:
        out.append("K1  数据不全（需要 mpc_only/calm 与 ppo/calm）")

    # K2 —— 分两个判据，因为 nominal 就是策略的训练分布：
    #  · nominal 违约率**低**才对（分布内 PPO 本来就该安全）。这不是场景失败，
    #    恰恰是"滤波器只在 OOD 才有价值"这个论点成立的前提。
    #  · 至少一个 OOD 档要落在可区分区间，否则没有改进空间。
    v, _, n = _m(df, "ppo", "nominal", HEADLINE)
    if n:
        ok = v <= 0.15
        verdicts["K2.nominal_low"] = ok
        out.append(f"K2  PPO 违约率 [nominal=训练分布] = {v:.3f}  → {'PASS' if ok else 'FAIL'}  "
                   f"(需 ≤0.15：分布内应当已经安全)")
        if not ok:
            out.append("    ⚠ 训练分布内就大量违约 → 策略欠训练或场景本身不可解，先修这个。")
    ood_ok = []
    for sc in ("strong", "fast"):
        v, _, n = _m(df, "ppo", sc, HEADLINE)
        if n:
            ok = a.k2_lo <= v <= a.k2_hi
            ood_ok.append(ok)
            out.append(f"K2  PPO 违约率 [{sc}=OOD] = {v:.3f}  → {'PASS' if ok else 'FAIL'}  "
                       f"(需落在 {a.k2_lo:.2f}–{a.k2_hi:.2f})")
    if ood_ok:
        verdicts["K2.ood_headroom"] = any(ood_ok)
        if not any(ood_ok):
            out.append("    ⚠ 没有任何 OOD 档落在可区分区间 → 跑 run_calibrate.sh 扫阵风幅值重新定档。")

    # K3
    for sc in ("strong", "fast"):
        r, _, n1 = _m(df, "react_soft", sc, HEADLINE)
        o, _, n2 = _m(df, "ours", sc, HEADLINE)
        p, _, n3 = _m(df, "ppo", sc, HEADLINE)
        if n1 and n2 and n3:
            # 反应式应当明显劣于 ours；理想情况下它接近 PPO（论文①里 B-only ≈ PPO 的对应物）
            ok = r > o * 1.25
            verdicts[f"K3.{sc}"] = ok
            near_ppo = abs(r - p) < 0.25 * max(p, 1e-6)
            out.append(f"K3  [{sc}] reactive={r:.3f}  ours={o:.3f}  ppo={p:.3f}  "
                       f"→ {'PASS' if ok else 'FAIL'}   反应式≈PPO: {near_ppo}")
            if not ok:
                out.append("    ⚠ 反应式门控与 ours 打平 → 『必须预测』缺实验支撑，需把上升沿做得更陡。")

    print("\n".join(out) if out else "没有可判决的数据")
    bad = [k for k, v in verdicts.items() if not v]
    print()
    print("结论：" + ("全部通过，可以铺开全量实验。" if not bad else f"未通过 → {bad}；先解决这些再烧 GPU。"))
    return 0


if __name__ == "__main__":
    sys.exit(main())

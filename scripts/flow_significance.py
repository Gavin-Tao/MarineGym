#!/usr/bin/env python3
"""显著性检验：各消融格 vs ours，逐指标给出差值的自助置信区间。

数据来自 `exp_eval.sh` 写出的 `<cell>__<scene>__s<seed>.episodes.csv`（逐 episode 值）。

为什么不做配对检验：各格虽然面对相同的扰动序列（阵风有独立 RNG），但
`episode_stats.pop()` 返回的 episode 顺序取决于各 env 的完成时刻，跨格并不保证一一对应。
硬做配对会把顺序错配当成真实差异。因此用**非配对自助法**（对均值差重采样），
对二值的违约率再补一个两比例 z 检验。这是保守的做法。

输出 outputs_flow/data/significance.csv，并打印可读表。
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

NAME_RE = re.compile(r"^(?P<cell>.+?)__(?P<scene>[a-z]+)__s(?P<seed>\d+)\.episodes$")
METRICS = ["stats.wall_violation", "stats.wall_viol_frac", "stats.wall_depth",
           "stats.min_wall_dist", "stats.tracking_error_ema", "stats.filter_lambda"]
REF = "ours"


def load_dir(d: Path):
    out = {}
    for p in sorted(d.glob("*.episodes.csv")):
        m = NAME_RE.match(p.stem)
        if not m:
            continue
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        if len(df):
            out[(m.group("cell"), m.group("scene"))] = df
    return out


def boot_diff(a, b, n=10000, rng=None):
    """b − a 的均值差自助分布 → (差值, 2.5%, 97.5%)。a=ours, b=对照。"""
    rng = rng or np.random.default_rng(0)
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 5 or len(b) < 5:
        return np.nan, np.nan, np.nan
    d = b.mean() - a.mean()
    ia = rng.integers(0, len(a), (n, len(a)))
    ib = rng.integers(0, len(b), (n, len(b)))
    ds = b[ib].mean(1) - a[ia].mean(1)
    return d, float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))


def two_prop_z(a, b):
    """两比例 z 检验（违约率是每 episode 0/1）。返回 (z, 双侧 p)。"""
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    if na < 5 or nb < 5:
        return np.nan, np.nan
    pa, pb = a.mean(), b.mean()
    p = (a.sum() + b.sum()) / (na + nb)
    se = np.sqrt(p * (1 - p) * (1 / na + 1 / nb))
    if se == 0:
        return 0.0, 1.0
    z = (pb - pa) / se
    # 正态近似的双侧 p（避免依赖 scipy）
    from math import erfc, sqrt
    return float(z), float(erfc(abs(z) / sqrt(2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="含 *.episodes.csv 的评测目录")
    ap.add_argument("--out", default="/home/jovyan/MarineGym-flow/scripts/outputs_flow/data/significance.csv")
    a = ap.parse_args()
    data = load_dir(Path(a.dir))
    if not data:
        print(f"{a.dir} 下没有 *.episodes.csv", file=sys.stderr)
        return 1

    rng = np.random.default_rng(12345)
    rows = []
    scenes = sorted({s for _, s in data})
    for sc in scenes:
        if (REF, sc) not in data:
            print(f"⚠ 场景 {sc} 缺少参照格 '{REF}'，跳过", file=sys.stderr)
            continue
        ref = data[(REF, sc)]
        for (cell, s2), df in sorted(data.items()):
            if s2 != sc or cell == REF:
                continue
            for mt in METRICS:
                if mt not in df.columns or mt not in ref.columns:
                    continue
                d, lo, hi = boot_diff(ref[mt].values, df[mt].values, rng=rng)
                if not np.isfinite(d):
                    continue
                sig = np.isfinite(lo) and (lo > 0 or hi < 0)     # 95% CI 不含 0
                r = dict(scene=sc, cell=cell, metric=mt,
                         ours_mean=float(np.nanmean(ref[mt].values)),
                         cell_mean=float(np.nanmean(df[mt].values)),
                         diff=d, ci_lo=lo, ci_hi=hi, significant=bool(sig),
                         n_ours=len(ref), n_cell=len(df))
                if mt == "stats.wall_violation":
                    z, p = two_prop_z(ref[mt].values, df[mt].values)
                    r.update(z=z, p=p)
                rows.append(r)

    if not rows:
        print("没有可比较的组合", file=sys.stderr)
        return 1
    out = pd.DataFrame(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, index=False)

    key = out[out.metric == "stats.wall_violation"]
    if len(key):
        print("== 违约率：各格 − ours（差>0 表示该格更差）==")
        print(f"{'scene':<9}{'cell':<14}{'ours':>8}{'cell':>8}{'diff':>9}"
              f"{'95%CI':>20}{'p':>9}  显著")
        for _, r in key.sort_values(["scene", "cell"]).iterrows():
            ci = f"[{r.ci_lo:+.3f},{r.ci_hi:+.3f}]"
            p = f"{r.p:.3g}" if "p" in r and np.isfinite(r.get("p", np.nan)) else "-"
            print(f"{r.scene:<9}{r.cell:<14}{r.ours_mean:>8.3f}{r.cell_mean:>8.3f}"
                  f"{r['diff']:>+9.3f}{ci:>20}{p:>9}  {'是' if r.significant else '否'}")
    print(f"\n完整结果 → {a.out}  ({len(out)} 行)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

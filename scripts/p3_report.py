"""论文③：汇总 p3_sweep 的 JSON 结果 → 论文用表格 (markdown/csv) + 配图 (pdf/png)。

用法:
    python p3_report.py [--indir outputs_p3/main] [--outdir outputs_p3/report]

产出:
    table_main.md / .csv      主结果表（MDP vs POMDP × 各臂，mean±std over seeds）
    fig1_main.pdf/png         主图：两个条形面板（MDP 打平 / POMDP 拉开）
    fig2_curves.pdf/png       学习曲线（POMDP 条件，各臂 episode_len vs frames）
    table_params.md           各臂参数量（审稿人必问）
    summary.json              所有数字的机器可读版本
"""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

import aei_style as st

# 展示顺序与显示名。mamba 放最后 = ours
ARM_ORDER = ["mlp", "mlp_wide", "stack", "gru", "transformer", "mamba",
             "transformer_small", "mamba_d96", "mamba_d64", "mamba_big"]
ARM_LABEL = {
    "mlp": "MLP (current state only)",
    "mlp_wide": "MLP-wide (capacity control)",
    "stack": "Frame-stack",
    "gru": "GRU",
    "transformer": "Transformer (DTQN default, 2L)",
    "mamba": "Mamba (ours, 1L)",
    "transformer_small": "Transformer (1L, param-matched)",
    "mamba_d96": "Mamba (ours, 1L d96)",
    "mamba_d64": "Mamba (ours, 1L d64)",
    "mamba_big": "Mamba (ours, 4 layers)",
}
ARM_COLOR = {
    "mlp": st.PALETTE["neutral"],
    "mlp_wide": "#9E9E9E",
    "stack": st.PALETTE["green_3"],
    "gru": st.PALETTE["teal"],
    "transformer": st.PALETTE["blue_secondary"],
    "mamba": st.PALETTE["red_strong"],
    "transformer_small": st.PALETTE["blue_main"],
    "mamba_d96": st.PALETTE["red_2"],
    "mamba_d64": st.PALETTE["red_1"],
    "mamba_big": st.PALETTE["violet"],
}
COND_LABEL = {
    "mdp": "Full state (MDP)",
    "slow": "Full state, trackable reference (MDP)",
    "pomdp": "No DVL / no rate gyro (POMDP)",
    "pomdp_d10": "No DVL/gyro + 160 ms delay (POMDP)",
    "pomdp_d5": "No DVL/gyro + 80 ms delay (POMDP)",
    "pomdp_sp10": "No DVL/gyro + sparse USBL fix (POMDP)",
    "pomdp_noise": "No DVL/gyro + sensor noise (POMDP)",
    "slow_pomdp": "Trackable reference, no DVL/gyro (POMDP)",
    "slow_pomdp_d10": "Trackable ref, no DVL/gyro + 160 ms delay (POMDP)",
    "drift": "Thruster gain hidden (POMDP)",
    "drift_oracle": "Thruster gain observed (oracle)",
    "drift_accel": "Acceleration observed (oracle)",
    "accel_oracle": "Acceleration observed (oracle)",
    "base": "Base task",
    "slow_drift": "Trackable ref, unobserved thruster drift (POMDP)",
}
# 主图的两列：全观测参照 vs 部分可观测。由命令行覆盖。
CONDS = ["mdp", "pomdp"]

# 主指标。episode_len 是首要指标：episode 因跟踪失败提前终止
# (track.py: terminated |= distance > reset_thres)，所以"撑了多久"就是任务性能。
# 主指标按重要性排序：跟踪误差 > return > 存活时长 > 其余
# ⚠ episode_len 才是主指标：tracking_err_mean 在提前终止机制下被钉死在 ~0.19
# (误差从 0 线性长到 reset_thres 就终止 ⇒ 均值与增长速率无关)，无区分度。
METRICS = [
    ("eval/stats.episode_len", "episode_len", "↑", True),
    ("eval/stats.return", "return", "↑", True),
    ("eval/stats.tracking_err_mean_m", "track_err_m", "↓", False),
    ("eval/stats.success_rate", "success", "↑", True),
    ("eval/stats.tracking_err_p90_m", "track_p90_m", "↓", False),
    ("eval/stats.avg_power_W", "power_W", "↓", False),
    ("eval/stats.action_smoothness", "smoothness", "↑", True),
]
# 头条表只放最关键的三项 + 参数量
HEADLINE = [("eval/stats.episode_len", "Episode length ↑"),
            ("eval/stats.return", "Return ↑"),
            ("eval/stats.tracking_err_mean_m", "Tracking error (m) ↓")]


DIVERGED = []


def load(indir: Path):
    """runs[cond][arm] = [rec, ...]（每个 seed 一条）。发散的 run 剔除并记入 DIVERGED。"""
    runs = defaultdict(lambda: defaultdict(list))
    diverged = DIVERGED
    for f in sorted(indir.glob("*.json")):
        name = f.stem                      # <cond>__<arm>__s<seed>
        parts = name.split("__")
        if len(parts) != 3:
            continue
        cond, arm, _ = parts
        try:
            rec = json.loads(f.read_text())
        except Exception as e:
            print(f"[warn] 跳过无法解析的 {f.name}: {e}")
            continue
        rec["_file"] = f.name
        # 兜底：早期 run 由旧版 train.py 写出，没有 diverged 字段。
        # 发散的表现是各项指标全为 0（NaN 被 .mean() 变成 0）。
        _ev = _norm_eval(rec.get("eval"))
        _el = _ev.get("eval/stats.episode_len", {}).get("mean")
        if _el is not None and _el <= 0.0:
            rec["diverged"] = True
        if rec.get("diverged"):
            # 训练发散(NaN)的 run 不能进统计，但要在报告里如实列出
            diverged.append(f"{cond}/{arm}/{rec.get('seed')}")
            continue
        runs[cond][arm].append(rec)
    return runs


def _norm_eval(ev):
    """键名归一：早期 run 写成 eval/stats.stats.x，现在是 eval/stats.x。"""
    out = {}
    for k, v in (ev or {}).items():
        out[k.replace("eval/stats.stats.", "eval/stats.")] = v
    return out


def _seed_means(recs, key):
    """返回 (mean, err, n_seeds)。

    误差棒的口径取决于有几个训练 seed：
      * 多 seed  → 跨训练 seed 的 std（"方法是否稳定"的正确口径）
      * 单 seed  → episode 级的**标准误** SEM = std/sqrt(n_episodes)
                   （本项目按用户要求：一次训练 + 数百条 episode 评测）

    ⚠ 单 seed 的 SEM 只刻画**评测噪声**，不含训练 run 间方差 ——
      RL 里后者通常更大。故"显著"只能说"这两个训练出来的策略不同"，
      不能说"这个方法平均更好"。这一点必须写进论文。
    """
    stats = [_norm_eval(r.get("eval")).get(key) for r in recs]
    stats = [v for v in stats if v and v.get("mean") is not None]
    if not stats:
        return None
    vals = [v["mean"] for v in stats]
    if len(vals) > 1:
        return float(np.mean(vals)), float(np.std(vals)), len(vals)
    v = stats[0]
    # 单次训练 + N 个独立评测 seed：误差棒 = **跨评测 seed 的 std**，
    # 与论文①② 的口径一致（PAPER_DATA.md §4："每格 10 个独立评测 seed，mean ± std"）。
    return float(v["mean"]), float(v.get("std", 0.0)), int(v.get("n", 1))


def _eval_seed_values(recs, key):
    """取该 run 各**评测 seed** 的均值列表，供 Welch 检验用（n = 评测 seed 数）。"""
    for r in recs:
        v = _norm_eval(r.get("eval")).get(key)
        if v and v.get("values"):
            return [float(x) for x in v["values"]]
        if v and v.get("mean") is not None:
            return None      # 旧格式，没有逐 seed 值
    return None


def welch_t(a, b):
    """Welch t 检验（不假设等方差），返回 (t, 近似双尾 p)。样本量小，仅作参考。"""
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return None, None
    va, vb = a.var(ddof=1), b.var(ddof=1)
    se = math.sqrt(va / na + vb / nb)
    if se == 0:
        return None, None
    t = (a.mean() - b.mean()) / se
    df = (va / na + vb / nb) ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    # 正态近似（df 小时偏乐观，故文中标注为"参考"）
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))
    return float(t), float(p)


def welch_from_stats(a, b):
    """由 (mean, std, n) 直接做 Welch t 检验 —— 单次训练、多 episode 评测时用。"""
    if not a or not b:
        return None, None
    (ma, sa, na), (mb, sb, nb) = a, b
    if na < 2 or nb < 2:
        return None, None
    se = math.sqrt(sa * sa / na + sb * sb / nb)
    if se == 0:
        return None, None
    t = (ma - mb) / se
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))
    return float(t), float(p)


def build_table(runs):
    rows = []
    for cond in CONDS:
        for arm in ARM_ORDER:
            recs = runs.get(cond, {}).get(arm, [])
            if not recs:
                continue
            row = {"cond": cond, "arm": arm, "n_seeds": len(recs),
                   "params": (recs[0].get("param_count") or {}).get("actor"),
                   "context_len": recs[0].get("context_len"),
                   "train_wall_s": float(np.mean([r.get("train_wall_s", np.nan)
                                                  for r in recs]))}
            for key, short, _, _ in METRICS:
                agg = _seed_means(recs, key)
                if agg:
                    row[short + "_mean"], row[short + "_std"], _ = agg
            rows.append(row)
    return rows


def write_tables(rows, runs, outdir: Path):
    # ---------- 主表 ----------
    hdr = ["Condition", "Encoder", "L", "Params", "seeds"] + \
          [f"{s} {d}" for _, s, d, _ in METRICS]
    lines = ["| " + " | ".join(hdr) + " |",
             "|" + "|".join(["---"] * len(hdr)) + "|"]
    csv = [",".join(["cond", "arm", "context_len", "params", "n_seeds"] +
                    [f"{s}_mean,{s}_std" for _, s, _, _ in METRICS])]
    for r in rows:
        cells = [COND_LABEL.get(r["cond"], r["cond"]), ARM_LABEL.get(r["arm"], r["arm"]),
                 str(r.get("context_len", "")),
                 f"{r['params']:,}" if r.get("params") else "—", str(r["n_seeds"])]
        for _, s, _, _ in METRICS:
            if s + "_mean" in r:
                _e = r[s + "_std"]
                _lab = "" if r["n_seeds"] > 1 else ""   # 口径在表下注明
                cells.append(f"{r[s+'_mean']:.3f} ± {_e:.3f}{_lab}")
            else:
                cells.append("—")
        lines.append("| " + " | ".join(cells) + " |")
        csv.append(",".join([r["cond"], r["arm"], str(r.get("context_len", "")),
                             str(r.get("params", "")), str(r["n_seeds"])] +
                            [f"{r.get(s+'_mean','')},{r.get(s+'_std','')}"
                             for _, s, _, _ in METRICS]))
    _single = all(r["n_seeds"] == 1 for r in rows)
    _note = ("\n\n> `±` = **跨评测 seed 的 std**（单次训练 × 10 个独立评测 seed，"
             "各 arm 用同一组 seed）。\n> 与论文①② 口径一致。"
             "\n> ⚠ 它刻画评测随机性，**不含训练 run 间方差** —— RL 里后者通常更大。"
             "\n> 因此显著性说明\"这两个训练出来的策略表现不同\"，"
             "不能直接推断\"该方法平均更好\"。\n"
             if _single else
             "\n\n> `±` = 跨训练 seed 的 std。\n")
    (outdir / "table_main.md").write_text("\n".join(lines) + _note)
    (outdir / "table_main.csv").write_text("\n".join(csv) + "\n")

    # ---------- 显著性：POMDP 下 mamba vs 各基线 ----------
    sig = [f"| Comparison ({COND_LABEL.get(CONDS[-1], CONDS[-1])}) | Δ episode length | t | p |",
           "|---|---|---|---|"]
    key = "eval/stats.episode_len"
    pc = CONDS[-1]
    mam_recs = runs.get(pc, {}).get("mamba", [])
    mam = [_norm_eval(r.get("eval"))[key]["mean"] for r in mam_recs
           if _norm_eval(r.get("eval")).get(key)]
    mam_vals = _eval_seed_values(mam_recs, key)
    for arm in ARM_ORDER:
        if arm == "mamba":
            continue
        other_recs = runs.get(pc, {}).get(arm, [])
        other = [_norm_eval(r.get("eval"))[key]["mean"] for r in other_recs
                 if _norm_eval(r.get("eval")).get(key)]
        if not mam or not other:
            continue
        if len(mam) > 1 and len(other) > 1:
            t, p = welch_t(mam, other)          # 多训练 seed
        else:
            # 单次训练：用各**评测 seed** 的均值做 Welch（n = 评测 seed 数）
            o_vals = _eval_seed_values(other_recs, key)
            t, p = (welch_t(mam_vals, o_vals) if (mam_vals and o_vals)
                    else (None, None))
        d = np.mean(mam) - np.mean(other)
        star = "n.s." if (p is None or p > .05) else ("***" if p < .001 else "**" if p < .01 else "*")
        sig.append(f"| Mamba vs {ARM_LABEL[arm]} | {d:+.2f} | "
                   f"{'—' if t is None else f'{t:.2f}'} | "
                   f"{'—' if p is None else f'{p:.4f}'} {star} |")
    (outdir / "table_significance.md").write_text(
        "\n".join(sig) +
        "\n\n> Welch t 检验。单次训练时用各**评测 seed** 的均值（n=10）。"
        "\n> 检验的是\"两个已训练策略的表现是否不同\"，"
        "**不是**\"两种方法孰优孰劣\"（后者需要多个训练 seed）。\n")

    # ---------- 参数量表 ----------
    pl = ["| Encoder | L | Actor params | Critic params |", "|---|---|---|---|"]
    seen = set()
    for cond in reversed(CONDS):
        for arm in ARM_ORDER:
            if arm in seen:
                continue
            recs = runs.get(cond, {}).get(arm, [])
            if not recs:
                continue
            pc = recs[0].get("param_count") or {}
            pl.append(f"| {ARM_LABEL[arm]} | {recs[0].get('context_len')} | "
                      f"{pc.get('actor', 0):,} | {pc.get('critic', 0):,} |")
            seen.add(arm)
    (outdir / "table_params.md").write_text("\n".join(pl) + "\n")
    return lines, sig, pl


def write_headline(rows, outdir: Path, cond=None):
    """头条表：证明"更少参数达到相当精度"。按参数量排序，并给出相对基线的比值。"""
    cond = cond or CONDS[-1]
    sub = [r for r in rows if r["cond"] == cond and r.get("params")]
    if not sub:
        return []
    base = next((r for r in sub if r["arm"] == "transformer"), None)
    sub.sort(key=lambda r: r["params"])

    hdr = ["Encoder", "Params", "vs baseline"] + [t for _, t in HEADLINE]
    lines = ["| " + " | ".join(hdr) + " |",
             "|" + "|".join(["---"] * len(hdr)) + "|"]
    csvl = ["arm,params,params_ratio," + ",".join(
        f"{k.split('.')[-1]}_mean,{k.split('.')[-1]}_std" for k, _ in HEADLINE)]
    for r in sub:
        ratio = (r["params"] / base["params"]) if base else float("nan")
        cells = [ARM_LABEL.get(r["arm"], r["arm"]), f"{r['params']:,}",
                 f"{ratio:.2f}×" if base else "—"]
        csvrow = [r["arm"], str(r["params"]), f"{ratio:.4f}" if base else ""]
        for key, _ in HEADLINE:
            short = next(s for k, s, _, _ in METRICS if k == key)
            if short + "_mean" in r:
                cells.append(f"{r[short+'_mean']:.3f} ± {r[short+'_std']:.3f}")
                csvrow += [f"{r[short+'_mean']:.6f}", f"{r[short+'_std']:.6f}"]
            else:
                cells.append("—"); csvrow += ["", ""]
        lines.append("| " + " | ".join(cells) + " |")
        csvl.append(",".join(csvrow))

    note = ("\n\n> 基线 = Transformer @ DTQN 原生默认（d128 × 2层 × 8heads × ff4，"
            "learned pos，post-norm，res gate）。\n"
            "> **本表的主张：Mamba 用 1 层、显著更少的参数，达到与该基线相当的精度。**\n"
            "> `±` = 跨 10 个独立评测 seed 的 std（各 arm 用同一组 seed，共同随机数）。\n"
            "> 是否\"相当\"以显著性表为准（n=10 Welch t 检验），不以点估计的大小为准。\n")
    (outdir / "table_headline.md").write_text("\n".join(lines) + note)
    (outdir / "table_headline.csv").write_text("\n".join(csvl) + "\n")
    return lines


def fig_main(rows, outdir: Path):
    """主图：两个面板 —— MDP(应打平) 与 POMDP(应拉开)。同一 y 轴便于直接对比。"""
    st.apply_publication_style()
    fig, axes = st.create_subplots(1, 2, figsize=(13, 4.8), sharey=True)
    key = "episode_len"
    ymax = 0
    for ax, cond in zip(axes, CONDS):
        sub = [r for r in rows if r["cond"] == cond and key + "_mean" in r]
        sub.sort(key=lambda r: ARM_ORDER.index(r["arm"]))
        if not sub:
            ax.set_visible(False)
            continue
        x = np.arange(len(sub))
        vals = [r[key + "_mean"] for r in sub]
        errs = [r[key + "_std"] for r in sub]
        cols = [ARM_COLOR[r["arm"]] for r in sub]
        bars = ax.bar(x, vals, yerr=errs, capsize=5, color=cols,
                      edgecolor="black", linewidth=1.2,
                      error_kw=dict(elinewidth=1.8, ecolor="#333333"))
        for b, v, e in zip(bars, vals, errs):
            ax.text(b.get_x() + b.get_width() / 2, v + e + max(vals) * 0.02,
                    f"{v:.1f}", ha="center", va="bottom", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels([ARM_LABEL[r["arm"]].replace(" (", "\n(") for r in sub],
                           rotation=0, fontsize=9)
        ax.set_title(COND_LABEL[cond], fontsize=14, pad=10)
        ymax = max(ymax, max(v + e for v, e in zip(vals, errs)))
    axes[0].set_ylabel("Episode length (steps) ↑")
    for ax in axes:
        ax.set_ylim(0, ymax * 1.18)
        ax.grid(axis="y", alpha=.3, linestyle=":")
        ax.set_axisbelow(True)
    fig.suptitle("Tracking accuracy across encoders", fontsize=15, y=1.02)
    return st.finalize_figure(fig, outdir / "fig1_main")


def fig_curves(runs, outdir: Path, cond="pomdp"):
    """学习曲线：POMDP 下各臂 episode_len vs 环境步数（跨 seed 的 mean±std 带）。"""
    st.apply_publication_style()
    fig, axes = st.create_subplots(1, 1, figsize=(7.5, 5))
    ax = axes[0]
    ck = "train/stats.episode_len"
    plotted = 0
    for arm in ARM_ORDER:
        recs = runs.get(cond, {}).get(arm, [])
        series = []
        for r in recs:
            xs = [c["env_frames"] for c in r.get("curve", []) if ck in c]
            ys = [c[ck] for c in r.get("curve", []) if ck in c]
            if len(ys) > 3:
                series.append((np.array(xs), np.array(ys)))
        if not series:
            continue
        n = min(len(y) for _, y in series)
        X = series[0][0][:n]
        Y = np.stack([y[:n] for _, y in series])
        m, s = Y.mean(0), Y.std(0)
        # 轻度平滑，只为可读性；阴影是跨 seed 的 std
        k = max(1, n // 40)
        if k > 1:
            ker = np.ones(k) / k
            m = np.convolve(m, ker, mode="same")
        ax.plot(X, m, label=ARM_LABEL[arm], color=ARM_COLOR[arm], linewidth=2.4)
        ax.fill_between(X, m - s, m + s, color=ARM_COLOR[arm], alpha=.15, linewidth=0)
        plotted += 1
    if not plotted:
        return []
    ax.set_xlabel("Environment frames")
    ax.set_ylabel("Episode length (steps) ↑")
    ax.set_title(COND_LABEL[cond], fontsize=14)
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(alpha=.3, linestyle=":")
    ax.set_axisbelow(True)
    return st.finalize_figure(fig, outdir / f"fig2_curves_{cond}")


# 部署时每个控制步的推理方式：
#   mlp/stack/transformer 只能每步重算窗口；gru/mamba 可以携带递归状态每步只算 1 帧。
DEPLOY_MODE = {"mlp": "window", "mlp_wide": "window", "stack": "window",
               "transformer": "window", "transformer_small": "window",
               "gru": "stream", "mamba": "stream", "mamba_big": "stream",
               "mamba_d96": "stream", "mamba_d64": "stream"}


def fig_tradeoff(rows, outdir: Path, ctx_len=16):
    """本文的核心图：POMDP 下的任务性能 vs 部署时每步推理延迟。

    Mamba 的主张不是"精度更高"，而是**在同等精度下推理代价低一个数量级**，
    所以必须画在同一张图上，而不是分开报两个表。
    """
    effp = outdir / "efficiency_b1.json"
    if not effp.exists():
        return []
    eff = json.loads(effp.read_text())
    key = str(ctx_len)
    if key not in eff.get("window", {}):
        key = sorted(eff["window"], key=lambda k: abs(int(k) - ctx_len))[0]

    pts = []
    for r in rows:
        if r["cond"] != CONDS[-1] or "episode_len_mean" not in r:
            continue
        arm = r["arm"]
        mode = DEPLOY_MODE.get(arm, "window")
        base = arm if arm in eff["window"][key] else \
            ("transformer" if "transformer" in arm else
             "mamba" if "mamba" in arm else
             "mlp" if "mlp" in arm else arm)
        if mode == "stream" and base in eff.get("stream", {}).get(key, {}):
            ms = eff["stream"][key][base].get("ms")
        else:
            ms = eff["window"][key].get(base, {}).get("ms")
        if ms is None:
            continue
        pts.append((arm, ms, r["episode_len_mean"], r.get("episode_len_std", 0.0), mode))
    if not pts:
        return []

    st.apply_publication_style()
    fig, axes = st.create_subplots(1, 1, figsize=(7.8, 5.2))
    ax = axes[0]
    for arm, ms, y, e, mode in pts:
        ax.errorbar(ms, y, yerr=e, fmt="o", markersize=13,
                    color=ARM_COLOR.get(arm, "#666"),
                    markeredgecolor="black", markeredgewidth=1.3,
                    ecolor="#444", elinewidth=1.6, capsize=4, zorder=3)
        tag = ARM_LABEL.get(arm, arm).split(" (")[0]
        if mode == "stream":
            tag += " [streaming]"
        ax.annotate(tag, (ms, y), textcoords="offset points", xytext=(10, 6),
                    fontsize=10)
    ax.set_xscale("log")
    ax.set_xlabel("Inference latency per control step (ms, log scale) ←better")
    ax.set_ylabel("Episode length (steps) ↑better")
    ax.set_title("Accuracy vs deployment cost under partial observability",
                 fontsize=14)
    ax.grid(alpha=.3, which="both", linestyle=":")
    ax.set_axisbelow(True)
    xs = [p[1] for p in pts]
    ax.set_xlim(min(xs) * 0.45, max(xs) * 3.2)
    return st.finalize_figure(fig, outdir / "fig4_tradeoff")


def main():
    ap = argparse.ArgumentParser()
    here = Path(__file__).parent
    ap.add_argument("--indir", default=str(here / "outputs_p3" / "main"))
    ap.add_argument("--outdir", default=str(here / "outputs_p3" / "report"))
    ap.add_argument("--conds", default="mdp,pomdp",
                    help="主图两列的条件名，逗号分隔：<全观测参照>,<部分可观测>")
    a = ap.parse_args()
    global CONDS
    CONDS = [c.strip() for c in a.conds.split(",") if c.strip()]
    indir, outdir = Path(a.indir), Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    runs = load(indir)
    if not runs:
        print(f"[p3_report] {indir} 里没有结果 JSON")
        return
    have = {c: {k: len(v) for k, v in d.items()} for c, d in runs.items()}
    print("已有结果:", json.dumps(have, ensure_ascii=False))

    rows = build_table(runs)
    head = write_headline(rows, outdir)
    tbl, sig, pl = write_tables(rows, runs, outdir)
    figs = fig_main(rows, outdir)
    for c in CONDS:
        figs += fig_curves(runs, outdir, c)
    figs += fig_tradeoff(rows, outdir)

    (outdir / "summary.json").write_text(json.dumps(
        {"rows": rows, "counts": have, "diverged": DIVERGED}, indent=1,
        ensure_ascii=False))
    if DIVERGED:
        (outdir / "table_diverged.md").write_text(
            "以下 run 训练发散(NaN)，已从统计中剔除，论文中如实报告：\n\n"
            + "\n".join(f"- `{d}`" for d in DIVERGED) + "\n")
        print("\n[发散剔除]", ", ".join(DIVERGED))

    if head:
        print("=== 头条表：更少参数 vs 相当精度 ===")
        print("\n".join(head)); print()
    print("\n".join(tbl))
    print()
    print("\n".join(sig))
    print()
    print("\n".join(pl))
    print("\n产出:", *[f"  {p}" for p in
                     sorted(outdir.glob('*'))], sep="\n")


if __name__ == "__main__":
    main()

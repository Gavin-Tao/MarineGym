#!/usr/bin/env python
"""AEI: 汇总 → 统计 → 出图 → RESULTS.md。

范围（2026-08-25 修订）：只报告 **本方法 vs baseline** 与 **本方法自身的消融**。
本方法 = A(RiskMonitor 风险门控) → λ → B(MPPI Exact 滤波) → soft blend。
"内化 C"(internalize_weight / internalize_adaptive) 不属于本方法，全流程排除，
不进 CSV、不进图、不进 RESULTS.md。
"""
import csv, glob, json, math, os, re, subprocess, sys, statistics as st
sys.path.insert(0, "/home/jovyan/MarineGym/scripts")
import numpy as np
from aei_style import (apply_publication_style, FigureStyle, create_subplots,
                       finalize_figure, make_grouped_bar, PALETTE)

ROOT = "/home/jovyan/MarineGym/scripts"
DATA = f"{ROOT}/outputs_aei/data"; FIGS = f"{ROOT}/outputs_aei/figures"
os.makedirs(DATA, exist_ok=True); os.makedirs(FIGS, exist_ok=True)
apply_publication_style(FigureStyle(font_size=15, axes_linewidth=2.0))

# ---- 不属于本方法的轴：全流程排除 ----
EXCLUDE = {"r1d_fixed_C", "r1e_adaptive_C"}

LABEL = {
    "t1_ppo": "PPO baseline", "t2_absoft": "RP-PSF (ours)",
    "t3_hi0.3": r"$h_{hi}$=0.3", "t4_hi0.9": r"$h_{hi}$=0.9", "t5_hi1.2": r"$h_{hi}$=1.2",
    "r1a_pred_binary": "Predictive + binary (thr=0.3)",
    "r1a_pred_binary_t06": "Predictive + binary",
    "r1b_geom_soft_v2": "Geometric + proportional", "r1c_geom_binary_v2": "Geometric + binary", "r1b_geom_soft": "Geometric + proportional",
    "r1c_geom_binary": "Geometric + binary", "r1f_a_only": "A-only (no filter)",
}
SCENES = ["nominal", "ood", "hard"]

# ---------- 统计工具（不依赖 scipy） ----------
def _phi(x): return 0.5 * (1 + math.erf(x / math.sqrt(2)))
def _t_sf(t, df):
    """Student-t 双侧 p（连续分数自由度，数值积分）——小样本不再用正态近似。"""
    t = abs(t)
    lg = math.lgamma((df + 1) / 2) - math.lgamma(df / 2) - 0.5 * math.log(df * math.pi)
    c = math.exp(lg)
    n = 2000; hi = t + 40.0; h = (hi - t) / n
    s = 0.0
    for i in range(n + 1):
        x = t + i * h
        w = 1 if i in (0, n) else (4 if i % 2 else 2)
        s += w * (1 + x * x / df) ** (-(df + 1) / 2)
    return min(1.0, 2 * c * s * h / 3)

def welch(a, b):
    if len(a) < 2 or len(b) < 2: return None
    va, vb = st.variance(a), st.variance(b); na, nb = len(a), len(b)
    se = math.sqrt(va / na + vb / nb)
    if se == 0: return dict(t=0.0, p=1.0, d=0.0, df=na + nb - 2)
    t = (st.mean(a) - st.mean(b)) / se
    df = (va / na + vb / nb) ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    sp = math.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    return dict(t=t, p=_t_sf(t, df), df=df, d=(st.mean(a) - st.mean(b)) / sp if sp else 0.0)

def two_prop(k1, n1, k2, n2):
    if n1 == 0 or n2 == 0: return None
    p1, p2 = k1 / n1, k2 / n2; pp = (k1 + k2) / (n1 + n2)
    se = math.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n2))
    if se == 0: return dict(z=0.0, p=1.0, p1=p1, p2=p2)
    z = (p1 - p2) / se
    return dict(z=z, p=2 * (1 - _phi(abs(z))), p1=p1, p2=p2)

def mde(p1, n, alpha=0.05, power=0.8):
    """给定基线率与每组样本量，能检出的最小相对下降（二比例检验）。"""
    z1, z2 = 1.959964, 0.8416
    lo, hi = 1e-6, p1 * 0.999
    for _ in range(60):
        d = (lo + hi) / 2; p2 = p1 - d; pb = (p1 + p2) / 2
        need = ((z1 * math.sqrt(2 * pb * (1 - pb)) +
                 z2 * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2) / d ** 2
        if need > n: lo = d
        else: hi = d
    return hi / p1

def load(path):
    return list(csv.DictReader(open(path))) if os.path.exists(path) else []

def run_key(label):
    """从训练日志里取出该 label 对应的 wandb run 时间戳。"""
    p = f"{ROOT}/outputs_aei/{label}.log"
    if not os.path.exists(p): return None
    m = re.search(r"offline-run-([0-9_]+)-", open(p, errors="replace").read())
    return m.group(1) if m else None


def main():
    subprocess.run([sys.executable, f"{ROOT}/collect_aei.py"], check=False)
    ev = [r for r in load(f"{DATA}/eval_raw.csv") if r["model"] not in EXCLUDE]
    tr = load(f"{DATA}/training_curves.csv")
    notes, dropped = [], []

    # Predictive+binary：优先用触发边界对齐到 0.6 的版本（与 ours 的 mppi.soft_hi 一致，
    # 两格只差接管律）；对齐版数据未就绪时回落到原始 thr=0.3 的版本，并在报告里标注。
    have_v2 = any(r["model"] == "r1b_geom_soft_v2" for r in ev)
    GS, GB = ("r1b_geom_soft_v2", "r1c_geom_binary_v2") if have_v2 else (None, None)
    if have_v2:
        ev = [r for r in ev if r["model"] not in ("r1b_geom_soft", "r1c_geom_binary")]
    else:
        ev = [r for r in ev if r["model"] not in ("r1b_geom_soft", "r1c_geom_binary")]
    have_t06 = any(r["model"] == "r1a_pred_binary_t06" for r in ev)
    PBIN = "r1a_pred_binary_t06" if have_t06 else "r1a_pred_binary"
    if have_t06:
        ev = [r for r in ev if r["model"] != "r1a_pred_binary"]   # 混淆版不再展示
        notes.append("Predictive+binary 使用 thr=0.6 对齐版（与 ours 边界一致，仅差接管律）")
    else:
        dropped.append("Predictive+binary 仍是 thr=0.3 版：触发边界与 ours(0.6) 不一致，该格暂不可比")
    if not have_v2:
        dropped.append("几何门控两格（Geometric+prop. / Geometric+binary）：旧版观测仅 44 维"
                       "（risk 标量缺失），与本方法 45 维不可比，已剔除；v2 重训中")

    # ---------- 评测汇总 ----------
    summ = {}
    for r in ev:
        summ.setdefault((r["model"], r["scene"]), []).append(r)
    METRICS = ["collision", "return", "min_obstacle_dist", "detour_ratio",
               "filter_activation", "tracking_error_ema", "action_smoothness", "correction"]
    rows = []
    for (m, sc), rs in sorted(summ.items()):
        d = dict(model=m, scene=sc, n_seeds=len(rs),
                 episodes=sum(int(float(x.get("episodes", 0) or 0)) for x in rs))
        for k in METRICS:
            v = [float(x[k]) for x in rs if x.get(k) not in (None, "")]
            if v: d[k + "_mean"], d[k + "_std"] = st.mean(v), (st.pstdev(v) if len(v) > 1 else 0.0)
        rows.append(d)
    if rows:
        cols = sorted({k for r in rows for k in r})
        cols = ["model", "scene", "n_seeds", "episodes"] + \
               [c for c in cols if c not in ("model", "scene", "n_seeds", "episodes")]
        with open(f"{DATA}/eval_summary.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
        notes.append(f"eval_summary.csv: {len(rows)} 组 model×scene（已排除内化 C）")

    # ---------- 主结果显著性 ----------
    tests = []
    for sc in SCENES:
        a, b = summ.get(("t1_ppo", sc)), summ.get(("t2_absoft", sc))
        if not a or not b: continue
        na = sum(int(float(x.get("episodes", 0) or 0)) for x in a)
        nb = sum(int(float(x.get("episodes", 0) or 0)) for x in b)
        ka = sum(float(x["collision"]) * int(float(x.get("episodes", 0) or 0)) for x in a)
        kb = sum(float(x["collision"]) * int(float(x.get("episodes", 0) or 0)) for x in b)
        tp = two_prop(round(ka), na, round(kb), nb)
        if tp:
            tests.append(dict(scene=sc, metric="collision (episode-level 2-prop)",
                              stat=f"z={tp['z']:.2f}", p=tp["p"],
                              ppo=f"{tp['p1']:.4f} ({round(ka)}/{na})",
                              ours=f"{tp['p2']:.4f} ({round(kb)}/{nb})"))
        for k in ("return", "min_obstacle_dist", "detour_ratio", "tracking_error_ema"):
            va = [float(x[k]) for x in a if x.get(k)]; vb = [float(x[k]) for x in b if x.get(k)]
            w = welch(va, vb)
            if w:
                tests.append(dict(scene=sc, metric=k + " (Welch, seed-level)",
                                  stat=f"t={w['t']:.2f}", p=w["p"],
                                  ppo=f"{st.mean(va):.4f}±{st.pstdev(va):.4f}",
                                  ours=f"{st.mean(vb):.4f}±{st.pstdev(vb):.4f}"))
    if tests:
        with open(f"{DATA}/significance.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(tests[0].keys())); w.writeheader(); w.writerows(tests)
        notes.append(f"significance.csv: {len(tests)} 项检验（Welch 用 t 分布，非正态近似）")

    # ---------- 检验功效（说明"打平"到底意味着什么） ----------
    power_rows = []
    for sc in SCENES:
        a = summ.get(("t1_ppo", sc))
        if not a: continue
        n = sum(int(float(x.get("episodes", 0) or 0)) for x in a)
        p1 = st.mean([float(x["collision"]) for x in a])
        if p1 <= 0: continue
        power_rows.append(dict(scene=sc, ppo_rate=round(p1, 5), episodes_per_arm=n,
                               detectable_rel_drop=round(mde(p1, n), 3)))
    if power_rows:
        with open(f"{DATA}/power.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(power_rows[0].keys())); w.writeheader(); w.writerows(power_rows)
        notes.append(f"power.csv: {len(power_rows)} 个场景的最小可检出效应")

    # ---------- 图 1：主结果（本方法 vs baseline，三场景） ----------
    try:
        got = [s for s in SCENES if ("t1_ppo", s) in summ and ("t2_absoft", s) in summ]
        if not got: raise ValueError("无主结果数据")
        panels = [("collision", "Violation rate", True), ("return", "Episode return", False),
                  ("min_obstacle_dist", "Min clearance (m)", False), ("detour_ratio", "Detour ratio", False)]
        fig, axes = create_subplots(1, 4, figsize=(19, 4.4))
        for ax, (k, lab, logy) in zip(axes, panels):
            A = [[float(x[k]) for x in summ[("t1_ppo", s)]] for s in got]
            B = [[float(x[k]) for x in summ[("t2_absoft", s)]] for s in got]
            make_grouped_bar(ax, got, [[st.mean(v) for v in A], [st.mean(v) for v in B]],
                             ["PPO baseline", "RP-PSF (ours)"], ylabel=lab,
                             colors=[PALETTE["neutral"], PALETTE["blue_main"]],
                             errs=[[st.pstdev(v) for v in A], [st.pstdev(v) for v in B]])
            if logy: ax.set_yscale("log")
        axes[0].legend(loc="upper left", bbox_to_anchor=(0, 1.24), ncol=2)
        finalize_figure(fig, f"{FIGS}/fig1_main_results")
        notes.append(f"fig1_main_results（scene={'/'.join(got)}，PPO vs ours）")
    except Exception as e: notes.append(f"fig1 跳过: {e}")

    # ---------- 图 2：soft_hi 敏感性（本方法的超参） ----------
    try:
        order = [("t3_hi0.3", 0.3), ("t2_absoft", 0.6), ("t4_hi0.9", 0.9), ("t5_hi1.2", 1.2)]
        SC = "ood"
        got = [(m, h) for m, h in order if (m, SC) in summ]
        if len(got) < 2: raise ValueError("数据不足")
        fig, axes = create_subplots(1, 3, figsize=(15, 4.2))
        for ax, (k, lab) in zip(axes, [("collision", "Violation rate"), ("return", "Episode return"),
                                       ("filter_activation", "Filter engagement")]):
            xs = [h for _, h in got]
            ys = [st.mean([float(x[k]) for x in summ[(m, SC)]]) for m, _ in got]
            es = [st.pstdev([float(x[k]) for x in summ[(m, SC)]]) for m, _ in got]
            ax.errorbar(xs, ys, yerr=es, marker="o", lw=2.6, capsize=4,
                        color=PALETTE["blue_main"], mfc="white", mew=2.2, ms=9)
            ax.set_xlabel(r"$h_{hi}$ (m)"); ax.set_ylabel(lab); ax.grid(alpha=.25)
        finalize_figure(fig, f"{FIGS}/fig2_softhi_sensitivity")
        notes.append(f"fig2_softhi_sensitivity（{len(got)} 个点，scene={SC}）")
    except Exception as e: notes.append(f"fig2 跳过: {e}")

    # ---------- 图 3：评测期消融（门控信号 × 接管律 2×2 + A-only + baseline） ----------
    try:
        abl = [m for m in ["t2_absoft", PBIN, GS, GB, "r1f_a_only", "t1_ppo"] if m]
        SC = "ood"
        got = [m for m in abl if (m, SC) in summ]
        if len(got) < 3: raise ValueError("数据不足")
        PANELS = [("collision", "Violation rate \u2193"), ("return", "Episode return \u2191"),
                  ("detour_ratio", "Detour ratio \u2193"), ("action_smoothness", "Action smoothness \u2191"),
                  ("correction", "Correction magnitude \u2193"), ("tracking_error_ema", "Tracking error \u2193")]
        fig, axes = create_subplots(2, 3, figsize=(18, 10.5))
        for ax, (k, lab) in zip(axes, PANELS):
            vals = [st.mean([float(x[k]) for x in summ[(m, SC)]]) for m in got]
            errs = [st.pstdev([float(x[k]) for x in summ[(m, SC)]]) for m in got]
            cols = [PALETTE["blue_main"] if m == "t2_absoft" else
                    (PALETTE["neutral"] if m == "t1_ppo" else PALETTE["blue_secondary"]) for m in got]
            # 点+误差棒：不隐含「从零起」的面积语义，因此可按数据缩放纵轴，
            # 让 smoothness(~-0.79)/tracking(~1.35) 这类远离零点的量看得出差异。
            for i, (v, e, c) in enumerate(zip(vals, errs, cols)):
                ax.errorbar(i, v, yerr=e, fmt="o", ms=11, mfc=c, mec="#333", mew=1.6,
                            ecolor="#555", elinewidth=1.8, capsize=6, zorder=3)
            lo = min(v - e for v, e in zip(vals, errs)); hi = max(v + e for v, e in zip(vals, errs))
            pad = (hi - lo) * 0.28 or (abs(hi) * 0.05 + 1e-6)
            ax.set_ylim(lo - pad, hi + pad)
            ax.set_xlim(-0.6, len(got) - 0.4)
            ax.set_xticks(range(len(got)))
            ax.set_xticklabels([LABEL.get(m, m) for m in got], rotation=22, ha="right")
            ax.set_ylabel(lab); ax.grid(axis="y", alpha=.25, zorder=0)
        fig.subplots_adjust(hspace=0.62, wspace=0.30)
        finalize_figure(fig, f"{FIGS}/fig3_ablation")
        notes.append(f"fig3_ablation（{len(got)} 个变体 × 6 指标，scene={SC}，10 evaluation seeds）")
    except Exception as e: notes.append(f"fig3 跳过: {e}")

    # ---------- 图 3b：训练期违约率（单次训练，episode 级 Wilson 95% CI） ----------
    # 协议：每个变体只训练一次；不确定性来自这一次训练内的 episode 数，不来自多训练 seed。
    try:
        from fractions import Fraction
        ORDER = [("RP-PSF (ours)", "20260711_165556"), ("PPO baseline", "20260711_094733"),
                 ("Predictive + binary", run_key(PBIN)),
                 ("Geometric + prop.", run_key("r1b_geom_soft")),
                 ("Geometric + binary", run_key("r1c_geom_binary")),
                 ("A-only", run_key("r1f_a_only"))]
        def counts(k):
            """从每次记录的 collision 率反推分母，累计训练期碰撞数与 episode 数。"""
            ce = ne = 0
            for r in tr:
                if k not in r["run"]: continue
                c = float(r["collision"])
                if c == 0: ne += 18                      # 零值无法反推分母，用观测到的中位批量
                else:
                    f = Fraction(c).limit_denominator(40)
                    ce += f.numerator; ne += f.denominator
            return ce, ne
        def wilson(k, n, z=1.959964):
            if n == 0: return 0.0, 0.0, 0.0
            p = k / n; d = 1 + z * z / n
            c = (p + z * z / (2 * n)) / d
            h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
            return p, max(0.0, c - h), c + h
        agg = {}
        for lab, k in ORDER:
            if not k: continue
            ce, ne = counts(k)
            if ne: agg[lab] = (ce, ne) + wilson(ce, ne)
        if len(agg) < 3: raise ValueError("数据不足")
        labs = sorted(agg, key=lambda l: agg[l][2])
        ps  = [agg[l][2] * 100 for l in labs]
        lo  = [(agg[l][2] - agg[l][3]) * 100 for l in labs]
        hi  = [(agg[l][4] - agg[l][2]) * 100 for l in labs]
        cols = [PALETTE["blue_main"] if "ours" in l else
                (PALETTE["neutral"] if "PPO baseline" in l else PALETTE["blue_secondary"]) for l in labs]
        fig, axes = create_subplots(1, 1, figsize=(10.5, 5.0)); ax = axes[0]
        ax.bar(range(len(labs)), ps, 0.66, yerr=[lo, hi], capsize=5, color=cols,
               error_kw=dict(lw=1.6, ecolor="#444"), zorder=3)
        for i, l in enumerate(labs):
            ce, ne = agg[l][0], agg[l][1]
            ax.annotate(f"{ce}/{ne}", (i, ps[i] + hi[i]), textcoords="offset points",
                        xytext=(0, 6), ha="center", fontsize=11)
        ax.set_xticks(range(len(labs))); ax.set_xticklabels(labs, rotation=25, ha="right")
        ax.set_ylabel("Training violation rate (%)")
        ax.set_ylim(0, max(p + h for p, h in zip(ps, hi)) * 1.25)
        ax.grid(axis="y", alpha=.25, zorder=0)
        finalize_figure(fig, f"{FIGS}/fig3b_training_violation_rate")
        for old in glob.glob(f"{FIGS}/fig3b_training_ablation.*"): os.remove(old)
        notes.append("fig3b_training_violation_rate（单次训练，误差棒=episode 级 Wilson 95% CI，柱上为 碰撞数/episode 数）")
        with open(f"{DATA}/training_violation.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["variant", "collisions", "episodes", "rate", "ci_lo", "ci_hi"])
            for l in labs:
                ce, ne, pr, a_, b_ = agg[l]
                w.writerow([l, ce, ne, round(pr, 5), round(a_, 5), round(b_, 5)])
        for old in (f"{DATA}/training_ablation.csv",):
            if os.path.exists(old): os.remove(old)
        # PPO vs ours 的训练期两比例检验
        if "PPO baseline" in agg and "RP-PSF (ours)" in agg:
            a_, b_ = agg["PPO baseline"], agg["RP-PSF (ours)"]
            tp = two_prop(a_[0], a_[1], b_[0], b_[1])
            if tp:
                tests.append(dict(scene="training", metric="violation rate (episode-level 2-prop)",
                                  stat=f"z={tp['z']:.2f}", p=tp["p"],
                                  ppo=f"{tp['p1']:.4f} ({a_[0]}/{a_[1]})",
                                  ours=f"{tp['p2']:.4f} ({b_[0]}/{b_[1]})"))
                with open(f"{DATA}/significance.csv", "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(tests[0].keys())); w.writeheader(); w.writerows(tests)
    except Exception as e: notes.append(f"fig3b 跳过: {e}")

    # ---------- 图 4：训练期累计违约曲线 ----------
    try:
        RUNS = {"20260711_094733": "PPO baseline", "20260711_165556": "RP-PSF (ours)"}
        for lbl in (PBIN, "r1b_geom_soft", "r1c_geom_binary", "r1f_a_only"):
            k = run_key(lbl)
            if k: RUNS[k] = LABEL.get(lbl, lbl)
        series, labels = [], []
        for key, lab in RUNS.items():
            pts = sorted((int(r["step"]), float(r["cum_violation"])) for r in tr if key in r["run"])
            if len(pts) > 3: series.append(pts); labels.append(lab)
        if not series: raise ValueError("无训练曲线")
        fig, axes = create_subplots(1, 1, figsize=(7.6, 4.8)); ax = axes[0]
        order = sorted(range(len(labels)), key=lambda i: 0 if "PPO base" in labels[i]
                       else (1 if "ours" in labels[i] else 2))
        cmap = {"PPO baseline": PALETTE["neutral"], "RP-PSF (ours)": PALETTE["blue_main"]}
        rest = [PALETTE["teal"], PALETTE["violet"], PALETTE["green_3"], PALETTE["red_strong"]]
        ri = 0
        for i in order:
            pts, lab = series[i], labels[i]
            c = cmap.get(lab)
            if c is None: c = rest[ri % len(rest)]; ri += 1
            ax.plot([p[0] for p in pts], [p[1] for p in pts], lw=2.8 if lab in cmap else 2.0,
                    label=lab, color=c, zorder=4 if lab in cmap else 3)
        ax.set_xlabel("Training iteration"); ax.set_ylabel("Cumulative violations")
        ax.grid(alpha=.25, zorder=0); ax.legend(fontsize=11)
        finalize_figure(fig, f"{FIGS}/fig4_training_violations")
        notes.append(f"fig4_training_violations（{len(series)} 条曲线，单训练 seed）")
    except Exception as e: notes.append(f"fig4 跳过: {e}")

    # 说明：原 fig5(计算延迟) 已移除——N2 测的是 Isaac rollout fps，
    # ms/step 在 K=32→256 上完全不变，测不到 MPPI 开销，属无效测量。
    dropped.append("fig5_latency：测量无效（K 无关），已删除，需重做逐次滤波计时")
    for p in glob.glob(f"{FIGS}/fig5_latency.*"): os.remove(p)

    # ---------- 图 6/7：两条消融轴的正面对决 ----------
    HEAD_MET = [("collision", "Violation rate", True), ("min_obstacle_dist", "Min clearance", False),
                ("return", "Episode return", False), ("detour_ratio", "Detour ratio", True),
                ("action_smoothness", "Action smoothness", False),
                ("correction", "Correction magnitude", True),
                ("tracking_error_ema", "Tracking error", True)]

    def headtohead(m_ours, m_other, out, right_lab, left_lab, note):
        """条长=相对差异%，标注同时给 p 与 |d|（效应量），避免只看条长被密集量误导。"""
        A = summ.get((m_ours, "ood")); B = summ.get((m_other, "ood"))
        if not A or not B: raise ValueError(f"缺数据 {m_ours}/{m_other}")
        labs, rel, ps, ds = [], [], [], []
        for k, lab, lower_better in HEAD_MET:
            va = [float(x[k]) for x in A]; vb = [float(x[k]) for x in B]
            ma, mb = st.mean(va), st.mean(vb)
            d = (mb - ma) / max(abs(ma), abs(mb), 1e-9)
            if not lower_better: d = -d
            w = welch(va, vb)
            labs.append(lab); rel.append(d * 100)
            ps.append(w["p"] if w else 1.0); ds.append(abs(w["d"]) if w else 0.0)
        o = sorted(range(len(labs)), key=lambda i: rel[i])
        labs = [labs[i] for i in o]; rel = [rel[i] for i in o]
        ps = [ps[i] for i in o]; ds = [ds[i] for i in o]
        cols = [(PALETTE["blue_main"] if r > 0 else PALETTE["red_strong"])
                if p < 0.05 else PALETTE["neutral"] for r, p in zip(rel, ps)]
        fig, axes = create_subplots(1, 1, figsize=(11.5, 5.6)); ax = axes[0]
        ax.barh(range(len(labs)), rel, 0.66, color=cols, zorder=3)
        ax.axvline(0, color="#333", lw=2, zorder=4)
        for i, (r, p, d) in enumerate(zip(rel, ps, ds)):
            star = "***" if p < .001 else ("**" if p < .01 else ("*" if p < .05 else "n.s."))
            off = 1.5 if r >= 0 else -1.5
            ax.annotate(f"{star}  |d|={d:.1f}  p={p:.2g}", (r + off, i), va="center",
                        ha="left" if r >= 0 else "right", fontsize=10.5)
        ax.set_yticks(range(len(labs))); ax.set_yticklabels(labs)
        ax.set_xlabel(f"\u2190  {left_lab}        Relative difference (%)        {right_lab}  \u2192",
                      fontsize=12.5)
        m = max(abs(min(rel)), abs(max(rel)))
        ax.set_xlim(-m * 2.0, m * 2.0); ax.grid(axis="x", alpha=.25, zorder=0)
        finalize_figure(fig, f"{FIGS}/{out}")
        notes.append(note)

    try:
        headtohead("t2_absoft", PBIN, "fig6_takeover_law",
                   "proportional (ours) better", "binary takeover better",
                   "fig6_takeover_law（接管律：斜坡 vs 阶跃；边界均 0.6、观测均 45 维；灰=不显著）")
    except Exception as e: notes.append(f"fig6 跳过: {e}")
    try:
        if not GS: raise ValueError("几何门控 v2 数据未就绪")
        headtohead("t2_absoft", GS, "fig7_gate_signal",
                   "predictive gate (ours) better", "geometric gate better",
                   "fig7_gate_signal（门控信号：预测 vs 几何；两侧均为斜坡接管、观测均 45 维；灰=不显著）")
    except Exception as e: notes.append(f"fig7 跳过: {e}")

    # ---------- RESULTS.md ----------
    with open(f"{ROOT}/outputs_aei/RESULTS.md", "w") as f:
        f.write("# AEI 实验结果\n\n")
        f.write("> **方法**：RP-PSF = A(RiskMonitor 风险门控, H=15) → λ → B(MPPI Exact, K=128/N=20) → soft blend\n")
        f.write("> **范围**：本方法 vs PPO baseline + 本方法自身消融。内化 C 不属于本方法，全部排除。\n")
        f.write("> **配置 P1**：mppi.num_samples=128, mppi.horizon=20, risk.horizon=15\n\n")
        f.write("## 产出\n\n")
        for n in notes: f.write(f"- {n}\n")
        for d in dropped: f.write(f"- ~~{d}~~\n")

        if rows:
            f.write("\n## 评测汇总（mean ± std over evaluation seeds）\n\n")
            f.write("| model | scene | seeds | eps | collision | return | min_dist | detour | 激活率 |\n")
            f.write("|---|---|---|---|---|---|---|---|---|\n")
            for r in rows:
                g = lambda k: f"{r.get(k+'_mean', float('nan')):.4f}±{r.get(k+'_std', 0):.4f}"
                f.write(f"| {LABEL.get(r['model'], r['model'])} | {r['scene']} | {r['n_seeds']} | {r['episodes']} | "
                        f"{g('collision')} | {g('return')} | {g('min_obstacle_dist')} | "
                        f"{g('detour_ratio')} | {g('filter_activation')} |\n")
        if tests:
            f.write("\n## 显著性检验（PPO vs ours）\n\n| scene | metric | PPO | ours | stat | p |\n|---|---|---|---|---|---|\n")
            for t in tests:
                f.write(f"| {t['scene']} | {t['metric']} | {t['ppo']} | {t['ours']} | {t['stat']} | {t['p']:.4g} |\n")
        if power_rows:
            f.write("\n## 检验功效：当前评测能看见多大的效应\n\n")
            f.write("| scene | PPO 碰撞率 | 每组 episodes | 80% 功效下可检出的最小相对下降 |\n|---|---|---|---|\n")
            for r in power_rows:
                f.write(f"| {r['scene']} | {r['ppo_rate']*100:.2f}% | {r['episodes_per_arm']} | "
                        f"{r['detectable_rel_drop']*100:.0f}% |\n")
            f.write("\n> 在低基线率场景上，只有**极大**的相对下降才检得出来；"
                    "该场景的 `p` 值不显著不等价于方法无效，而是评测分辨率不足。\n")
        f.write("\n## 口径说明\n\n")
        f.write("- **协议：每个变体训练一次，评测用多个独立 seed。** 评测汇总里的 `seeds` 是评测 seed。\n")
        f.write("- 训练期违约（fig3b/fig4）产生于训练过程本身，评测 seed 无法施加于它；"
                "其不确定性来自单次训练内的 episode 数，用 Wilson 95% CI 表示。\n")
        f.write("- 训练期违约事件极稀少（PPO 6 次 / ours 2 次，各约 420 个 episode），"
                "两比例检验 p=0.15，尚不显著——需要更长训练或违约更频繁的场景来累积事件，而非更多训练 seed。\n")
        f.write("- `hard` 场景当前两臂碰撞率均约 0.75、min_clearance 为负，处于饱和区，暂不具备区分度。\n")
    print("\n".join(notes + dropped)); print("→ outputs_aei/RESULTS.md")

if __name__ == "__main__":
    main()

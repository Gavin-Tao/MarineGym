#!/usr/bin/env python
"""AEI: 汇总 → 统计 → 出图 → RESULTS.md。对缺失数据自动跳过，可反复运行。"""
import csv, glob, json, math, os, re, subprocess, sys, statistics as st
sys.path.insert(0, "/home/jovyan/MarineGym/scripts")
import numpy as np
from aei_style import (apply_publication_style, FigureStyle, create_subplots,
                       finalize_figure, make_grouped_bar, make_trend, PALETTE)

ROOT = "/home/jovyan/MarineGym/scripts"
DATA = f"{ROOT}/outputs_aei/data"; FIGS = f"{ROOT}/outputs_aei/figures"
os.makedirs(DATA, exist_ok=True); os.makedirs(FIGS, exist_ok=True)
apply_publication_style(FigureStyle(font_size=15, axes_linewidth=2.0))

LABEL = {  # 出图用的显示名
    "t1_ppo": "PPO baseline", "t2_absoft": "RP-PSF (ours)",
    "t3_hi0.3": r"$h_{hi}$=0.3", "t4_hi0.9": r"$h_{hi}$=0.9", "t5_hi1.2": r"$h_{hi}$=1.2",
    "r1a_pred_binary": "Predictive + binary", "r1b_geom_soft": "Geometric + proportional",
    "r1c_geom_binary": "Geometric + binary", "r1f_a_only": "A-only (no filter)",
    "r1d_fixed_C": "+ fixed C", "r1e_adaptive_C": "+ adaptive C",
}

# ---------- 统计工具（不依赖 scipy）----------
def _phi(x): return 0.5 * (1 + math.erf(x / math.sqrt(2)))
def welch(a, b):
    if len(a) < 2 or len(b) < 2: return None
    va, vb = st.variance(a), st.variance(b); na, nb = len(a), len(b)
    se = math.sqrt(va/na + vb/nb)
    if se == 0: return dict(t=0.0, p=1.0, d=0.0)
    t = (st.mean(a) - st.mean(b)) / se
    df = (va/na + vb/nb)**2 / ((va/na)**2/(na-1) + (vb/nb)**2/(nb-1))
    p = 2 * (1 - _phi(abs(t)))          # 正态近似，df>=8 时足够
    sp = math.sqrt(((na-1)*va + (nb-1)*vb) / (na+nb-2))
    return dict(t=t, p=p, df=df, d=(st.mean(a)-st.mean(b))/sp if sp else 0.0)
def two_prop(k1, n1, k2, n2):
    if n1 == 0 or n2 == 0: return None
    p1, p2 = k1/n1, k2/n2; pp = (k1+k2)/(n1+n2)
    se = math.sqrt(pp*(1-pp)*(1/n1 + 1/n2))
    if se == 0: return dict(z=0.0, p=1.0, p1=p1, p2=p2)
    z = (p1-p2)/se
    return dict(z=z, p=2*(1-_phi(abs(z))), p1=p1, p2=p2)

def load(path):
    return list(csv.DictReader(open(path))) if os.path.exists(path) else []

def main():
    subprocess.run([sys.executable, f"{ROOT}/collect_aei.py"], check=False)
    ev = load(f"{DATA}/eval_raw.csv")
    tr = load(f"{DATA}/training_curves.csv")
    notes = []

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
            if v: d[k+"_mean"], d[k+"_std"] = st.mean(v), (st.pstdev(v) if len(v) > 1 else 0.0)
        rows.append(d)
    if rows:
        cols = sorted({k for r in rows for k in r})
        cols = ["model","scene","n_seeds","episodes"] + [c for c in cols if c not in ("model","scene","n_seeds","episodes")]
        with open(f"{DATA}/eval_summary.csv","w",newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
        notes.append(f"eval_summary.csv: {len(rows)} 组 model×scene")

    # ---------- 主结果显著性 ----------
    tests = []
    for sc in ("hard", "ood", "nominal"):
        a = summ.get(("t1_ppo", sc)); b = summ.get(("t2_absoft", sc))
        if not a or not b: continue
        na = sum(int(float(x.get("episodes",0) or 0)) for x in a)
        nb = sum(int(float(x.get("episodes",0) or 0)) for x in b)
        ka = sum(float(x["collision"])*int(float(x.get("episodes",0) or 0)) for x in a)
        kb = sum(float(x["collision"])*int(float(x.get("episodes",0) or 0)) for x in b)
        tp = two_prop(round(ka), na, round(kb), nb)
        if tp: tests.append(dict(scene=sc, metric="collision (episode-level 2-prop)",
                                 stat=f"z={tp['z']:.2f}", p=tp["p"],
                                 ppo=f"{tp['p1']:.4f} ({round(ka)}/{na})",
                                 ours=f"{tp['p2']:.4f} ({round(kb)}/{nb})"))
        for k in ("return","min_obstacle_dist","detour_ratio","tracking_error_ema"):
            va = [float(x[k]) for x in a if x.get(k)]; vb = [float(x[k]) for x in b if x.get(k)]
            w = welch(va, vb)
            if w: tests.append(dict(scene=sc, metric=k+" (Welch, seed-level)",
                                    stat=f"t={w['t']:.2f}", p=w["p"],
                                    ppo=f"{st.mean(va):.4f}±{st.pstdev(va):.4f}",
                                    ours=f"{st.mean(vb):.4f}±{st.pstdev(vb):.4f}"))
    if tests:
        with open(f"{DATA}/significance.csv","w",newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(tests[0].keys())); w.writeheader(); w.writerows(tests)
        notes.append(f"significance.csv: {len(tests)} 项检验")


    def best_scene(models):
        """在候选场景里挑覆盖模型最多的那个；并列时偏好有区分度的 hard。"""
        cand = ["hard", "ood", "nominal"]
        best, bn = "ood", -1
        for x in cand:
            n = sum(1 for m in models if (m, x) in summ)
            if n > bn: best, bn = x, n
        return best, bn

    # ---------- 图 1：主结果 ----------
    try:
        sc, _ = best_scene(["t1_ppo","t2_absoft"])
        a, b = summ[("t1_ppo",sc)], summ[("t2_absoft",sc)]
        panels = [("collision","Violation rate"), ("return","Episode return"),
                  ("min_obstacle_dist","Min clearance (m)"), ("detour_ratio","Detour ratio")]
        fig, axes = create_subplots(1, 4, figsize=(19, 4.2))
        for ax,(k,lab) in zip(axes, panels):
            va=[float(x[k]) for x in a]; vb=[float(x[k]) for x in b]
            make_grouped_bar(ax, [""], [[st.mean(va)],[st.mean(vb)]],
                             ["PPO baseline","RP-PSF (ours)"], ylabel=lab,
                             colors=[PALETTE["neutral"], PALETTE["blue_main"]],
                             errs=[[st.pstdev(va)],[st.pstdev(vb)]])
            ax.set_xticks([])
        axes[0].legend(loc="upper left", bbox_to_anchor=(0,1.22), ncol=2)
        finalize_figure(fig, f"{FIGS}/fig1_main_results")
        notes.append(f"fig1_main_results (scene={sc}, n={len(a)} seeds)")
    except Exception as e: notes.append(f"fig1 跳过: {e}")

    # ---------- 图 2：soft_hi 敏感性 ----------
    try:
        order = [("t3_hi0.3",0.3),("t2_absoft",0.6),("t4_hi0.9",0.9),("t5_hi1.2",1.2)]
        SC2, _ = best_scene([m for m,_ in order])
        got = [(m,h) for m,h in order if (m,SC2) in summ]
        if len(got) < 2: raise ValueError("数据不足")
        fig, axes = create_subplots(1, 3, figsize=(15, 4.2))
        for ax,(k,lab) in zip(axes, [("collision","Violation rate"),("return","Episode return"),
                                     ("filter_activation","Filter engagement")]):
            xs=[h for _,h in got]
            ys=[st.mean([float(x[k]) for x in summ[(m,SC2)]]) for m,_ in got]
            es=[st.pstdev([float(x[k]) for x in summ[(m,SC2)]]) for m,_ in got]
            ax.errorbar(xs, ys, yerr=es, marker="o", lw=2.6, capsize=4,
                        color=PALETTE["blue_main"], mfc="white", mew=2.2, ms=9)
            ax.set_xlabel(r"$h_{hi}$ (m)"); ax.set_ylabel(lab); ax.grid(alpha=.25)
        finalize_figure(fig, f"{FIGS}/fig2_softhi_sensitivity")
        notes.append(f"fig2_softhi_sensitivity ({len(got)} 个点)")
    except Exception as e: notes.append(f"fig2 跳过: {e}")

    # ---------- 图 3：消融 ----------
    try:
        abl = ["t2_absoft","r1a_pred_binary","r1b_geom_soft","r1c_geom_binary",
               "r1f_a_only","r1d_fixed_C","r1e_adaptive_C","t1_ppo"]
        SC2, _ = best_scene(abl)
        got = [m for m in abl if (m,SC2) in summ]
        if len(got) < 3: raise ValueError("数据不足")
        fig, axes = create_subplots(1, 2, figsize=(16, 4.6))
        for ax,(k,lab) in zip(axes,[("collision","Violation rate"),("return","Episode return")]):
            vals=[st.mean([float(x[k]) for x in summ[(m,SC2)]]) for m in got]
            errs=[st.pstdev([float(x[k]) for x in summ[(m,SC2)]]) for m in got]
            cols=[PALETTE["blue_main"] if m=="t2_absoft" else
                  (PALETTE["neutral"] if m=="t1_ppo" else PALETTE["blue_secondary"]) for m in got]
            ax.bar(range(len(got)), vals, 0.68, yerr=errs, capsize=4, color=cols,
                   error_kw=dict(lw=1.5, ecolor="#444"), zorder=3)
            ax.set_xticks(range(len(got)))
            ax.set_xticklabels([LABEL.get(m,m) for m in got], rotation=30, ha="right")
            ax.set_ylabel(lab); ax.grid(axis="y", alpha=.25, zorder=0)
        finalize_figure(fig, f"{FIGS}/fig3_ablation")
        notes.append(f"fig3_ablation ({len(got)} 个变体)")
    except Exception as e: notes.append(f"fig3 跳过: {e}")


    # ---------- 图 3b：训练期消融（2×2 + 变体）----------
    try:
        import re as _re
        FIXED = {"T1 PPO baseline": "20260711_094733", "RP-PSF (pred.+prop.)": "20260711_165556"}
        DYN = [("Predictive + binary","r1a_pred_binary"), ("Geometric + prop.","r1b_geom_soft"),
               ("Geometric + binary","r1c_geom_binary"), ("A-only","r1f_a_only"),
               ("+ fixed C","r1d_fixed_C"), ("+ adaptive C","r1e_adaptive_C")]
        keys = dict(FIXED)
        for lab, lbl in DYN:
            fp = f"{ROOT}/outputs_aei/{lbl}.log"
            if os.path.exists(fp):
                m = _re.search(r"offline-run-([0-9_]+)-", open(fp, errors="replace").read())
                if m: keys[lab] = m.group(1)
        last = {}
        for r in tr: last[r["run"]] = r
        labs, vals = [], []
        for lab, k in keys.items():
            for run, r in last.items():
                if k in run: labs.append(lab); vals.append(float(r["cum_violation"])); break
        if len(vals) < 3: raise ValueError("数据不足")
        idx = sorted(range(len(vals)), key=lambda i: vals[i])
        labs = [labs[i] for i in idx]; vals = [vals[i] for i in idx]
        cols = [PALETTE["blue_main"] if "RP-PSF" in l else
                (PALETTE["neutral"] if "PPO baseline" in l else PALETTE["blue_secondary"]) for l in labs]
        fig, axes = create_subplots(1, 1, figsize=(9.5, 4.8)); ax = axes[0]
        ax.bar(range(len(vals)), vals, 0.68, color=cols, zorder=3)
        for i, v in enumerate(vals):
            ax.annotate(f"{v:.3f}", (i, v), textcoords="offset points",
                        xytext=(0, 4), ha="center", fontsize=11)
        ax.set_xticks(range(len(labs)))
        ax.set_xticklabels(labs, rotation=30, ha="right")
        ax.set_ylabel("Cumulative violations (training)")
        ax.grid(axis="y", alpha=.25, zorder=0)
        finalize_figure(fig, f"{FIGS}/fig3b_training_ablation")
        notes.append(f"fig3b_training_ablation ({len(vals)} 个变体, 单训练 seed)")
        with open(f"{DATA}/training_ablation.csv","w",newline="") as f:
            w = csv.writer(f); w.writerow(["variant","cum_violation"]); w.writerows(zip(labs, vals))
    except Exception as e: notes.append(f"fig3b 跳过: {e}")

    # ---------- 图 4：训练期累计违约曲线 ----------
    try:
        RUNS = {"20260711_094733":"PPO baseline","20260711_165556":"RP-PSF (ours)"}
        for lbl in ("r1a_pred_binary","r1b_geom_soft","r1c_geom_binary","r1f_a_only"):
            p=f"{ROOT}/outputs_aei/{lbl}.log"
            if os.path.exists(p):
                m=re.search(r"offline-run-([0-9_]+)-", open(p, errors="replace").read())
                if m: RUNS[m.group(1)] = LABEL.get(lbl, lbl)
        series, labels = [], []
        for key, lab in RUNS.items():
            pts=[(int(r["step"]), float(r["cum_violation"])) for r in tr if key in r["run"]]
            if len(pts) > 3:
                pts.sort(); series.append(pts); labels.append(lab)
        if not series: raise ValueError("无训练曲线")
        fig, axes = create_subplots(1, 1, figsize=(7.2, 4.6))
        ax = axes[0]
        for i,(pts,lab) in enumerate(zip(series,labels)):
            xs=[p[0] for p in pts]; ys=[p[1] for p in pts]
            ax.plot(xs, ys, lw=2.6, label=lab, zorder=3,
                    color=[PALETTE["neutral"],PALETTE["blue_main"],PALETTE["red_strong"],
                           PALETTE["teal"],PALETTE["violet"],PALETTE["green_3"]][i%6])
        ax.set_xlabel("Training iteration"); ax.set_ylabel("Cumulative violations")
        ax.grid(alpha=.25, zorder=0); ax.legend()
        finalize_figure(fig, f"{FIGS}/fig4_training_violations")
        notes.append(f"fig4_training_violations ({len(series)} 条曲线)")
    except Exception as e: notes.append(f"fig4 跳过: {e}")

    # ---------- 图 5：计算延迟 ----------
    try:
        lat = json.load(open(f"{DATA}/latency.json"))
        fig, axes = create_subplots(1, 1, figsize=(7.2, 4.6)); ax = axes[0]
        Ks = sorted({int(k.split("_")[0]) for k in lat})
        Hs = sorted({int(k.split("_")[1]) for k in lat})
        for i,H in enumerate(Hs):
            ys=[lat.get(f"{K}_{H}",{}).get("ms_per_step") for K in Ks]
            ax.plot(Ks, ys, marker="o", lw=2.4, label=f"N={H}",
                    color=[PALETTE["blue_main"],PALETTE["teal"],PALETTE["violet"]][i%3])
        ax.axhline(16.0, ls="--", lw=2, color=PALETTE["red_strong"])
        ax.text(Ks[0], 16.6, "62.5 Hz budget (16 ms)", color=PALETTE["red_strong"], fontsize=12)
        ax.set_xlabel("MPPI samples K"); ax.set_ylabel("Latency per step (ms)")
        ax.grid(alpha=.25); ax.legend()
        finalize_figure(fig, f"{FIGS}/fig5_latency")
        notes.append("fig5_latency")
    except Exception as e: notes.append(f"fig5 跳过: {e}")

    # ---------- RESULTS.md ----------
    with open(f"{ROOT}/outputs_aei/RESULTS.md","w") as f:
        f.write("# AEI 实验结果\n\n> 配置 P1：mppi.num_samples=128, mppi.horizon=20, risk.horizon=15\n")
        f.write("> 单训练 seed（seed=0），评测 10 个独立 seed\n\n## 产出\n\n")
        for n in notes: f.write(f"- {n}\n")
        if rows:
            f.write("\n## 评测汇总（mean ± std over seeds）\n\n")
            f.write("| model | scene | seeds | eps | collision | return | min_dist | detour | 激活率 |\n")
            f.write("|---|---|---|---|---|---|---|---|---|\n")
            for r in rows:
                g=lambda k: f"{r.get(k+'_mean',float('nan')):.4f}±{r.get(k+'_std',0):.4f}"
                f.write(f"| {LABEL.get(r['model'],r['model'])} | {r['scene']} | {r['n_seeds']} | {r['episodes']} | "
                        f"{g('collision')} | {g('return')} | {g('min_obstacle_dist')} | "
                        f"{g('detour_ratio')} | {g('filter_activation')} |\n")
        if tests:
            f.write("\n## 显著性检验\n\n| scene | metric | PPO | ours | stat | p |\n|---|---|---|---|---|---|\n")
            for t in tests:
                f.write(f"| {t['scene']} | {t['metric']} | {t['ppo']} | {t['ours']} | {t['stat']} | {t['p']:.4g} |\n")
        nom=f"{DATA}/nominal_onestep.json"
        if os.path.exists(nom):
            d=json.load(open(nom))
            f.write("\n## 名义模型单步误差（N1）\n\n| 变体 | 中位相对误差 | 均值 | n |\n|---|---|---|---|\n")
            for k,v in d.items():
                f.write(f"| {k} | {v['median']*100:.2f}% | {v['mean']*100:.2f}% | {v['n']} |\n")
    print("\n".join(notes)); print("→ outputs_aei/RESULTS.md")

if __name__ == "__main__":
    main()

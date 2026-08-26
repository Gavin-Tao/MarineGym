"""论文③ 轨迹对比图：直观展示"载具追不上参考"这一能力受限现象。

用 `train.py +save_traj=<npz>` 采到的轨迹（含 pos / ref / full_ref / done），
画三个面板：
  (a) XY 平面：参考 lemniscate vs 载具实际路径，标出 episode 终止点
  (b) 跟踪误差 vs 时间：多条 episode 叠加 + 均值，与 reset_thres 阈值线对比
  (c) 速度对比：参考点速度 vs 载具速度 —— 能力受限的直接证据

用法:
    python p3_traj_fig.py --npz outputs_p3/traj/drift_mlp.npz [--npz2 ... --label ...]
"""
import argparse
import json
from pathlib import Path

import numpy as np

import aei_style as st


def load(npz_path, dt=0.016):
    d = np.load(npz_path)
    pos = d["pos"]            # [T, E, 3]
    ref = d["ref"]            # [T, E, 3]
    done = d["done"]          # [T, E]
    err = np.linalg.norm(pos - ref, axis=-1)          # [T, E]
    # 参考点速度与载具速度（差分，首帧补齐）
    vref = np.linalg.norm(np.diff(ref, axis=0), axis=-1) / dt
    vveh = np.linalg.norm(np.diff(pos, axis=0), axis=-1) / dt
    out = dict(pos=pos, ref=ref, done=done, err=err, vref=vref, vveh=vveh, dt=dt)
    if "full_ref" in d:
        out["full_ref"] = d["full_ref"]
    return out


def first_done(done):
    """每个 env 第一次 done 的步号；没 done 过则返回 T。"""
    T, E = done.shape
    idx = np.full(E, T, dtype=int)
    for e in range(E):
        w = np.flatnonzero(done[:, e])
        if len(w):
            idx[e] = w[0]
    return idx


def main():
    ap = argparse.ArgumentParser()
    here = Path(__file__).parent
    ap.add_argument("--npz", required=True)
    ap.add_argument("--npz2", default=None)
    ap.add_argument("--label", default="MLP (single frame)")
    ap.add_argument("--label2", default=None)
    ap.add_argument("--reset-thres", type=float, default=0.5)
    ap.add_argument("--out", default=str(here / "outputs_p3" / "report" / "fig5_traj"))
    a = ap.parse_args()

    A = load(a.npz)
    B = load(a.npz2) if a.npz2 else None
    fd = first_done(A["done"])
    # 选一条有代表性的 episode：终止步数最接近中位数的那个 env
    med = int(np.median(fd))
    e0 = int(np.argmin(np.abs(fd - med)))
    T_end = int(fd[e0])

    st.apply_publication_style()
    fig, axes = st.create_subplots(1, 3, figsize=(18, 4.8))
    C_ref, C_a, C_b = "#333333", st.PALETTE["blue_main"], st.PALETTE["red_strong"]

    # ---------------- (a) XY 轨迹 ----------------
    ax = axes[0]
    if "full_ref" in A:
        fr = A["full_ref"][e0] if A["full_ref"].ndim == 3 else A["full_ref"]
        ax.plot(fr[:, 0], fr[:, 1], "--", color=C_ref, lw=1.6, alpha=.55,
                label="Reference (full lap)")
    ax.plot(A["ref"][:T_end, e0, 0], A["ref"][:T_end, e0, 1], "-", color=C_ref,
            lw=3.0, label="Reference (flown span)")
    ax.plot(A["pos"][:T_end, e0, 0], A["pos"][:T_end, e0, 1], "-", color=C_a,
            lw=2.6, label=a.label)
    ax.scatter(*A["pos"][0, e0, :2], s=90, color=C_a, edgecolor="k", zorder=5,
               label="start")
    ax.scatter(*A["pos"][T_end - 1, e0, :2], s=150, marker="X", color=C_a,
               edgecolor="k", zorder=5, label=f"terminated (step {T_end})")
    if B is not None:
        fdB = first_done(B["done"]); TB = int(fdB[e0])
        ax.plot(B["pos"][:TB, e0, 0], B["pos"][:TB, e0, 1], "-", color=C_b,
                lw=2.6, label=a.label2 or "arm B")
        ax.scatter(*B["pos"][TB - 1, e0, :2], s=150, marker="X", color=C_b,
                   edgecolor="k", zorder=5)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title("(a) Reference vs flown path", fontsize=13)
    ax.legend(fontsize=8, loc="best"); ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=.3, ls=":"); ax.set_axisbelow(True)

    # ---------------- (b) 跟踪误差 vs 时间 ----------------
    ax = axes[1]
    t = np.arange(A["err"].shape[0]) * A["dt"]
    for e in range(min(A["err"].shape[1], 16)):
        n = int(fd[e])
        ax.plot(t[:n], A["err"][:n, e], color=C_a, alpha=.18, lw=1.0)
    n_med = int(np.median(fd))
    mean_err = np.array([np.mean([A["err"][s, e] for e in range(A["err"].shape[1])
                                  if fd[e] > s]) for s in range(n_med)])
    ax.plot(t[:n_med], mean_err, color=C_a, lw=3.0, label=f"{a.label} (mean)")
    if B is not None:
        fdB = first_done(B["done"]); nB = int(np.median(fdB))
        mB = np.array([np.mean([B["err"][s, e] for e in range(B["err"].shape[1])
                                if fdB[e] > s]) for s in range(nB)])
        ax.plot(t[:nB], mB, color=C_b, lw=3.0, label=f"{a.label2 or 'arm B'} (mean)")
    ax.axhline(a.reset_thres, color=st.PALETTE["red_strong"], ls="--", lw=2.0)
    ax.text(t[1], a.reset_thres * 1.03, f"reset_thres = {a.reset_thres} m",
            fontsize=10, va="bottom")
    # 线性增长参考线：由速度亏空决定的理论包络
    if len(mean_err) > 5:
        k = np.polyfit(t[:n_med], mean_err, 1)[0]
        ax.plot(t[:n_med], k * t[:n_med], ":", color="#666", lw=2.0,
                label=f"linear fit: {k:.2f} m/s deficit")
    ax.set_xlabel("time (s)"); ax.set_ylabel("tracking error (m)")
    ax.set_title("(b) Error grows linearly until termination", fontsize=13)
    ax.legend(fontsize=9); ax.grid(alpha=.3, ls=":"); ax.set_axisbelow(True)

    # ---------------- (c) 速度对比 ----------------
    ax = axes[2]
    tv = np.arange(A["vref"].shape[0]) * A["dt"]
    mask = np.arange(A["vref"].shape[0])[:, None] < fd[None, :]
    vr = np.where(mask, A["vref"], np.nan)
    vv = np.where(mask, A["vveh"], np.nan)
    n = int(np.median(fd))
    ax.plot(tv[:n], np.nanmean(vr[:n], axis=1), "-", color=C_ref, lw=3.0,
            label="Reference speed")
    ax.plot(tv[:n], np.nanmean(vv[:n], axis=1), "-", color=C_a, lw=3.0,
            label=f"{a.label} speed")
    ax.fill_between(tv[:n], np.nanmean(vv[:n], axis=1), np.nanmean(vr[:n], axis=1),
                    color=st.PALETTE["red_1"], alpha=.55, label="speed deficit")
    ax.set_xlabel("time (s)"); ax.set_ylabel("speed (m/s)")
    ax.set_title("(c) The vehicle cannot keep up", fontsize=13)
    ax.legend(fontsize=9); ax.grid(alpha=.3, ls=":"); ax.set_axisbelow(True)

    saved = st.finalize_figure(fig, a.out)

    summary = {
        "median_episode_len": int(np.median(fd)),
        "mean_ref_speed": float(np.nanmean(vr[:n])),
        "mean_vehicle_speed": float(np.nanmean(vv[:n])),
        "speed_deficit": float(np.nanmean(vr[:n]) - np.nanmean(vv[:n])),
        "err_growth_rate_m_per_s": float(np.polyfit(t[:n_med], mean_err, 1)[0])
        if len(mean_err) > 5 else None,
        "reset_thres": a.reset_thres,
    }
    if summary["speed_deficit"] > 0:
        summary["predicted_episode_len_steps"] = (
            a.reset_thres / summary["speed_deficit"] / A["dt"])
    Path(a.out).with_name(Path(a.out).name + "_summary.json").write_text(
        json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    print("saved:", *saved, sep="\n  ")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""论文② 环境体检表 (V 层)。

纯后处理：只读 train.py `+save_traj` 产出的 .npz，不启动 Isaac。
产出 PASS/FAIL 表 —— 每条都有硬判据，不靠肉眼看日志说"差不多对"。

用法
----
  # V0 回归：新代码 vs 参考 rollout（先跑 noise 建立复现噪声底）
  python flow_validate.py noise --a v0_ref.npz --b v0_ref2.npz
  python flow_validate.py v0    --ref v0_ref.npz --new v0_new.npz

  # V1/V2/V3 洋流物理：每个方向一个 npz，const_action=0 自由漂移
  python flow_validate.py flow --npz drift_px.npz --expect 0.5 0 0
  python flow_validate.py flow-suite --dir outputs_flow/flowcheck

  # 全表
  python flow_validate.py report --dir outputs_flow

npz 字段（train.py 记录）：pos/ref/quat/vel/flow/reward/done/u_applied ...
  vel  = drone_state[7:13]，**世界系** [lin(3), ang(3)]（underwaterVehicle.py:132 self.vel = self.vel_w）
  flow = drone.flow_vels，**世界系** [lin(3), ang(3)]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

# ── 判据 ────────────────────────────────────────────────────────────────────
# V0 的容差不是拍脑袋：用 `noise` 子命令跑两次同参采集测出的复现噪声底。
# 实测 2026-08-25（BlueROV/Track/GPU2/num_envs=16/300步）：pos/vel/reward 三项
# 的 max|Δ| 全为 **0** —— 环境逐位确定性。因此 V0 用严格相等，任何改动无处可藏。
# 若换机器/换 GPU 后噪声底不再为 0，先重跑 noise 再按实测值放宽。
V0_TOL = dict(pos=0.0, vel=0.0, reward=0.0)
DRIFT_RATIO_TOL = (0.80, 1.20)   # 稳态漂移速度 / 设定流速
DRIFT_COS_TOL = 0.95             # 漂移方向与流向的余弦

_rows = []


def _chk(name, ok, detail):
    _rows.append((name, "PASS" if ok else "FAIL", detail))
    return ok


def _load(p):
    d = np.load(p)
    return {k: d[k] for k in d.files}


def _fmt():
    if not _rows:
        return "(无检查项)"
    w = max(len(r[0]) for r in _rows)
    out = []
    for name, verdict, detail in _rows:
        # "----" = 信息性行（如 K4 的三档对比），不是判定，别渲染成 FAIL
        mark = {"PASS": "\033[32mPASS\033[0m", "FAIL": "\033[31mFAIL\033[0m"}.get(verdict, "    ")
        out.append(f"{name:<{w}}  {mark}  {detail}")
    return "\n".join(out)


def _failed():
    return any(r[1] == "FAIL" for r in _rows)


# ── V0 回归 ─────────────────────────────────────────────────────────────────
def _diff(a, b):
    """两份 rollout 的逐字段最大绝对差。形状不同 → inf。"""
    out = {}
    for k in ("pos", "vel", "reward"):
        if k not in a or k not in b:
            out[k] = float("nan")
            continue
        x, y = a[k], b[k]
        if x.shape != y.shape:
            out[k] = float("inf")
            continue
        out[k] = float(np.nanmax(np.abs(x - y)))
    return out


def cmd_noise(args):
    """同参数跑两次，测 GPU 物理的 run-to-run 复现噪声底。

    这是 V0 的前置实验控制：不知道噪声底就不知道 V0 的容差该设多少。
    理想是 0（完全确定性）；若不是 0，V0_TOL 要按此调整。
    """
    a, b = _load(args.a), _load(args.b)
    d = _diff(a, b)
    for k, v in d.items():
        _chk(f"noise-floor {k}", v == 0.0, f"max|Δ| = {v:.3e}  ({'确定性' if v == 0 else '非确定性，V0_TOL 需 ≥ 此值'})")
    print(_fmt())
    print("\n→ 把 V0_TOL 设为上面各项的 ~10 倍（全 0 则保持严格相等）。")
    return 0


def cmd_v0(args):
    """回归：默认配置（tube/keepout 全关）下，新代码与参考 rollout 是否一致。"""
    ref, new = _load(args.ref), _load(args.new)
    d = _diff(ref, new)
    for k, v in d.items():
        tol = V0_TOL[k]
        _chk(f"V0 regression {k}", v <= tol, f"max|Δ| = {v:.3e}  (tol {tol:.1e})")
    # episode 结构也要一致：done 模式一变说明终止条件被改了
    if "done" in ref and "done" in new and ref["done"].shape == new["done"].shape:
        same = bool((ref["done"] == new["done"]).all())
        _chk("V0 regression done", same, f"done 模式{'一致' if same else '不一致 → 终止条件被改动'}")
    print(_fmt())
    return 1 if _failed() else 0


# ── V1/V2/V3 洋流物理 ───────────────────────────────────────────────────────
def _drift(npz, tail_frac=0.3):
    """自由漂移分析。要求该 rollout 用 const_action=0（零推力）+ 恒定流。

    **逐 env** 比较，而不是先跨 env 平均 —— `set_flow_velocities` 采的是
    `rand_like(...) * max_flow_vel`，即均匀[0,max] 且每个 env 取值不同。拿配置的
    max 当分母会得到 ratio≈0.5 的假失败；正确的参照是 npz 里实际记录的 flow_vels。

    只看水平分量：垂直方向被浮力/重力不平衡污染（载具未必中性浮力），
    不能用来判流场是否正确。
    """
    vel, flow = npz["vel"], npz["flow"]                 # [T,E,6] 均为世界系
    n = max(1, int(vel.shape[0] * tail_frac))
    v_ss = vel[-n:, :, :2].mean(axis=0)                 # [E,2] 稳态水平速度
    v_fl = flow[-n:, :, :2].mean(axis=0)                # [E,2] 实际生效的水平流速
    nv = np.linalg.norm(v_ss, axis=-1)
    nf = np.linalg.norm(v_fl, axis=-1)
    m = nf > 1e-6                                        # 只统计流速非零的 env
    ratio = nv[m] / nf[m]
    cos = (v_ss[m] * v_fl[m]).sum(-1) / (nv[m] * nf[m] + 1e-12)
    return ratio, cos, v_ss, v_fl, vel[-n:, :, :3].mean(axis=(0, 1))


def _fit_asymptote(npz, dt=0.016, head_frac=0.35):
    """把 r(t)=|v_xy|/|v_flow,xy| 拟合成 r = r∞ − A·exp(−t/τ)，返回 (r∞, τ[s])。

    做法：先对尾段用两点法估 τ（假设 r∞=1 求初值），再用该 τ 对
    r(t) = r∞ − A·exp(−t/τ) 做线性最小二乘解出 r∞ 和 A。
    只用后 (1−head_frac) 段，避开二次阻尼主导的初始快速段。
    """
    vel, flow = npz["vel"], npz["flow"]
    nf = np.linalg.norm(flow[:, :, :2], axis=-1)
    nv = np.linalg.norm(vel[:, :, :2], axis=-1)
    m = nf[0] > 1e-6
    if m.sum() == 0:
        return float("nan"), float("nan")
    r = np.median(nv[:, m] / nf[:, m], axis=1)           # [T]
    T = len(r)
    i0 = int(T * head_frac)
    t = np.arange(i0, T) * dt
    y = r[i0:]
    resid = 1.0 - y
    if (resid <= 1e-6).any():                             # 已经收敛到 1，无需外推
        return float(y[-1]), float("nan")
    # 两点法估 τ
    k = len(resid) // 2
    a, b = resid[:k].mean(), resid[k:].mean()
    ta, tb = t[:k].mean(), t[k:].mean()
    if not (a > 0 and b > 0 and a != b):
        return float("nan"), float("nan")
    tau = (tb - ta) / np.log(a / b)
    if not np.isfinite(tau) or tau <= 0:
        return float("nan"), float("nan")
    # 固定 τ，线性解 r∞ 与 A:  y = r∞ − A·exp(−t/τ)
    X = np.stack([np.ones_like(t), -np.exp(-t / tau)], axis=1)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(coef[0]), float(tau)


def _flow_checks(tag, npz, expect):
    ratio, cos, v_ss, v_fl, v_ss3 = _drift(npz)
    e = np.asarray(expect, dtype=float)
    if ratio.size == 0:
        _chk(f"V1 flow-drives  [{tag}]", False, "记录的 flow_vels 全为 0 → enable_flow 没生效")
        return
    # V1：流真的驱动了载具（零推力下不该原地不动）
    med = float(np.median(np.linalg.norm(v_ss, axis=-1)))
    _chk(f"V1 flow-drives    [{tag}]", med > 0.05,
         f"|v_ss,xy| 中位 = {med:.3f} m/s   |v_flow,xy| 中位 = {float(np.median(np.linalg.norm(v_fl, axis=-1))):.3f}")
    # V2：方向/帧正确（world↔body 转换错、符号错都会在这里露馅）
    c = float(np.median(cos))
    _chk(f"V2 flow-direction [{tag}]", c >= DRIFT_COS_TOL,
         f"cos(v_ss, v_flow) 中位 = {c:.3f}  (≥{DRIFT_COS_TOL})   v_ss(xyz)={np.round(v_ss3, 3)}")
    # V3：幅值标定（稳态阻力平衡下应≈1:1）
    r = float(np.median(ratio))
    _chk(f"V3 flow-magnitude [{tag}]", DRIFT_RATIO_TOL[0] <= r <= DRIFT_RATIO_TOL[1],
         f"|v_ss|/|v_flow| 中位 = {r:.3f}  (tol {DRIFT_RATIO_TOL})  逐 env 范围 [{ratio.min():.2f}, {ratio.max():.2f}]")
    # V3b：渐近线拟合。零推力下载具最终应被流完全平流（r→1），但低速段只剩线性阻尼，
    # 时间常数 τ≈25 s，尾巴极长。拟合 log(1−r) 对 t 的直线，外推 r 的渐近值 —— 比等收敛快得多，
    # 也能区分"没跑够"和"物理错了"：r∞≈1 说明只是没收敛，r∞ 明显偏离 1 才是真 bug。
    r_inf, tau_s = _fit_asymptote(npz)
    if np.isfinite(r_inf):
        _chk(f"V3b drift-asymptote[{tag}]", 0.90 <= r_inf <= 1.10,
             f"外推 r∞ = {r_inf:.3f}  (期望 1.0，零推力最终被完全平流)   τ = {tau_s:.1f} s")
    # V2b：采样特性存档 —— 记录实际 flow 相对配置 max 的分布。
    # 期望看到均匀[0,max]：mean/max ≈ 0.5，且符号与配置一致。这不是 FAIL 项，是事实记录。
    ne = np.linalg.norm(e[:2])
    if ne > 1e-9:
        frac = np.linalg.norm(v_fl, axis=-1) / ne
        _chk(f"V2b sampling      [{tag}]", bool((frac <= 1.02).all()),
             f"|flow|/|配置max| ∈ [{frac.min():.2f}, {frac.max():.2f}] 均值 {frac.mean():.2f} "
             f"(均匀[0,max] → 期望均值≈0.50)")


def cmd_flow(args):
    _flow_checks(Path(args.npz).stem, _load(args.npz), args.expect)
    print(_fmt())
    return 1 if _failed() else 0


def cmd_flow_suite(args):
    """六向漂移套件：drift_{px,nx,py,ny,pz,nz}.npz，文件名编码期望流向。"""
    dirs = {"px": (1, 0, 0), "nx": (-1, 0, 0), "py": (0, 1, 0),
            "ny": (0, -1, 0), "pz": (0, 0, 1), "nz": (0, 0, -1)}
    d = Path(args.dir)
    found = 0
    for tag, unit in dirs.items():
        f = d / f"drift_{tag}.npz"
        if not f.exists():
            continue
        found += 1
        _flow_checks(tag, _load(f), np.asarray(unit, dtype=float) * args.speed)
    if not found:
        print(f"没找到任何 drift_*.npz（在 {d}）", file=sys.stderr)
        return 2
    print(_fmt())
    if any(t in r[0] for r in _rows if r[1] == "FAIL" for t in ("nx", "ny", "nz")):
        print("\n注意：负方向失败但正方向通过 = underwaterVehicle.py:324 的 "
              "`rand_like(...) * max_flow_vel` 恒正采样 bug。")
    return 1 if _failed() else 0


# ── K4：扰动观测器是否让预测门控可信 ───────────────────────────────────────
def cmd_k4(args):
    """比较三种 d̂ 下的**预测最小侧壁裕度** 与 未来 H 步的**实际最小裕度**。

    测的是门控真正用到的那个量，而不是泛泛的位置预测误差 —— 这才是决定
    "本篇一号贡献成不成立"的量。判据：
      · d̂=est 的 MAE 必须显著小于 d̂=0，否则观测器没贡献 → 一号贡献不成立
      · est 应接近 oracle，差距即在线估计相对上界的损失
      · 门控决策一致率（pred<thr 与 actual<thr 是否同号）比 MAE 更直接
    """
    d = _load(args.npz)
    H = int(args.horizon)
    need = ("wall_clear", "pred_clear", "pred_clear_zero", "pred_clear_oracle")
    miss = [k for k in need if k not in d or np.isnan(d[k]).all()]
    if miss:
        print(f"缺字段(或全 NaN): {miss} —— 需要 task.safety.k4=true 且 risk.enable=true", file=sys.stderr)
        return 2
    wc = d["wall_clear"]                                   # [T,E] 实际裕度
    T = wc.shape[0]
    if T <= H:
        print(f"步数 {T} <= H {H}，无法构造未来窗口", file=sys.stderr)
        return 2
    # 未来 H 步的实际最小裕度（与 assess_corridor 的 min_{k=1..H} 同口径）
    fut = np.stack([wc[t + 1:t + 1 + H].min(axis=0) for t in range(T - H)])   # [T-H,E]
    # 只统计 episode 内部：窗口里有 done 的样本跨了 reset，实际轨迹不连续，必须剔除
    dn = d["done"]
    ok = np.stack([~dn[t + 1:t + 1 + H].any(axis=0) for t in range(T - H)])
    thr = float(args.threshold)
    for tag, key in (("d̂=0 (论文①假设)", "pred_clear_zero"),
                     ("d̂=est (ours)", "pred_clear"),
                     ("d̂=oracle (上界)", "pred_clear_oracle")):
        pr = d[key][:T - H]
        m = ok & np.isfinite(pr) & np.isfinite(fut)
        if m.sum() == 0:
            _chk(f"K4 {tag}", False, "无有效样本")
            continue
        mae = float(np.abs(pr[m] - fut[m]).mean())
        bias = float((pr[m] - fut[m]).mean())
        agree = float(((pr[m] < thr) == (fut[m] < thr)).mean())
        _rows.append((f"K4 {tag}", "----",
                      f"MAE={mae:.4f} m   偏置={bias:+.4f}   门控一致率={agree:.3f}   n={m.sum()}"))
    print(_fmt())
    print("\n判据：est 的 MAE 应显著低于 zero；est 接近 oracle。若 est≈zero → 观测器没贡献，"
          "本篇一号贡献不成立，需要改方向。")
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("noise", help="测 GPU 物理 run-to-run 复现噪声底（V0 的前置控制）")
    s.add_argument("--a", required=True); s.add_argument("--b", required=True)
    s.set_defaults(fn=cmd_noise)

    s = sub.add_parser("v0", help="回归：新代码 vs 参考 rollout")
    s.add_argument("--ref", required=True); s.add_argument("--new", required=True)
    s.set_defaults(fn=cmd_v0)

    s = sub.add_parser("flow", help="单个方向的洋流漂移检查")
    s.add_argument("--npz", required=True)
    s.add_argument("--expect", nargs=3, type=float, required=True, help="世界系设定流速 vx vy vz")
    s.set_defaults(fn=cmd_flow)

    s = sub.add_parser("k4", help="扰动观测器是否让预测门控可信（三种 d̂ 对比）")
    s.add_argument("--npz", required=True)
    s.add_argument("--horizon", type=int, default=15)
    s.add_argument("--threshold", type=float, default=0.6)
    s.set_defaults(fn=cmd_k4)

    s = sub.add_parser("flow-suite", help="六向漂移套件")
    s.add_argument("--dir", required=True)
    s.add_argument("--speed", type=float, default=0.5)
    s.set_defaults(fn=cmd_flow_suite)

    a = p.parse_args()
    sys.exit(a.fn(a))


if __name__ == "__main__":
    main()

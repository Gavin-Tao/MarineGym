"""论文③ 判别性探针：各编码器能否从历史中恢复"必须用历史才能得到的量"。

动机：在 RL 曲线上分不清两件事 ——
  (a) 编码器实现/优化有问题，学不动；
  (b) 任务本身不需要历史，所以历史模型没优势。
这个探针把 (a) 单独隔离出来，用一个**解析上可判定**的监督任务：

  输入   随机游走窗口 x[t] = x[t-1] + d[t]，形状 [B, L, D]
  目标   最后一步的增量 d[L-1]（即"速度"）

单帧观测 x[L-1] 与目标**统计独立**，故只看当前帧的 MLP 的最优解就是预测均值 0，
MSE 必然收敛到目标方差；任何真正用上历史的模型都能做差得到精确解。

于是：
  * MLP 的 MSE ≈ 目标方差   ⇒ 探针设计正确（确实需要历史）
  * 序列臂 MSE ≪ 目标方差   ⇒ 该臂的实现与优化没问题

用法: python p3_probe.py [--out outputs_p3/report] [--steps 2000]
产出: table_probe.md/.csv, fig0_probe.pdf/png, probe.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import aei_style as st
from marinegym.learning.modules.encoders import build_encoder

D, L, B = 37, 16, 256
SIGMA = 0.3                      # 增量标准差 → 目标方差 = SIGMA²

ARMS = {
    "mlp":         ({"name": "mlp"}, True),
    "stack":       ({"name": "stack"}, False),
    "gru":         ({"name": "gru", "d_model": 128, "n_layers": 1}, False),
    "transformer": ({"name": "transformer", "d_model": 128, "n_layers": 2,
                     "n_heads": 8, "max_len": L}, False),
    "mamba":       ({"name": "mamba", "d_model": 128, "n_layers": 1}, False),
}
LABEL = {"mlp": "MLP (current state only)", "stack": "Frame-stack", "gru": "GRU",
         "transformer": "Transformer (DTQN-style)", "mamba": "Mamba (ours)"}
COLOR = {"mlp": st.PALETTE["neutral"], "stack": st.PALETTE["green_3"],
         "gru": st.PALETTE["teal"], "transformer": st.PALETTE["blue_secondary"],
         "mamba": st.PALETTE["red_strong"]}


def make_batch(dev):
    d = torch.randn(B, L, D, device=dev) * SIGMA
    x = torch.cumsum(d, dim=1)
    return x.unsqueeze(1), d[:, -1, :3]      # [B,1,L,D] -> [B,3]


def run_arm(name, cfg, last_only, steps, lr, dev, seed):
    torch.manual_seed(seed)
    enc = build_encoder(cfg).to(dev)
    head = nn.Linear(256, 3).to(dev)
    x, _ = make_batch(dev)
    enc(x[:, :, -1:, :] if last_only else x)          # 具体化 LazyLinear
    opt = torch.optim.Adam(list(enc.parameters()) + list(head.parameters()), lr=lr)
    curve = []
    for step in range(steps + 1):
        x, y = make_batch(dev)
        xin = x[:, :, -1:, :] if last_only else x
        loss = ((head(enc(xin).squeeze(1)) - y) ** 2).mean()
        if step % 25 == 0:
            curve.append((step, loss.item()))
        opt.zero_grad(); loss.backward(); opt.step()
    return curve


# ---------------------------------------------------------------------------
# 探针 B：为什么窗口必须按步长采样(帧跳)
# ---------------------------------------------------------------------------
# 与真实情形同构的构造：载具做平滑运动，观测带噪声
#     x[t] = offset + v*t + eps[t],  eps ~ N(0, sigma_obs)
# 目标 = v。用相隔 k 帧的两点估计 v，噪声为 sqrt(2)*sigma_obs / k
# —— 即**信噪比正比于帧间隔 k**。
#
# 这正是 Track 任务里的处境：控制 62.5 Hz，位置量级 ~2 m，每步位移仅 ~2.6 cm，
# 相对差异约 1%。stride=1 的窗口相当于在最差的信噪比下工作。
STRIDE_GRID = [1, 2, 4, 8, 16]


# 量纲对齐真实 Track 任务：位置量级 ~2 m，每步位移 ~2.6 cm，
# 观测噪声(数值精度/传感器)取 ~1 cm ⇒ 相邻两帧估速度的信噪比只有 ~1.8。
V_SCALE = 0.026        # 每步位移 (m)
SIGMA_OFF = 2.0        # 位置量级 (m)


def smooth_batch(dev, stride, L, sigma_obs, sigma_off=SIGMA_OFF):
    v = torch.randn(B, 3, device=dev) * V_SCALE              # 待估的每步位移
    t = (torch.arange(L, device=dev) * stride).float()       # 窗口按 stride 采样
    off = torch.randn(B, 1, D, device=dev) * sigma_off       # 位置偏置(跨时间恒定)
    x = off + torch.zeros(B, L, D, device=dev)
    x[:, :, :3] = x[:, :, :3] + v.unsqueeze(1) * t.view(1, L, 1)
    x = x + torch.randn_like(x) * sigma_obs
    # 目标按 V_SCALE 归一化，MSE 才好读：完全学不会 ⇒ MSE ≈ 1.0
    return x.unsqueeze(1), v / V_SCALE


def probe_stride(out, steps, lr, dev, seeds, sigma_obs=0.01):
    """固定 L，扫 stride，看各臂估计速度的误差。"""
    res = {}
    print(f"\n=== 探针 B：窗口步长 vs 速度估计误差 ===")
    print(f"    每步位移 {V_SCALE} m，位置量级 {SIGMA_OFF} m，观测噪声 {sigma_obs} m")
    print(f"    相邻帧估速度信噪比 ≈ {V_SCALE/(1.414*sigma_obs):.1f}；MSE≈1.0 表示完全学不会")
    print(f"{'arm':>28} " + " ".join(f"stride={k:<3}" for k in STRIDE_GRID))
    for name, (cfg, last_only) in ARMS.items():
        if last_only:
            continue                      # 单帧臂原理上无法估速度，跳过
        row = []
        for stride in STRIDE_GRID:
            errs = []
            for sd in range(seeds):
                torch.manual_seed(sd)
                enc = build_encoder(cfg).to(dev); head = nn.Linear(256, 3).to(dev)
                x, _ = smooth_batch(dev, stride, L, sigma_obs); enc(x)
                opt = torch.optim.Adam(list(enc.parameters()) + list(head.parameters()), lr=lr)
                for step in range(steps + 1):
                    x, y = smooth_batch(dev, stride, L, sigma_obs)
                    loss = ((head(enc(x).squeeze(1)) - y) ** 2).mean()
                    opt.zero_grad(); loss.backward(); opt.step()
                with torch.no_grad():
                    x, y = smooth_batch(dev, stride, L, sigma_obs)
                    errs.append(((head(enc(x).squeeze(1)) - y) ** 2).mean().item())
            row.append(float(np.mean(errs)))
        res[name] = row
        print(f"{LABEL[name]:>28} " + " ".join(f"{v:<10.5f}" for v in row))
    return res


def main():
    ap = argparse.ArgumentParser()
    here = Path(__file__).parent
    ap.add_argument("--out", default=str(here / "outputs_p3" / "report"))
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    var = SIGMA ** 2

    res = {"meta": {"D": D, "L": L, "B": B, "sigma": SIGMA, "target_var": var,
                    "steps": a.steps, "lr": a.lr, "seeds": a.seeds}, "arms": {}}
    print(f"目标方差 = {var:.4f}（MLP 的理论下界；序列臂应远低于它）\n")
    print(f"{'arm':>28} {'step 0':>9} {'step 500':>9} {'final':>9} {'/var':>7}")
    for name, (cfg, last_only) in ARMS.items():
        curves = [run_arm(name, cfg, last_only, a.steps, a.lr, dev, s)
                  for s in range(a.seeds)]
        steps = [s for s, _ in curves[0]]
        Y = np.array([[v for _, v in c] for c in curves])
        m, sd = Y.mean(0), Y.std(0)
        res["arms"][name] = {"steps": steps, "mean": m.tolist(), "std": sd.tolist(),
                             "final": float(m[-1]), "final_over_var": float(m[-1] / var)}
        i500 = min(range(len(steps)), key=lambda i: abs(steps[i] - 500))
        print(f"{LABEL[name]:>28} {m[0]:>9.4f} {m[i500]:>9.4f} {m[-1]:>9.4f} "
              f"{m[-1]/var:>7.3f}")

    (out / "probe.json").write_text(json.dumps(res, indent=1))

    # ---- 表 ----
    lines = ["| Encoder | MSE @ step 0 | MSE @ 500 | Final MSE | Final / target var | Verdict |",
             "|---|---|---|---|---|---|"]
    csvl = ["arm,mse_0,mse_500,mse_final,final_over_var"]
    for name in ARMS:
        r = res["arms"][name]
        m = r["mean"]; steps = r["steps"]
        i500 = min(range(len(steps)), key=lambda i: abs(steps[i] - 500))
        v = r["final_over_var"]
        verdict = ("**cannot** (≈ predicting the mean)" if v > 0.8
                   else "partial" if v > 0.3 else "recovers it")
        lines.append(f"| {LABEL[name]} | {m[0]:.4f} | {m[i500]:.4f} | "
                     f"{r['final']:.4f} | {v:.3f} | {verdict} |")
        csvl.append(f"{name},{m[0]:.6f},{m[i500]:.6f},{r['final']:.6f},{v:.6f}")
    (out / "table_probe.md").write_text(
        "\n".join(lines) +
        f"\n\n> 随机游走窗口 (L={L}, D={D}, σ={SIGMA})，目标 = 最后一步增量。"
        f"\n> 单帧观测与目标统计独立 ⇒ 只看当前帧的最优 MSE = 目标方差 {var:.4f}。"
        f"\n> {a.seeds} 个种子，Adam lr={a.lr}，{a.steps} 步。\n")
    (out / "table_probe.csv").write_text("\n".join(csvl) + "\n")

    # ---- 图 ----
    st.apply_publication_style()
    fig, axes = st.create_subplots(1, 1, figsize=(7.5, 5))
    ax = axes[0]
    for name in ARMS:
        r = res["arms"][name]
        x = np.array(r["steps"]); m = np.array(r["mean"]); sd = np.array(r["std"])
        ax.plot(x, m, label=LABEL[name], color=COLOR[name], linewidth=2.4)
        ax.fill_between(x, np.maximum(m - sd, 1e-6), m + sd,
                        color=COLOR[name], alpha=.15, linewidth=0)
    ax.axhline(var, color="black", linestyle="--", linewidth=1.8)
    ax.text(x[-1], var * 1.15, "single-frame optimum (target variance)",
            ha="right", va="bottom", fontsize=10)
    ax.set_yscale("log")
    ax.set_xlabel("Gradient steps")
    ax.set_ylabel("MSE of recovered increment")
    ax.set_title("Can the encoder recover a quantity that needs history?", fontsize=14)
    ax.legend(fontsize=10, loc="lower left")
    ax.grid(alpha=.3, which="both", linestyle=":")
    ax.set_axisbelow(True)
    st.finalize_figure(fig, out / "fig0_probe")

    # ---- 探针 B：窗口步长 ----
    sres = probe_stride(out, max(a.steps // 2, 500), a.lr, dev, a.seeds)
    res["stride"] = {"grid": STRIDE_GRID, "arms": sres}
    (out / "probe.json").write_text(json.dumps(res, indent=1))

    lines2 = ["| Encoder | " + " | ".join(f"stride={k}" for k in STRIDE_GRID) + " |",
              "|" + "|".join(["---"] * (len(STRIDE_GRID) + 1)) + "|"]
    for name, row in sres.items():
        lines2.append(f"| {LABEL[name]} | " + " | ".join(f"{v:.5f}" for v in row) + " |")
    (out / "table_stride.md").write_text(
        "\n".join(lines2) +
        f"\n\n> 平滑运动 + 观测噪声。每步位移 {V_SCALE} m、位置量级 {SIGMA_OFF} m、"
        "噪声 0.01 m —— 与 Track 任务同量级。目标已按每步位移归一化(MSE≈1.0 = 完全学不会)。"
        "\n> 相隔 k 帧估计速度的"
        "\n> 噪声为 sqrt(2)*sigma/k，故**信噪比正比于帧间隔**。这解释了为什么"
        "\n> 62.5 Hz、相邻帧相对差异仅 ~1% 的 Track 任务里，stride=1 的窗口学不动。\n")

    st.apply_publication_style()
    fig, axes = st.create_subplots(1, 1, figsize=(7, 4.8))
    ax = axes[0]
    for name, row in sres.items():
        ax.plot(STRIDE_GRID, row, "o-", label=LABEL[name], color=COLOR[name],
                linewidth=2.4, markersize=6)
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlabel("Window stride (frame skip)")
    ax.set_ylabel("MSE of estimated velocity")
    ax.set_title("Why the history window must be strided", fontsize=14)
    ax.legend(fontsize=10); ax.grid(alpha=.3, which="both", linestyle=":")
    ax.set_axisbelow(True)
    st.finalize_figure(fig, out / "fig0b_stride")

    print("\n产出:", *[f"  {p}" for p in sorted(out.glob("*probe*")) + sorted(out.glob("*stride*"))], sep="\n")


if __name__ == "__main__":
    main()

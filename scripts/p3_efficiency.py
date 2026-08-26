"""论文③ 效率实验 (C2)：Mamba vs Transformer 的推理代价随上下文长度 L 的标度。

两种部署模式（这是全文效率主张的支撑，必须都报）：

  window   每个控制步重新前向整个长度 L 的窗口。
           Transformer 只能这样做 —— 注意力没有可增量更新的固定状态。 O(L²·d)
  stream   携带固定大小的递归状态，每步只前向 1 帧。
           只有 Mamba/GRU 能这样做。                                    O(1)

AUV 机载算力受限且要求实时控制，`stream` 列才是真实部署代价。

用法:
    python p3_efficiency.py [--out outputs_p3/report] [--batch 1] [--reps 100]

产出: table_efficiency.md/.csv, fig3_efficiency.pdf/png, efficiency.json
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import aei_style as st
from marinegym.learning.modules.encoders import build_encoder

CTX_GRID = [1, 4, 8, 16, 32, 64, 128, 256]
ARMS = {
    "mlp":         {"name": "mlp"},          # 单帧基线，窗口恒为 1
    "stack":       {"name": "stack"},
    "gru":         {"name": "gru", "d_model": 128, "n_layers": 1},
    "transformer": {"name": "transformer", "d_model": 128, "n_layers": 2, "n_heads": 8},
    "mamba":       {"name": "mamba", "d_model": 128, "n_layers": 1, "d_state": 16},
}
LABEL = {"mlp": "MLP (current state only)", "stack": "Frame-stack", "gru": "GRU",
         "transformer": "Transformer (DTQN-style)", "mamba": "Mamba (ours)"}
COLOR = {"mlp": st.PALETTE["neutral"], "stack": st.PALETTE["green_3"], "gru": st.PALETTE["teal"],
         "transformer": st.PALETTE["blue_secondary"], "mamba": st.PALETTE["red_strong"]}


def bench(fn, reps, warmup=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps * 1e3      # ms


@torch.no_grad()
def bench_window(arm, L, D, B, reps, dev):
    """每步重新前向整个窗口（Transformer 的唯一选项）。"""
    enc = build_encoder(ARMS[arm]).to(dev).eval()
    if arm == "mlp":
        L = 1                       # 单帧臂的代价与上下文长度无关
    x = torch.randn(B, 1, L, D, device=dev)
    enc(x)                                    # 具体化 LazyLinear
    torch.cuda.reset_peak_memory_stats(dev)
    ms = bench(lambda: enc(x), reps)
    mem = torch.cuda.max_memory_allocated(dev) / 2**20
    n = sum(p.numel() for p in enc.parameters())
    del enc
    torch.cuda.empty_cache()
    return ms, mem, n


@torch.no_grad()
def bench_stream_mamba(L, D, B, reps, dev):
    """Mamba 递归模式：携带 (conv_state, ssm_state)，每步只前向 1 帧。

    状态大小与 L 无关 —— 这正是 O(1) 的来源。L 只用来设置 max_seqlen。
    """
    from mamba_ssm.modules.mamba_simple import Mamba
    from mamba_ssm.utils.generation import InferenceParams
    m = Mamba(d_model=128, d_state=16, d_conv=4, expand=2, layer_idx=0).to(dev).eval()
    proj = torch.nn.Linear(D, 128).to(dev).eval()
    x1 = torch.randn(B, 1, D, device=dev)
    ip = InferenceParams(max_seqlen=L + 1, max_batch_size=B)
    ip.key_value_memory_dict[0] = m.allocate_inference_cache(B, L + 1)
    ip.seqlen_offset = 1
    step = lambda: m(proj(x1), inference_params=ip)
    step()
    torch.cuda.reset_peak_memory_stats(dev)
    ms = bench(step, reps)
    mem = torch.cuda.max_memory_allocated(dev) / 2**20
    del m, proj
    torch.cuda.empty_cache()
    return ms, mem


@torch.no_grad()
def bench_stream_gru(L, D, B, reps, dev):
    """GRU 递归模式：携带 h，每步 1 帧。同样 O(1)，作为对照。"""
    inp = torch.nn.Linear(D, 128).to(dev).eval()
    cell = torch.nn.GRUCell(128, 128).to(dev).eval()
    x1 = torch.randn(B, D, device=dev)
    h = torch.zeros(B, 128, device=dev)
    def step():
        nonlocal h
        h = cell(torch.relu(inp(x1)), h)
    step()
    torch.cuda.reset_peak_memory_stats(dev)
    ms = bench(step, reps)
    mem = torch.cuda.max_memory_allocated(dev) / 2**20
    del inp, cell
    torch.cuda.empty_cache()
    return ms, mem


def main():
    ap = argparse.ArgumentParser()
    here = Path(__file__).parent
    ap.add_argument("--out", default=str(here / "outputs_p3" / "report"))
    ap.add_argument("--obs-dim", type=int, default=43)
    ap.add_argument("--batch", type=int, default=1,
                    help="部署场景是单载具在线推理，故默认 B=1")
    ap.add_argument("--reps", type=int, default=200)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    D, B = a.obs_dim, a.batch

    res = {"meta": {"gpu": torch.cuda.get_device_name(0), "obs_dim": D,
                    "batch": B, "reps": a.reps,
                    "torch": torch.__version__}, "window": {}, "stream": {}}

    print(f"GPU={res['meta']['gpu']}  obs_dim={D}  batch={B}  reps={a.reps}\n")
    print("=== window 模式：每步重新前向整个窗口 (ms) ===")
    print(f"{'L':>5} " + " ".join(f"{LABEL[k].split(' ')[0]:>13}" for k in ARMS))
    for L in CTX_GRID:
        row = {}
        for arm in ARMS:
            try:
                ms, mem, n = bench_window(arm, L, D, B, a.reps, dev)
                row[arm] = {"ms": ms, "mem_MiB": mem, "params": n}
            except Exception as e:
                row[arm] = {"error": str(e)[:80]}
        res["window"][L] = row
        print(f"{L:>5} " + " ".join(
            f"{row[k].get('ms', float('nan')):>13.4f}" for k in ARMS))

    print("\n=== stream 模式：携带递归状态，每步 1 帧 (ms) ===")
    print(f"{'L':>5} {'GRU':>13} {'Mamba':>13}")
    for L in CTX_GRID:
        row = {}
        try:
            ms, mem = bench_stream_gru(L, D, B, a.reps, dev)
            row["gru"] = {"ms": ms, "mem_MiB": mem}
        except Exception as e:
            row["gru"] = {"error": str(e)[:80]}
        try:
            ms, mem = bench_stream_mamba(L, D, B, a.reps, dev)
            row["mamba"] = {"ms": ms, "mem_MiB": mem}
        except Exception as e:
            row["mamba"] = {"error": str(e)[:80]}
        res["stream"][L] = row
        print(f"{L:>5} {row['gru'].get('ms', float('nan')):>13.4f} "
              f"{row['mamba'].get('ms', float('nan')):>13.4f}")

    (out / f"efficiency_b{B}.json").write_text(json.dumps(res, indent=1))

    # ---------------- 表 ----------------
    lines = ["| L | " + " | ".join(f"{LABEL[k]} (window)" for k in ARMS) +
             " | Mamba (stream) | GRU (stream) |",
             "|" + "|".join(["---"] * (len(ARMS) + 3)) + "|"]
    csvl = ["L," + ",".join(f"{k}_window_ms" for k in ARMS) +
            ",mamba_stream_ms,gru_stream_ms"]
    for L in CTX_GRID:
        w = res["window"][L]; s = res["stream"][L]
        cells = [f"{w[k].get('ms', float('nan')):.3f}" for k in ARMS]
        lines.append(f"| {L} | " + " | ".join(cells) +
                     f" | **{s['mamba'].get('ms', float('nan')):.3f}** "
                     f"| {s['gru'].get('ms', float('nan')):.3f} |")
        csvl.append(f"{L}," + ",".join(cells) +
                    f",{s['mamba'].get('ms', float('nan')):.4f}"
                    f",{s['gru'].get('ms', float('nan')):.4f}")
    (out / f"table_efficiency_b{B}.md").write_text(
        "\n".join(lines) +
        "\n\n> window = 每个控制步重新前向整个窗口（Transformer 的唯一选项）；"
        "\n> stream = 携带固定大小递归状态，每步只前向 1 帧（仅 Mamba/GRU 可行）。"
        f"\n> {res['meta']['gpu']}, batch={B}, obs_dim={D}, {a.reps} 次平均。\n")
    (out / f"table_efficiency_b{B}.csv").write_text("\n".join(csvl) + "\n")

    # ---------------- 图 ----------------
    st.apply_publication_style()
    fig, axes = st.create_subplots(1, 2, figsize=(13, 4.8))
    ax = axes[0]
    for arm in ARMS:
        ys = [res["window"][L][arm].get("ms", np.nan) for L in CTX_GRID]
        ax.plot(CTX_GRID, ys, "o-", label=LABEL[arm] + " (window)",
                color=COLOR[arm], linewidth=2.4, markersize=6)
    ys = [res["stream"][L]["mamba"].get("ms", np.nan) for L in CTX_GRID]
    ax.plot(CTX_GRID, ys, "s--", label="Mamba (ours, streaming)",
            color=COLOR["mamba"], linewidth=2.8, markersize=7)
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlabel("Context length L"); ax.set_ylabel("Latency per control step (ms)")
    ax.set_title("Inference cost vs context length", fontsize=14)
    ax.legend(fontsize=9, loc="upper left"); ax.grid(alpha=.3, which="both", linestyle=":")
    ax.set_axisbelow(True)

    ax = axes[1]
    for arm in ARMS:
        ys = [res["window"][L][arm].get("params", np.nan) for L in CTX_GRID]
        ax.plot(CTX_GRID, ys, "o-", label=LABEL[arm], color=COLOR[arm],
                linewidth=2.4, markersize=6)
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlabel("Context length L"); ax.set_ylabel("Encoder parameters")
    ax.set_title("Parameter count vs context length", fontsize=14)
    ax.legend(fontsize=9, loc="upper left"); ax.grid(alpha=.3, which="both", linestyle=":")
    ax.set_axisbelow(True)
    st.finalize_figure(fig, out / f"fig3_efficiency_b{B}")

    print("\n产出:", *[f"  {p}" for p in sorted(out.glob("*efficiency*"))], sep="\n")
    print(*[f"  {p}" for p in sorted(out.glob("fig3*"))], sep="\n")


if __name__ == "__main__":
    main()

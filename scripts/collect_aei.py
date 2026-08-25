#!/usr/bin/env python
"""AEI: 把训练 wandb 历史 + 评测日志汇总成 tidy CSV，供出图与写作直接使用。"""
import glob, json, os, re, sys, csv
from collections import defaultdict

ROOT = "/home/jovyan/MarineGym/scripts"
OUT  = os.path.join(ROOT, "outputs_aei", "data")
os.makedirs(OUT, exist_ok=True)

# ---------- 1. 训练曲线（wandb offline） ----------
def read_wandb(run_dir):
    from wandb.sdk.internal import datastore
    from wandb.proto import wandb_internal_pb2 as pb
    fs = glob.glob(os.path.join(run_dir, "*.wandb"))
    if not fs: return []
    ds = datastore.DataStore(); ds.open_for_scan(fs[0])
    rows = []
    while True:
        try: data = ds.scan_data()
        except Exception: break
        if data is None: break
        rec = pb.Record(); rec.ParseFromString(data)
        if rec.WhichOneof("record_type") != "history": continue
        d = {}
        for it in rec.history.item:
            k = ".".join(it.nested_key) if it.nested_key else it.key
            try: d[k] = json.loads(it.value_json)
            except Exception: d[k] = it.value_json
        rows.append(d)
    return rows

# ---------- 2. 评测日志 ----------
def read_eval(path):
    d, eps = {}, None
    for line in open(path, encoding="utf-8", errors="replace"):
        m = re.match(r"\s*stats\.(\w+):\s*([-\d.]+)", line)
        if m: d[m.group(1)] = float(m.group(2))
        m2 = re.search(r"EVAL-ONLY RESULTS \(episodes=(\d+)\)", line)
        if m2: eps = int(m2.group(1))
    if eps is not None: d["episodes"] = eps
    return d

def main():
    # --- 训练曲线 ---
    # 内化 C 的两次训练不属于本方法，其曲线也不收进数据集
    excl = set()
    for lbl in ("r1d_fixed_C", "r1e_adaptive_C"):
        fp = os.path.join(ROOT, "outputs_aei", f"{lbl}.log")
        if os.path.exists(fp):
            m = re.search(r"offline-run-([0-9_]+)-", open(fp, errors="replace").read())
            if m: excl.add(m.group(1))
    tr_rows = []
    for run in sorted(glob.glob(os.path.join(ROOT, "wandb", "offline-run-*"))):
        if any(e in run for e in excl): continue
        rows = read_wandb(run)
        if not rows: continue
        cum = 0.0
        for r in rows:
            if "train/stats.collision" not in r: continue
            cum += float(r["train/stats.collision"])
            tr_rows.append(dict(run=os.path.basename(run), step=r.get("_step"),
                                env_frames=r.get("env_frames"),
                                collision=r["train/stats.collision"], cum_violation=cum,
                                ret=r.get("train/stats.return"),
                                filter_activation=r.get("train/stats.filter_activation"),
                                min_dist=r.get("train/stats.min_obstacle_dist"),
                                power_W=r.get("train/stats.avg_power_W")))
    if tr_rows:
        with open(os.path.join(OUT, "training_curves.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(tr_rows[0].keys())); w.writeheader(); w.writerows(tr_rows)
        print("training_curves.csv:", len(tr_rows), "rows,",
              len(set(r['run'] for r in tr_rows)), "runs")

    # --- 评测 ---
    ev_rows = []
    for p in sorted(glob.glob(os.path.join(ROOT, "outputs_aei", "eval", "*.log"))):
        base = os.path.basename(p)[:-4]
        m = re.match(r"(.+)_(nominal|ood|hard)_s(\d+)$", base)
        if not m: continue
        # 内化 C 不属于本方法（A 风险门控 + B MPPI 滤波 + soft blend），不进数据集
        if m.group(1) in ("r1d_fixed_C", "r1e_adaptive_C"): continue
        d = read_eval(p)
        if not d: continue
        d.update(model=m.group(1), scene=m.group(2), seed=int(m.group(3)))
        ev_rows.append(d)
    if ev_rows:
        cols = sorted({k for r in ev_rows for k in r})
        cols = ["model","scene","seed"] + [c for c in cols if c not in ("model","scene","seed")]
        with open(os.path.join(OUT, "eval_raw.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(ev_rows)
        print("eval_raw.csv:", len(ev_rows), "rows,",
              len(set((r['model'],r['scene']) for r in ev_rows)), "model×scene")
    else:
        print("eval_raw.csv: 暂无评测数据")

if __name__ == "__main__":
    main()

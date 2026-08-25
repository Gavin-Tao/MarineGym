#!/usr/bin/env python3
"""把 outputs_flow/eval/*.log 里的 EVAL-ONLY RESULTS 解析成一张长表 CSV。

文件名编码实验坐标：`<cell>__<scene>__s<seed>.log`。
只解析日志，不做任何数值加工 —— 论文里的数字必须能一路追回到具体某个 log。
"""
import argparse
import csv
import re
import sys
from pathlib import Path

NAME_RE = re.compile(r"^(?P<cell>[^_]+(?:_[^_]+)*?)__(?P<scene>[a-z]+)__s(?P<seed>\d+)$")
KV_RE = re.compile(r"^\s{2,}(?P<k>[\w.]+):\s*(?P<v>-?[\d.]+(?:[eE][-+]?\d+)?)\s*$")
EP_RE = re.compile(r"EVAL-ONLY RESULTS \(episodes=(\d+)\)")


def parse_log(p: Path):
    txt = p.read_text(errors="replace")
    i = txt.rfind("=== EVAL-ONLY RESULTS")
    if i < 0:
        return None
    tail = txt[i:]
    m = EP_RE.search(tail)
    out = {"episodes": int(m.group(1)) if m else -1}
    for line in tail.splitlines()[1:]:
        mm = KV_RE.match(line)
        if mm:
            out[mm.group("k")] = float(mm.group("v"))
        elif line.strip() and not line.startswith(" "):
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/home/jovyan/MarineGym-flow/scripts/outputs_flow/eval")
    ap.add_argument("--out", default="/home/jovyan/MarineGym-flow/scripts/outputs_flow/data/eval_raw.csv")
    a = ap.parse_args()

    rows, skipped = [], []
    for p in sorted(Path(a.dir).glob("*.log")):
        m = NAME_RE.match(p.stem)
        if not m:
            skipped.append((p.name, "文件名不符合 <cell>__<scene>__s<seed>"))
            continue
        d = parse_log(p)
        if d is None:
            skipped.append((p.name, "没有 EVAL-ONLY RESULTS（未跑完/崩了）"))
            continue
        d.update(cell=m.group("cell"), scene=m.group("scene"), seed=int(m.group("seed")), log=p.name)
        rows.append(d)

    if not rows:
        print("没有可解析的结果", file=sys.stderr)
        for n, why in skipped:
            print(f"  skip {n}: {why}", file=sys.stderr)
        return 1

    keys = ["cell", "scene", "seed", "episodes"]
    keys += sorted({k for r in rows for k in r} - set(keys) - {"log"}) + ["log"]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"写出 {len(rows)} 行 → {a.out}")
    # 未完成的必须显式报出来，否则汇总表会"看起来完整"实则缺格
    for n, why in skipped:
        print(f"  ⚠ 跳过 {n}: {why}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

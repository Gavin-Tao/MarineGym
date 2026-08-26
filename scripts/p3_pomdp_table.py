"""论文③：POMDP 机制筛选表。

系统性回答一个问题：**AUV 轨迹跟踪任务在什么条件下才真正需要历史？**

方法：固定策略为单帧 MLP（无历史），逐一施加各种部分可观测机制，
看 `episode_len` 相对同任务无 POMDP 基准掉多少。
掉得越多 ⇒ 该机制造成的信息缺失越致命 ⇒ 序列模型的补偿空间越大。

只有当某个机制确实让单帧策略明显变差时，"比较序列架构"才是有意义的实验。

用法: python p3_pomdp_table.py [--dirs outputs_p3/screen2 outputs_p3/screen3 ...]
"""
import argparse
import json
from pathlib import Path

# 条件名 → (人类可读描述, 该条件所属的任务基准 key)
COND_INFO = {
    # 原始快速轨迹（能力受限，基准 57.86）
    "drift":        ("Unobserved thruster-gain drift  U(0.5,1.0)", "fast"),
    "drift_oracle": ("…same, but gain given to the policy (oracle)", "fast"),
    "delay10":      ("Observation delay 160 ms", "fast"),
    "pomdp_d10":    ("Delay 160 ms + no DVL/gyro", "fast"),
    "noff":         ("No future waypoints, no time encoding", "fast"),
    # 可跟踪任务（slow2，基准 s2）
    "s2":           ("— (no degradation, baseline)", "s2"),
    "s2_novel":     ("No DVL, no rate gyro", "s2"),
    "s2_d20":       ("Observation delay 320 ms", "s2"),
    "s2_sparse":    ("USBL fix every 10 steps (6.2 Hz)", "s2"),
    "s2_noise":     ("Observation noise σ=0.02", "s2"),
    "s2_noff":      ("No future waypoints, no time encoding", "s2"),
    "s2_noff_v":    ("…and no DVL/gyro", "s2"),
    "s2_sp60":      ("USBL fix every 60 steps (1 Hz, realistic)", "s2"),
    "s2_sp60_v":    ("USBL 1 Hz + no DVL/gyro (dead reckoning)", "s2"),
    "s2_sp125_v":   ("USBL 0.5 Hz + no DVL/gyro", "s2"),
    "s2_hard":      ("No DVL/gyro + delay + sparse USBL", "s2"),
    "s2_wave":      ("Unobserved periodic disturbance A=0.3", "s2"),
    "s2_wave5":     ("Unobserved periodic disturbance A=0.5", "s2"),
    "s2_wavefast":  ("Unobserved periodic disturbance A=0.4, high-freq", "s2"),
}
# 每个任务基准的参照值（无 POMDP 时的 episode_len）
BASE_LABEL = {"fast": "original reference speed (capability-limited)",
              "s2": "slowed reference (trackable)"}


def load(dirs):
    out = {}
    for d in dirs:
        for f in sorted(Path(d).glob("*.json")):
            parts = f.stem.split("__")
            if len(parts) != 3:
                continue
            cond, arm, _ = parts
            if arm != "mlp":
                continue
            r = json.loads(f.read_text())
            if r.get("diverged"):
                continue
            ev = {k.replace("eval/stats.stats.", "eval/stats."): v
                  for k, v in r["eval"].items()}
            g = lambda k: ev.get("eval/stats." + k, {})
            out[cond] = dict(
                el=g("episode_len").get("mean"), el_sd=g("episode_len").get("std", 0),
                ret=g("return").get("mean"), te=g("tracking_err_mean_m").get("mean"),
                n=g("episode_len").get("n"), file=f.name)
    return out


def main():
    ap = argparse.ArgumentParser()
    here = Path(__file__).parent
    ap.add_argument("--dirs", nargs="+",
                    default=[str(here / "outputs_p3" / d)
                             for d in ("main", "feas", "screen2", "screen3")])
    ap.add_argument("--out", default=str(here / "outputs_p3" / "report" / "table_pomdp_screen.md"))
    a = ap.parse_args()
    R = load(a.dirs)
    if not R:
        print("没有可用结果"); return

    bases = {"fast": R.get("drift", {}).get("el"), "s2": R.get("s2", {}).get("el")}
    lines = ["# POMDP 机制筛选：AUV 轨迹跟踪在什么条件下才需要历史？", "",
             "策略固定为**单帧 MLP**（无历史）。逐一施加部分可观测机制，看 `episode_len`",
             "相对**同任务**无 POMDP 基准掉多少。掉得越多 ⇒ 信息缺失越致命 ⇒",
             "序列模型的补偿空间越大。只有明显掉下去的条件，才值得用来比较序列架构。", ""]

    for bkey in ("s2", "fast"):
        base = bases.get(bkey)
        rows = [(c, v) for c, v in R.items()
                if COND_INFO.get(c, (None, None))[1] == bkey and v.get("el")]
        if not rows:
            continue
        rows.sort(key=lambda kv: kv[1]["el"])
        lines += [f"## 任务设定：{BASE_LABEL[bkey]}"
                  + (f"（基准 episode_len = {base:.1f}）" if base else ""), "",
                  "| 部分可观测机制 | episode_len ↑ | 相对基准 | Return ↑ | 判定 |",
                  "|---|---|---:|---|---|"]
        for c, v in rows:
            desc = COND_INFO.get(c, (c, bkey))[0]
            d = (v["el"] - base) / base * 100 if base else float("nan")
            # 判定阈值：跌幅 >5% 才算"有效"（评测 std 约 ±0.5%）
            verdict = ("**有效**" if d < -5 else
                       "边缘" if d < -2 else "无效（策略无需历史）")
            if c in ("s2", "drift"):
                verdict = "基准"
            lines.append(f"| `{c}` — {desc} | {v['el']:.1f} ± {v['el_sd']:.2f} | "
                         f"{d:+.1f}% | {v['ret']:.1f} | {verdict} |")
        lines.append("")

    lines += [
        "## 结论", "",
        "> 误差棒是跨评测 seed 的 std（约 ±0.5%），故 |相对基准| < 2% 视为无差异。", "",
        "**只要目标位置持续可见，「追当前目标点」就是无记忆最优策略。**",
        "这解释了为何屏蔽速度、加观测延迟、稀疏定位、切断前馈参考、执行器增益漂移",
        "等机制都测不出差异 —— 它们只是让策略「看得少一点」，而比例控制依然够用。", "",
        "真正需要历史的条件必须满足其一：", "",
        "1. **目标位置不再持续可见**（真实 USBL 速率 0.5–1 Hz + 无 DVL ⇒ 必须航位推算）；",
        "2. **存在时变、有相位的外部扰动**（涌浪诱导力 ⇒ 反应式控制永远滞后半个周期，",
        "   要抵消必须从历史估计 (A, ω, φ) 做前馈 —— 内模原理）。", "",
        "> 另外必须注意：在**原始参考速度**下任务是**能力受限**的",
        "> （载具只跑出参考速度的 51%，episode 长度由运动学锁死在 ~58 步），",
        "> 此时任何机制都测不出差异。比较策略架构前必须先把参考速度降到载具能力之内。",
    ]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("\n写出:", a.out)


if __name__ == "__main__":
    main()

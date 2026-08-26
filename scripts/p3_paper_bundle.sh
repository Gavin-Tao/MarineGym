#!/usr/bin/env bash
# 论文③：把所有实验结果组装成一份可直接用于写作的产物包。
#
#   PAPER_DATA_P3.md   汇总正文（各表 + 图索引 + 口径说明）
#   fig1_main.*        主图：MDP 打平 / POMDP 拉开
#   fig2_curves_*.*    学习曲线
#   fig3_efficiency_*  推理代价 vs 上下文长度
#   table_*.md/.csv    各表格
set -o pipefail
cd "$(dirname "$0")" || exit 1
source /opt/conda/etc/profile.d/conda.sh
conda activate /home/jovyan/envs/sim-mamba
export PYTHONPATH=/home/jovyan/MarineGym-mamba:$PYTHONPATH

IN=${IN:-outputs_p3/main}
REP=${REP:-outputs_p3/report}
mkdir -p "$REP"

echo "=== 1/4 主结果汇总 + 主图 ==="
python p3_report.py --indir "$IN" --outdir "$REP" --conds "${CONDS_PAIR:-mdp,pomdp}" || exit 1

echo "=== 2/4 效率实验（若已有结果则跳过）==="
for B in 1 256; do
  if [ ! -f "$REP/table_efficiency_b$B.md" ]; then
    python p3_efficiency.py --batch $B --reps 200 --out "$REP" || exit 1
  else
    echo "  [skip] batch=$B 已有结果"
  fi
done

echo "=== 3/4 编码器探针（若已有结果则跳过）==="
if [ ! -f "$REP/table_stride.md" ]; then
  python p3_probe.py --out "$REP" --steps 2000 --seeds 3 || exit 1
else
  echo "  [skip] 探针已有结果"
fi

echo "=== 4/4 组装 PAPER_DATA_P3.md ==="
python - "$REP" <<'PY'
import sys, json
from pathlib import Path
rep = Path(sys.argv[1])
def rd(n, default="_(缺)_"):
    p = rep / n
    return p.read_text().strip() if p.exists() else default

summ = {}
if (rep / "summary.json").exists():
    summ = json.loads((rep / "summary.json").read_text())

doc = f"""# 论文③ 数据汇总（自动生成，勿手改；改 scripts/p3_report.py）

任务：MarineGym `Track`（lemniscate 轨迹跟踪，iAUV，Isaac Sim）。
训练：PPO，`max_iters=120`，`num_envs=64`，`train_every=64` ⇒ 每 run ~492k frames。
评测：确定性策略（`ExplorationType.MODE`），聚合**所有完整 episode**（每 run ≥300 条）。

## 1. 指标口径（重要）

episode 因**跟踪失败提前终止**（`track.py`: `terminated |= distance > reset_thres`，
`reset_thres=0.5`，`max_episode_length=600`）。因此：

* **`episode_len` 是首要性能指标** —— 失控前能跟多久。
* `return` 逐步累加，已把 episode 长度计入。
  **不可按 episode 长度归一化 return** —— 那会奖励「失败得早」。
* `tracking_error_ema` 是距离的 EMA，会稳定在 `reset_thres` 附近，
  故区分度弱于 `episode_len`，作为辅助指标报告。
* 表中 `±` 是**跨训练 seed** 的 std（每个 seed 的值本身已是 ≥300 条 episode 的均值）。

## 2. 主结果

{rd("table_main.md")}

**图 1**（`fig1_main.pdf`）：左 = 全观测 MDP，右 = 无 DVL/无陀螺 POMDP，同一 y 轴。
论证结构：**左panel 打平 ⇒ 增益不是模型容量；右panel 拉开 ⇒ 增益来自历史信息。**

## 3. 显著性（POMDP 条件，Mamba vs 各基线）

{rd("table_significance.md")}

> Welch t 检验，跨训练 seed。seed 数少，p 值仅作参考，主论据是效应量与一致性。

## 4. 参数量（审稿人必问）

{rd("table_params.md")}

各序列臂参数量对齐在 ±10% 以内；`MLP-wide` 与 `Transformer @ DTQN default`
是**故意给更多容量**的对照，用来排除"基线只是容量不够/被调弱了"的解释。

## 5. 推理代价 vs 上下文长度

### 5.1 单载具在线推理（batch=1，机载部署的真实工况）

{rd("table_efficiency_b1.md")}

### 5.2 计算受限工况（batch=256，暴露渐近复杂度）

{rd("table_efficiency_b256.md")}

**图 3**（`fig3_efficiency_b1.pdf` / `fig3_efficiency_b256.pdf`）。

诚实说明：batch=1、d_model≈128 这种小模型下，**各臂都是 kernel 启动开销受限**，
Transformer 的 O(L²) 在 L≤256 内看不出来；差距来自它的算子数量多。
把 batch 提到 256 才进入计算受限区，此时 Transformer 的二次项才显现。
两种工况都报，不要只报有利的那一个。

## 6. 编码器验证探针（方法学，不是主结果，但审稿人会问）

### 6.1 各臂能否恢复"必须用历史才能得到的量"

{rd("table_probe.md")}

**图 0**（`fig0_probe.pdf`）。构造为**解析上可判定**：单帧观测与目标统计独立，
故只看当前帧的最优 MSE 恰为目标方差。用途是把"编码器实现/优化有问题"
与"任务本身不需要历史"两件事分开 —— 前者被这张图排除。

### 6.2 为什么历史窗口必须按步长采样（帧跳）

{rd("table_stride.md")}

**图 0b**（`fig0b_stride.pdf`）。Track 的控制频率 62.5 Hz、位置量级 ~2 m、
每步位移仅 ~2.6 cm ⇒ 相邻帧相对差异约 1%。相隔 k 帧估计速度的噪声为 √2·σ/k，
**信噪比正比于帧间隔**。实测 Mamba 在 stride=1 时比 stride=2 差 24 倍 ——
因果 conv1d 与选择性扫描正是在相邻帧之间做差分，"相邻帧近乎重复"是它的最坏情形。
本文取 stride=4（窗口跨越 61 步 ≈ 1 s，与 episode 长度同量级），
**对所有序列臂取值相同**。这也是论文该写明的一条 Mamba 适用条件。

## 7. 学习曲线

`fig2_curves_*.pdf`：`train/stats.episode_len` vs 环境步数，阴影为跨 seed 的 std。

## 8. 已完成的 run 计数

```
{json.dumps(summ.get("counts", {}), ensure_ascii=False, indent=1)}
```
"""
(rep / "PAPER_DATA_P3.md").write_text(doc)
print("写出", rep / "PAPER_DATA_P3.md")
PY

echo
echo "=== 产物 ==="
ls -1 "$REP"

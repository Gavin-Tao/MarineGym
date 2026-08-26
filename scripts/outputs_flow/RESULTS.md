# 论文② 结果表

> 由 `flow_collect.py` + `flow_report.py` 从评测日志直接生成，未手工誊写。
> 相关文档：`K4_FINDINGS.md`（观测器判决与机理）、`PERF_NOTES.md`（成本结构）、
> `RERUN_TODO.md`（待补跑项）。

### K1 为什么不能直接用 MPC

- 无扰动下 MPC-only(λ≡1) 的跟踪 RMSE = **0.927 m**，PPO = **0.341 m**（2.7×）。
- 也就是说 MPC 单独用时**跟不准**；它安全但精度差，PPO 精度好但不安全。
  λ 混合的意义正来自这个互补 —— 若 MPC 跟踪不劣于 PPO，方法即自我否定。

### K3 预测门控 vs 反应式门控

| 场景 | PPO | Reactive+soft | **Ours** |
|---|---|---|---|
| calm | 0.0530 | 0.0530 | **0.0448** |
| nominal | 0.2132 | 0.2148 | **0.2029** |
| strong | 0.5751 | 0.6469 | **0.6051** |
| fast | 0.6113 | 0.6068 | **0.6184** |

反应式门控若与 PPO 打平（做了修正却没换来安全收益），说明它介入得太晚或全是误触发 —— 这正是『必须预测』这一论点的实验支撑。

### 读数须知

- `wall_violation` 是**每 episode 是否触壁**的 0/1，跨 episode 取均值 = 违约率。
  侧壁不终止 episode，因此各方法**曝光量相同**、违约率可比。
- `wall_viol_frac` 是触壁步数占比的 EMA；`stats.tracking_error` 是未归一化的累计量，
  **看 `tracking_error_ema`**。
- 统计功效：每格约 80 个 episode 时，只能分辨 ~0.2 量级的违约率差（strong 档够用）；
  nominal 档若要分辨 ~0.10 的差，每格需要约 250 个 episode。CI 跨 0 的差异不要下结论。
- 各格**共用同一条策略**，且阵风与 MPPI 各有独立 RNG，因此各格面对的 episode 与扰动序列完全一致。

## 场景 calm — No gust

| method | seeds | Wall-violation rate ↓ | Violating-step fraction ↓ | Cumulative intrusion (m·s) ↓ | Min wall clearance (m) ↑ | Tracking RMSE (m) ↓ | Mean takeover $\lambda$ - | Episode length (steps) - |
|---|---|---|---|---|---|---|---|---|
| PPO | 1 | 0.0530 | 0.0530 | 0.0242 | 0.7206 | 0.3408 | 0.0000 | 575.9015 |
| MPC-only ($\lambda\equiv1$) | 1 | 0.0000 | 0.0000 | 0.0000 | 0.7486 | 0.9269 | 1.0000 | 598.0000 |
| Reactive + soft | 1 | 0.0530 | 0.0530 | 0.0266 | 0.7210 | 0.3413 | 0.0557 | 575.4318 |
| Ours (pred. + soft) | 1 | 0.0448 | 0.0440 | 0.0238 | 0.7206 | 0.3900 | 0.0652 | 567.7687 |

## 场景 nominal — Nominal (train dist.)

| method | seeds | Wall-violation rate ↓ | Violating-step fraction ↓ | Cumulative intrusion (m·s) ↓ | Min wall clearance (m) ↑ | Tracking RMSE (m) ↓ | Mean takeover $\lambda$ - | Episode length (steps) - |
|---|---|---|---|---|---|---|---|---|
| PPO | 1 | 0.2132 | 0.1773 | 0.0956 | 0.3975 | 1.1834 | 0.0000 | 500.3603 |
| MPC-only ($\lambda\equiv1$) | 1 | 0.0000 | 0.0000 | 0.0000 | 0.3558 | 1.2851 | 1.0000 | 598.0000 |
| Reactive + soft | 1 | 0.2148 | 0.1836 | 0.0947 | 0.3783 | 1.2133 | 0.2250 | 500.8148 |
| Ours (pred. + soft) | 1 | 0.2029 | 0.1789 | 0.0991 | 0.3803 | 1.1994 | 0.2573 | 497.0507 |

## 场景 strong — Strong gust (OOD amplitude)

| method | seeds | Wall-violation rate ↓ | Violating-step fraction ↓ | Cumulative intrusion (m·s) ↓ | Min wall clearance (m) ↑ | Tracking RMSE (m) ↓ | Mean takeover $\lambda$ - | Episode length (steps) - |
|---|---|---|---|---|---|---|---|---|
| PPO | 1 | 0.5751 | 0.5140 | 0.2029 | -0.2176 | 2.5708 | 0.0000 | 361.0984 |
| MPC-only ($\lambda\equiv1$) | 1 | 0.9706 | 0.6177 | 0.4526 | -0.4620 | 1.9502 | 1.0000 | 571.9485 |
| Reactive + soft | 1 | 0.6469 | 0.5828 | 0.1752 | -0.2412 | 2.7085 | 0.6900 | 266.2750 |
| Ours (pred. + soft) | 1 | 0.6051 | 0.4788 | 0.1842 | -0.1676 | 2.4867 | 0.6774 | 375.3641 |

## 场景 fast — Strong + fast onset (OOD amp. + ramp)

| method | seeds | Wall-violation rate ↓ | Violating-step fraction ↓ | Cumulative intrusion (m·s) ↓ | Min wall clearance (m) ↑ | Tracking RMSE (m) ↓ | Mean takeover $\lambda$ - | Episode length (steps) - |
|---|---|---|---|---|---|---|---|---|
| PPO | 1 | 0.6113 | 0.5095 | 0.1542 | -0.1748 | 2.6263 | 0.0000 | 293.6820 |
| MPC-only ($\lambda\equiv1$) | 1 | 1.0000 | 0.5723 | 0.3161 | -0.4041 | 1.9069 | 1.0000 | 568.4348 |
| Reactive + soft | 1 | 0.6068 | 0.5036 | 0.1680 | -0.1760 | 2.6273 | 0.6280 | 286.6712 |
| Ours (pred. + soft) | 1 | 0.6184 | 0.4798 | 0.1327 | -0.1414 | 2.6581 | 0.6554 | 290.1590 |

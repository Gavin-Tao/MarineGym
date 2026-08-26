# 论文② 结果表

> 由 `flow_collect.py` + `flow_report.py` 从评测日志直接生成，未手工誊写。
> 相关文档：`K4_FINDINGS.md`（观测器判决与机理）、`PERF_NOTES.md`（成本结构）、
> `RERUN_TODO.md`（待补跑项）。

### K1 为什么不能直接用 MPC

- 无扰动下 MPC-only(λ≡1) 的跟踪 RMSE = **0.927 m**，PPO = **0.341 m**（2.7×）。
- 也就是说 MPC 单独用时**跟不准**；它安全但精度差，PPO 精度好但不安全。
  λ 混合的意义正来自这个互补 —— 若 MPC 跟踪不劣于 PPO，方法即自我否定。

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

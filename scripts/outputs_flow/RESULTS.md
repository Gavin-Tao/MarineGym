# 论文② 结果表

> 由 `flow_collect.py` + `flow_report.py` 从 `outputs_flow/eval/*.log` 直接生成，未手工誊写。

## 场景 nominal — Nominal (train dist.)

| method | seeds | Wall-violation rate ↓ | Violating-step fraction ↓ | Cumulative intrusion (m·s) ↓ | Min wall clearance (m) ↑ | Tracking RMSE (m) ↓ | Mean takeover $\lambda$ - | Episode length (steps) - |
|---|---|---|---|---|---|---|---|---|
| PPO | 1 | 0.0156 | 1.2344 | 0.0027 | 0.6533 | 1.6004 | 0.0000 | 574.8906 |

## 场景 strong — Strong gust (OOD amp.)

| method | seeds | Wall-violation rate ↓ | Violating-step fraction ↓ | Cumulative intrusion (m·s) ↓ | Min wall clearance (m) ↑ | Tracking RMSE (m) ↓ | Mean takeover $\lambda$ - | Episode length (steps) - |
|---|---|---|---|---|---|---|---|---|
| PPO | 1 | 0.6000 | 47.0308 | 0.2210 | -0.0602 | 2.2553 | 0.0000 | 492.8615 |

## 场景 fast — Fast onset (OOD ramp)

| method | seeds | Wall-violation rate ↓ | Violating-step fraction ↓ | Cumulative intrusion (m·s) ↓ | Min wall clearance (m) ↑ | Tracking RMSE (m) ↓ | Mean takeover $\lambda$ - | Episode length (steps) - |
|---|---|---|---|---|---|---|---|---|
| PPO | 1 | 0.0156 | 0.9062 | 0.0012 | 0.7092 | 1.5956 | 0.0000 | 570.8906 |

| Condition | Encoder | L | Params | seeds | episode_len ↑ | return ↑ | tracking_err ↓ |
|---|---|---|---|---|---|---|---|
| Thruster gain observed (oracle) | MLP (current state only) | 1 | 144,390 | 1 | 76.431 ± 3.138 | 64.288 ± 2.503 | 0.452 ± 0.002 |
| Thruster gain observed (oracle) | Frame-stack | 16 | 182,918 | 1 | 79.900 ± 4.030 | 66.501 ± 3.170 | 0.455 ± 0.002 |
| Thruster gain observed (oracle) | Mamba (ours, 1L) | 16 | 156,550 | 1 | 80.961 ± 3.334 | 68.540 ± 2.631 | 0.451 ± 0.002 |
| Thruster gain hidden (POMDP) | MLP (current state only) | 1 | 143,622 | 1 | 76.115 ± 3.353 | 64.847 ± 3.029 | 0.449 ± 0.002 |
| Thruster gain hidden (POMDP) | Frame-stack | 16 | 176,774 | 1 | 74.176 ± 3.060 | 62.192 ± 2.407 | 0.450 ± 0.002 |
| Thruster gain hidden (POMDP) | Mamba (ours, 1L) | 16 | 156,166 | 1 | 81.845 ± 4.053 | 69.135 ± 3.301 | 0.450 ± 0.002 |

> `±` = **episode 级标准误** SEM = std/√n（单次训练，评测聚合 数百条 episode）。
> ⚠ 它只刻画评测噪声，**不含训练 run 间方差** —— RL 里后者通常更大。
> 因此显著性只说明"这两个训练出来的策略不同"，不能推断"该方法平均更好"。

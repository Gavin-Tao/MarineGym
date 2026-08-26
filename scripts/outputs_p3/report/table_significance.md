| Comparison (Thruster gain hidden (POMDP)) | Δ episode_len | t | p |
|---|---|---|---|
| Mamba vs MLP (current state only) | +5.73 | 1.09 | 0.2761 n.s. |
| Mamba vs Frame-stack | +7.67 | 1.51 | 0.1311 n.s. |

> Welch t 检验。单次训练时用 episode 级 (mean, std, n)，
> 故检验的是"两个已训练策略的表现分布是否不同"，**不是**"两种方法孰优孰劣"（后者需要多个训练 seed）。

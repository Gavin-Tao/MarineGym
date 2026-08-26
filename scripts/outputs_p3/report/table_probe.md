| Encoder | MSE @ step 0 | MSE @ 500 | Final MSE | Final / target var | Verdict |
|---|---|---|---|---|---|
| MLP (current state only) | 0.4363 | 0.0890 | 0.0879 | 0.977 | **cannot** (≈ predicting the mean) |
| Frame-stack | 0.4267 | 0.0497 | 0.0045 | 0.050 | recovers it |
| GRU | 0.3501 | 0.0067 | 0.0020 | 0.023 | recovers it |
| Transformer (DTQN-style) | 0.3790 | 0.0495 | 0.0008 | 0.009 | recovers it |
| Mamba (ours) | 0.4505 | 0.0061 | 0.0010 | 0.011 | recovers it |

> 随机游走窗口 (L=16, D=37, σ=0.3)，目标 = 最后一步增量。
> 单帧观测与目标统计独立 ⇒ 只看当前帧的最优 MSE = 目标方差 0.0900。
> 2 个种子，Adam lr=0.0005，1200 步。

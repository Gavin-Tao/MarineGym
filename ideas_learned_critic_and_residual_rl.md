# 两个后续论文 idea:Learned Safety Critic 与 Residual RL

> 与主线论文①(A+B:Risk-Gated Predictive Safety Filter)共用 testbed 与资产:
> MarineGym 动态拦截障碍环境(已建)、<2% 精确名义动力学(已建)、批量 MPPI(已建)、PPO 管线。
> 两个 idea 都是**在这些资产上加一层新东西**,增量工程可控。

---

## Idea A:Learned + Model-based 双通道风险评估(Hybrid Risk Assessment)

### 一句话定位
学习型风险评估器(learned safety critic)和模型型风险评估器(我们的组件 A)**失效模式互补**——
learned 在没见过的数据上崩,model-based 在模型不准时崩;把两者融合,得到两边都稳的混合风险监控器。

### 故事(headline)
> *"When to trust the model, when to trust the data: hybrid risk assessment for safe underwater RL."*

三幕结构,每一幕有明确的预期赢家:
1. **In-distribution**:learned ≈ model-based(都行)——公平起点;
2. **OOD 障碍(更快/未见几何)**:learned **崩**(没学过),model-based **稳**(物理不变)→ model 的价值;
3. **模型失准场景(强域随机化:payload 质量扰动、强洋流、附加质量扰动)**:model-based 前滚**不准**(名义模型≠真实),learned **稳**(从真实交互数据学的)→ learned 的价值;
4. **Hybrid(融合)**:三个场景全稳 → 论文的方法贡献。

这个故事的好处:**两边各有赢的场景,hybrid 全赢**——不是"A 打败 B"的单边故事,审稿人无法用"那你为什么不直接用 X"反驳。

### 方法设计
```
                    ┌─ model-based 风险 r_m(组件A:精确模型前滚,已有)
state, action ──┤
                    └─ learned 风险 r_l(safety critic 网络,新)
                            ↓
                  融合:r = max(r_m, r_l)(保守)或不确定性加权
                            ↓
                  门控 B(MPPI filter,已有)+ 风险进观测
```
- **Safety critic 网络**:MLP,输入 (state, action),输出未来 H 步内碰撞概率。
  训练数据用 **hindsight 标注**:rollout 里实际碰撞的时刻,向前回溯 H 步全部标正样本;其余负样本。
  与 PPO 同步在线训练(拿 collector 的数据顺带训),二分类交叉熵。
- **融合**:v1 用 `max`(保守);v2 用各自不确定性加权(critic 用 ensemble/dropout 方差,model 用已知的模型误差界)——v2 是理论加分项。

### 复用 vs 新增
| 已有(复用) | 新增(工程量) |
|---|---|
| 组件A、MPPI、动态障碍环境、PPO | critic 网络+hindsight 标签管线(~200 行);域随机化加强配置(改 yaml);融合逻辑(~30 行) |

### 实验矩阵
| 场景 | PPO | +learned critic | +model A(已有数据) | +hybrid |
|---|---|---|---|---|
| in-dist | ✓ | ✓ | ✓ | ✓ |
| OOD 速度(2.2–2.8) | ✓ | **预期崩** | 稳(已有 S4 数据) | 稳 |
| 强域随机化/洋流(模型失准) | ✓ | 稳 | **预期崩** | 稳 |

指标:沿用主线(coll / cum违约 / minDist / filtAct / detour_ratio)。

### 风险与对策
- critic 学不好(碰撞样本太少)→ 提高障碍难度产生足够正样本;或用接近事件(clearance<0.3)作代理标签。
- "模型失准场景"里 model-based 可能还没崩(名义模型鲁棒性超预期)→ 先跑一个探针实验确认失效点(payload/洋流开多大 model 才崩),再定实验设置。**这是第一步要做的 go/no-go 实验。**

---

## Idea B:精确模型打底的 Residual RL(Residual Policy Learning)

### 一句话定位
不让 PPO 从零学:先放一个**用精确名义模型构造的基础控制器**(能大致跟踪+避障),
PPO 只学**残差修正**,`u = clip(u_base + α·u_residual)`。

### 故事(headline)
> *"Don't learn from scratch: a certified nominal controller as the backbone for underwater RL."*

三个卖点(都可量化):
1. **样本效率**:达到相同 return 的迭代数减少 N 倍(基础控制器从第 1 步就能大致跟踪);
2. **训练全程安全**:早期不乱飞(base 兜底)→ 训练累计违约远低于从零学的 PPO——**与主线论文①的 safe exploration 指标同源,可直接对比两条路线**(filter 保护 vs base 打底,谁的过程更安全/最终更好);
3. **可回退部署**:残差可以随时关掉退回纯 base 控制器(工业友好;残差幅度 α 是显式的信任旋钮)。

加分点(与前作呼应):残差小 → 动作平滑 → **功率更低**——能量指标又能用上(前作的延续性)。

### 方法设计
```
state ──▶ [base 控制器:MPPI-tracking(精确模型,cost=跟踪+避障)] ──u_base──┐
      └─▶ [PPO 残差策略] ──u_res──────────────────────────────────────┴──▶ u = clip(u_base + α·u_res)
```
- **base 控制器**:把已有的 `MPPIExactShield` 的 cost 从"贴近 u_rl+避障"改成"**跟踪参考**+避障"——**~20 行改动**,立即得到一个 standalone 的模型型跟踪控制器(它自己就是一个不用学的 baseline,顺带多一行对比)。
- **PPO 残差**:观测不变,动作解释为残差,α≈0.3 起步。
- 注:base 每步跑 MPPI 有算力开销(训练时),可用 K=16,N=10 的小配置——或者 base 用 PD+模型前馈(更便宜,效果差些,可作消融)。

### 实验矩阵
| 方法 | 看什么 |
|---|---|
| 纯 PPO(已有数据) | baseline |
| base 控制器 alone(不学) | "不学能到多好"的下限锚点 |
| base + residual PPO | **主方法**:收敛速度倍数、cum违约、最终 return/coll |
| 消融:α∈{0.1,0.3,0.5};base 用粗模型 vs 精确模型 | 后者再次证明 <2% 模型的价值(贯穿系列工作的资产) |

### 风险与对策
- residual RL 文献成熟(Johannink 2019 等),**新意靠场景(水下+动态障碍)+精确模型 base+与 filter 路线的正面对比**;单独可能只够 workshop/中档会,与 Idea A 或主线捆绑引用成系列更划算。
- base 太强 → 残差学不到东西(return 无提升)→ 调低 base 的 MPPI 预算制造改进空间,或在 base 不擅长的分布(强扰动)训练。

---

## 优先级建议
1. **主线论文①先收尾**(变体矩阵 + A 信息/门控分离消融 + OOD 广度);
2. **Idea A 第二做**——故事最完整(双向失效+hybrid 全赢),且 go/no-go 探针便宜(一轮实验);
3. **Idea B 第三做**——工程最省(base 控制器 ~20 行改出来),但新意上限较低,适合快出一篇中档。

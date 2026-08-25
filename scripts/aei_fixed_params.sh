# ============================================================================
# AEI 组件消融：唯一冻结的超参集合。所有消融格子必须 source 本文件。
# 每一格只允许覆盖「被消融的那一个开关」，不得改动下面任何数值。
# ============================================================================

# --- 训练 ---
# seed / max_iters 不在冻结集内 —— 它们是每次运行都要指定的量。
# 2026-08-25 曾因把 seed=0 写进这里，导致 10 次评测全跑 seed=0、std=0、检验失真。
FIXED_TRAIN="task=Track algo=ppo task.drone_model.name=BlueROV task.traj_scale_mult=2.5 \
algo.train_every=16 save_interval=40 \
headless=true enable_livestream=false wandb.mode=offline"
TRAIN_ITERS="max_iters=120"

# --- 场景（禁区 + 动态拦截障碍）---
# 注意：障碍速度 speed 不属于冻结集 —— 它是评测时【故意要变】的场景维度
# (nominal=[0.4,0.9] / ood=[2.2,2.8])。2026-08-25 曾因把它写进这里，
# 导致 ood 评测被冻结值反压成 nominal，白跑一轮。场景覆盖必须单独传且置于末位。
FIXED_SCENE="task.keepout.enable=true task.keepout.radius=0.8 \
task.keepout.dynamic.enable=true \
task.keepout.dynamic.intercept_steps=[150,350] task.keepout.dynamic.re_aim_period=200"
SCENE_NOMINAL="task.keepout.dynamic.speed=[0.4,0.9]"
SCENE_OOD="task.keepout.dynamic.speed=[2.2,2.8]"

# --- 组件 A：预测式风险监视器 ---
# threshold 与 mppi.soft_hi 是同一个量 min_clear_pred 上的同一个边界（阶跃 vs 斜坡），
# 因此二者必须同为 0.6，整套只有这一个边界。
FIXED_A="task.keepout.risk.enable=true task.keepout.risk.horizon=15 \
task.keepout.risk.risk_norm=2.0 task.keepout.risk.in_obs=true \
task.keepout.risk.threshold=0.6"

# --- 组件 B：MPPI 精确预测滤波 ---
# mppi.enable 不在冻结集内 —— A-only 消融要把它设为 false，放进来会被反压。
MPPI_ON="task.keepout.mppi.enable=true"
FIXED_B="task.keepout.mppi.exact=true \
task.keepout.mppi.horizon=20 task.keepout.mppi.num_samples=128 \
task.keepout.mppi.noise_sigma=0.4 task.keepout.mppi.temperature=0.05 \
task.keepout.mppi.w_coll=5.0 task.keepout.mppi.w_track=1.0 \
task.keepout.mppi.soft_lo=0.0 task.keepout.mppi.soft_hi=0.6"

# --- 内化：整套实验一律关闭（不属于本方法）---
FIXED_NOC="task.keepout.internalize_weight=0.0 task.keepout.internalize_adaptive=false"

# 本方法的完整配置 = 以上全部 + soft_blend=true
FIXED_ALL="$FIXED_TRAIN $FIXED_SCENE $FIXED_A $FIXED_B $FIXED_NOC"
# 用法：$FIXED_ALL 放前面，每次运行独有的量(seed/场景/消融开关)一律放【最后】
OURS="$FIXED_ALL $MPPI_ON task.keepout.mppi.soft_blend=true"

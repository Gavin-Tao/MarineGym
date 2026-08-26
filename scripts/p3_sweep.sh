#!/usr/bin/env bash
# 论文③ 主实验矩阵：MDP vs POMDP × 各编码器 × seeds
#
# 核心论证结构（两列缺一不可）：
#   MDP 列   全观测 → 各臂应打平        ⇒ 增益不是模型容量带来的
#   POMDP 列 无 DVL/无陀螺 → 拉开差距   ⇒ 增益来自历史信息
#
# 参数量（obs_dim=37, L=16，位置编码按 context_len 分配）：
#   mlp 142,848 | transformer 142,368 | gru 137,472 | mamba 155,136 | stack 176,000
#   —— 全部落在 ±10%，"赢的是架构不是容量"这条质疑被堵住。
#   大容量组另报: mlp_wide 312,576 | transformer_dtqn 436,992 (DTQN 原生默认)
#
# 用法:
#   bash p3_sweep.sh                                   # 全矩阵，4 GPU 并行
#   ARMS="mlp mamba" SEEDS="0" GPUS="0" bash p3_sweep.sh
set -o pipefail
cd "$(dirname "$0")" || exit 1

OUT=${OUT:-/home/jovyan/MarineGym-mamba/scripts/outputs_p3/main}
ARMS=${ARMS:-"mlp stack gru transformer mamba"}
CONDS=${CONDS:-"pomdp mdp"}          # 先跑 POMDP —— 那是主结论所在
SEEDS=${SEEDS:-"0 1 2"}
ITERS=${ITERS:-120}                  # 与论文①② 一致
NENV=${NENV:-64}
CTX=${CTX:-16}
# 窗口采样步长（帧跳）。62.5 Hz 下相邻帧相对差异仅 ~1%，stride=1 的窗口
# 信噪比极差（pilot 实测序列臂学不动）。stride=4 时 L=16 跨越 61 步 ≈ 1 s，
# 与 episode 长度（~76 步）同量级。
STRIDE=${STRIDE:-4}
EVAL_EP=${EVAL_EP:-300}
GPUS=${GPUS:-"0 1 2 3"}              # 每卡一个 job 并行；Isaac 单 job 约 3 GB
mkdir -p "$OUT"

arm_overrides () {
  case "$1" in
    mlp)         echo "algo.encoder.name=mlp task.context_len=1" ;;
    # 容量对照：参数量 2.2 倍，堵住"MLP 只是容量不够"的质疑
    mlp_wide)    echo "algo.encoder.name=mlp task.context_len=1 +algo.encoder.hidden=[384,384,384]" ;;
    stack)       echo "algo.encoder.name=stack task.context_len=$CTX task.context_stride=$STRIDE" ;;
    gru)         echo "algo.encoder.name=gru task.context_len=$CTX task.context_stride=$STRIDE algo.encoder.d_model=128 algo.encoder.n_layers=1" ;;
    # 主基线 = DTQN 原生默认（run.py: --in-embed 128 --heads 8 --layers 2 --ff 4x
    # --pos learned --gate res --identity False）。436,992 参数。
    # 不做参数量对齐 —— 本文要证明的正是"Mamba 用更少参数达到相当精度"。
    transformer) echo "algo.encoder.name=transformer task.context_len=$CTX task.context_stride=$STRIDE algo.encoder.d_model=128 algo.encoder.n_layers=2 algo.encoder.n_heads=8" ;;
    # 额外对照：把 Transformer 缩到与 Mamba 同量级参数(d96×1层, 143,142)。
    # 用来回答"Mamba 赢是不是只因为基线太大/太难训"这条反向质疑。
    transformer_small) echo "algo.encoder.name=transformer task.context_len=$CTX task.context_stride=$STRIDE algo.encoder.d_model=96 algo.encoder.n_layers=1 algo.encoder.n_heads=8" ;;
    # ours：同宽(d128)但只用 1 层 —— 155,392 参数，是 DTQN 基线的 0.36 倍。
    # 故意不做参数量对齐：本文的主张就是"更少参数、更浅、达到相当精度"。
    mamba)       echo "algo.encoder.name=mamba task.context_len=$CTX task.context_stride=$STRIDE algo.encoder.d_model=128 algo.encoder.n_layers=1" ;;
    # 容量对照：对齐 transformer_dtqn(d128×2层, 436,992)
    # 容量扫描：Mamba 一层、逐级变窄，看"最小到多少还能匹配 DTQN 基线"
    mamba_d96)   echo "algo.encoder.name=mamba task.context_len=$CTX task.context_stride=$STRIDE algo.encoder.d_model=96 algo.encoder.n_layers=1" ;;
    # ---- 连续历史窗口(过去 N 个连续时间步，stride=1)----
    # 与默认的 L=16/stride=4 对照：后者隔 4 步采样，扔掉 3/4 的帧。
    # 航位推算需要对推力历史积分，采样会丢掉积分所需的样本。
    # DTQN 原生用的也是连续帧(--context 50)。
    transformer_c64) echo "algo.encoder.name=transformer task.context_len=64 task.context_stride=1 algo.encoder.d_model=128 algo.encoder.n_layers=2 algo.encoder.n_heads=8" ;;
    mamba_c64)       echo "algo.encoder.name=mamba task.context_len=64 task.context_stride=1 algo.encoder.d_model=128 algo.encoder.n_layers=1" ;;
    transformer_c50) echo "algo.encoder.name=transformer task.context_len=50 task.context_stride=1 algo.encoder.d_model=128 algo.encoder.n_layers=2 algo.encoder.n_heads=8" ;;
    mamba_d64)   echo "algo.encoder.name=mamba task.context_len=$CTX task.context_stride=$STRIDE algo.encoder.d_model=64 algo.encoder.n_layers=1" ;;
    # 额外对照：Mamba 放大到与 DTQN 基线同参数量(d128×4层, ~503k)
    mamba_big)   echo "algo.encoder.name=mamba task.context_len=$CTX task.context_stride=$STRIDE algo.encoder.d_model=128 algo.encoder.n_layers=4" ;;
    *) echo "UNKNOWN_ARM" ;;
  esac
}

# POMDP：无 DVL（线速度）+ 无速率陀螺（角速度）= 仅位姿可测（USBL + AHRS）。
# 控制论上最干净的论证：二阶系统只有位置反馈无法定阻尼，必须从历史估微分。
cond_overrides () {
  case "$1" in
    mdp)   echo "" ;;
    pomdp) echo "task.pomdp.drop_linear_vel=true task.pomdp.drop_angular_vel=true" ;;
    # ---- POMDP 强度筛选档（pilot 发现仅屏蔽速度不够狠：水下阻力大，
    #      系统接近一阶，且观测里的 throttle 泄漏了速度）----
    # 控制频率 1/0.016 = 62.5 Hz，故 delay=5 ≈ 80 ms、delay=10 ≈ 160 ms，
    # 与声学链路 + 滤波的真实延迟量级相符。
    pomdp_d5)    echo "task.pomdp.drop_linear_vel=true task.pomdp.drop_angular_vel=true task.pomdp.obs_delay=5" ;;
    pomdp_d10)   echo "task.pomdp.drop_linear_vel=true task.pomdp.drop_angular_vel=true task.pomdp.obs_delay=10" ;;
    pomdp_sp10)  echo "task.pomdp.drop_linear_vel=true task.pomdp.drop_angular_vel=true task.pomdp.sparse_pos_period=10" ;;
    pomdp_noise) echo "task.pomdp.drop_linear_vel=true task.pomdp.drop_angular_vel=true task.pomdp.obs_noise_std=0.05" ;;
    # 纯延迟（保留速度观测）：单独测延迟这一项的威力
    delay10)     echo "task.pomdp.obs_delay=10" ;;
    # ---- 任务可行性档 ----
    # pilot 里 tracking_error_ema=0.45 紧贴 reset_thres=0.5，episode 只有 76/600 步，
    # 说明参考速度(≈ traj_scale 1.4-2.6 m × traj_w 0.7-0.9 rad/s ≈ 1.6 m/s)接近甚至
    # 超出载具能力 —— 任务是"能力受限"而非"信息受限"，那样再多信息也没用。
    # 调慢参考让任务真正可跟踪，差异才有空间显现。
    # 任务可行性扫描：参考线速度 ∝ traj_scale × traj_w。
    # traj_w=[0.7,0.9] 时参考 ≈1.6 m/s，而 iAUV 实测只能 ~1.05 m/s
    # ⇒ 误差以 0.55 m/s 累积，0.91 s(=57 步)必然撞上 reset_thres=0.5
    # ⇒ ep_len 由运动学锁死在 ~58，与策略/信息/架构全都无关。
    slow)        echo "task.traj_w_range=[0.3,0.5]" ;;
    slow2)       echo "task.traj_w_range=[0.2,0.35]" ;;
    slow3)       echo "task.traj_w_range=[0.1,0.2]" ;;
    # ===== 在可跟踪任务(slow2)上重筛 POMDP 机制 =====
    # 原始快速轨迹下任务是能力受限的，任何 POMDP 都测不出差异(全部 57.86)。
    # slow2 把 ep_len 抬到 182(仍有到 600 的余量)，此时信息缺失才可能显现。
    s2)          echo "task.traj_w_range=[0.2,0.35]" ;;
    s2_novel)    echo "task.traj_w_range=[0.2,0.35] task.pomdp.drop_linear_vel=true task.pomdp.drop_angular_vel=true" ;;
    s2_d10)      echo "task.traj_w_range=[0.2,0.35] task.pomdp.obs_delay=10" ;;
    s2_d20)      echo "task.traj_w_range=[0.2,0.35] task.pomdp.obs_delay=20" ;;
    s2_sparse)   echo "task.traj_w_range=[0.2,0.35] task.pomdp.sparse_pos_period=10" ;;
    s2_noise)    echo "task.traj_w_range=[0.2,0.35] task.pomdp.obs_noise_std=0.02" ;;
    s2_drift)    echo "task.traj_w_range=[0.2,0.35] task.pomdp.thrust_gain_range=[0.5,1.0]" ;;
    # ---- 未观测周期扰动(涌浪)：理论上保证需要历史 ----
    s2_wave)     echo "task.traj_w_range=[0.2,0.35] task.pomdp.wave_amp=0.3" ;;
    s2_wave5)    echo "task.traj_w_range=[0.2,0.35] task.pomdp.wave_amp=0.5" ;;
    s2_wavefast) echo "task.traj_w_range=[0.2,0.35] task.pomdp.wave_amp=0.4 task.pomdp.wave_freq_range=[6.0,16.0]" ;;
    # ---- 真实 USBL 速率：0.5-1 Hz，即每 60-125 步才一次定位 ----
    # 之前 sparse_pos_period=10 (6.25 Hz) 远高于真实声学定位速率，
    # 两次定位间隔仅 0.16 s、位置误差 ~3 cm，策略察觉不到 ⇒ 测不出差异。
    # 真实速率下必须靠自身推力/速度历史做航位推算，这才是真的需要记忆。
    s2_sp60)     echo "task.traj_w_range=[0.2,0.35] task.pomdp.sparse_pos_period=60" ;;
    s2_sp60_v)   echo "task.traj_w_range=[0.2,0.35] task.pomdp.sparse_pos_period=60 task.pomdp.drop_linear_vel=true task.pomdp.drop_angular_vel=true" ;;
    s2_sp125_v)  echo "task.traj_w_range=[0.2,0.35] task.pomdp.sparse_pos_period=125 task.pomdp.drop_linear_vel=true task.pomdp.drop_angular_vel=true" ;;
    # ---- 切断前馈通道：真正需要历史的条件 ----
    s2_noff)     echo "task.traj_w_range=[0.2,0.35] task.pomdp.drop_future_ref=true task.pomdp.drop_time_encoding=true" ;;
    s2_noff_v)   echo "task.traj_w_range=[0.2,0.35] task.pomdp.drop_future_ref=true task.pomdp.drop_time_encoding=true task.pomdp.drop_linear_vel=true task.pomdp.drop_angular_vel=true" ;;
    noff)        echo "task.pomdp.drop_future_ref=true task.pomdp.drop_time_encoding=true" ;;
    s2_hard)     echo "task.traj_w_range=[0.2,0.35] task.pomdp.drop_linear_vel=true task.pomdp.drop_angular_vel=true task.pomdp.obs_delay=10 task.pomdp.sparse_pos_period=10" ;;
    slow_pomdp)  echo "task.traj_w_range=[0.3,0.5] task.pomdp.drop_linear_vel=true task.pomdp.drop_angular_vel=true" ;;
    slow_pomdp_d10) echo "task.traj_w_range=[0.3,0.5] task.pomdp.drop_linear_vel=true task.pomdp.drop_angular_vel=true task.pomdp.obs_delay=10" ;;
    # 兜底：未观测的执行器增益漂移（必须在线辨识）
    # 主对照：动力学/难度完全相同，唯一差别是增益是否进观测
    drift)        echo "task.pomdp.thrust_gain_range=[0.5,1.0]" ;;
    drift_oracle) echo "task.pomdp.thrust_gain_range=[0.5,1.0] task.pomdp.thrust_gain_in_obs=true" ;;
    # 机制验证：把加速度白送给策略。若序列模型的优势因此消失 ⇒ 优势确实来自恢复加速度
    accel_oracle) echo "task.pomdp.accel_in_obs=true" ;;
    # 机制验证的正确对照：与 drift 完全相同的任务，只多给一项加速度观测。
    # drift vs drift_accel 是一对严格受控的比较（动力学相同，只差这 6 维信息）。
    drift_accel)  echo "task.pomdp.thrust_gain_range=[0.5,1.0] task.pomdp.accel_in_obs=true" ;;
    base)         echo "" ;;
    slow_drift)  echo "task.traj_w_range=[0.3,0.5] task.pomdp.thrust_gain_range=[0.5,1.0]" ;;
    *) echo "UNKNOWN_COND" ;;
  esac
}

# ---------------------------------------------------------------------------
# 调度：这台机器是共享的，别的用户常年占掉每卡 ~44/46 GB。
# 所以不能盲目按卡号轮转，必须：等到有卡真的腾出 MIN_FREE_MB 才投，
# 并对 OOM 自动重试。
# ---------------------------------------------------------------------------
MIN_FREE_MB=${MIN_FREE_MB:-3500}     # 单个 Isaac job 峰值约 2.5-3 GB
MAX_PAR=${MAX_PAR:-2}                # 并行度；共享机器上 4 并行必 OOM
RETRIES=${RETRIES:-3}

# GPU 占用锁：nvidia-smi 的 free 要等 Isaac 完成分配(~2 min)才反映出来，
# 期间 pick_gpu 会把第二个 job 派到同一张卡 → 必然 OOM。
# 用锁文件记录"本 sweep 已把哪张卡派出去了"，锁里存 PID，进程没了锁自动失效。
LOCKDIR=${LOCKDIR:-/tmp/p3_gpu_locks}
mkdir -p "$LOCKDIR"

gpu_locked () {   # $1=gpu idx；被别的活进程占着则返回 0(真)
  local f="$LOCKDIR/gpu$1.lock"
  [ -f "$f" ] || return 1
  local pid; pid=$(cat "$f" 2>/dev/null)
  if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then return 0; fi
  rm -f "$f"; return 1
}

pick_gpu () {   # 打印剩余显存最多、未被本 sweep 占用、且 >= MIN_FREE_MB 的卡号
  local best="" bestfree=0 idx free
  for idx in $GPUS; do
    gpu_locked "$idx" && continue
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$idx" 2>/dev/null)
    [ -z "$free" ] && continue
    if [ "$free" -gt "$bestfree" ]; then bestfree=$free; best=$idx; fi
  done
  [ -n "$best" ] && [ "$bestfree" -ge "$MIN_FREE_MB" ] && echo "$best"
}

wait_for_gpu () {   # 阻塞直到有卡可用，最多等 WAIT_MAX 秒
  local waited=0 g
  while :; do
    g=$(pick_gpu); [ -n "$g" ] && { echo "$g"; return 0; }
    sleep 60; waited=$((waited+60))
    [ "$waited" -ge "${WAIT_MAX:-7200}" ] && { echo ""; return 1; }
  done
}

# PhysX 的 GPU 缓冲默认值是按大规模场景配的（found_lost_aggregate_pairs 3355 万、
# heap 64 MB、软体/粒子接触各 100 万），而我们只有 64 个刚体、无软体无粒子。
# 这台共享机器每卡只剩 ~2.5 GB，削掉这些能省出几百 MB，正好够 mamba 的窗口激活。
# 只改缓冲容量，不改任何物理参数（dt/solver/摩擦等一律不动）。
LOWMEM_OVERRIDES="task.sim.gpu_max_rigid_contact_count=131072 \
task.sim.gpu_max_rigid_patch_count=32768 \
task.sim.gpu_found_lost_pairs_capacity=262144 \
task.sim.gpu_found_lost_aggregate_pairs_capacity=262144 \
task.sim.gpu_total_aggregate_pairs_capacity=262144 \
task.sim.gpu_max_soft_body_contacts=1024 \
task.sim.gpu_max_particle_contacts=1024 \
task.sim.gpu_heap_capacity=16777216 \
task.sim.gpu_temp_buffer_capacity=8388608"
[ "${LOWMEM:-1}" = "0" ] && LOWMEM_OVERRIDES=""

run_one () {   # $1=tag $2=gpu $3...=overrides
  local tag=$1 gpu=$2; shift 2
  echo $$ > "$LOCKDIR/gpu$gpu.lock"          # 占卡；本函数返回时释放
  trap 'rm -f "$LOCKDIR/gpu'"$gpu"'.lock"' RETURN
  CUDA_VISIBLE_DEVICES=$gpu bash run_mamba.sh train.py task=Track headless=true \
    algo.num_minibatches=${NMINI:-16} \
    $LOWMEM_OVERRIDES \
    task.env.num_envs=$NENV max_iters=$ITERS total_frames=100_000_000 \
    "$@" +p3_out="$OUT/$tag.json" +p3_eval_episodes=$EVAL_EP \
    +p3_eval_max_batches=${EVAL_MAXB:-24} +p3_eval_seed=${EVAL_SEED:-12345} \
    +p3_eval_seeds=${EVAL_SEEDS:-10} \
    > "$OUT/$tag.log" 2>&1
}

run_with_retry () {   # OOM/失败自动重试；每次重试都重新选卡
  local tag=$1; shift
  local try
  for try in $(seq 1 $RETRIES); do
    local gpu
    gpu=$(wait_for_gpu)
    if [ -z "$gpu" ]; then
      echo "[$tag] 等不到空闲 GPU，放弃"; return 1
    fi
    echo "[start] $tag -> GPU$gpu (第 $try 次) $(date +%H:%M:%S)"
    run_one "$tag" "$gpu" "$@"
    if [ -f "$OUT/$tag.json" ]; then echo "[ok] $tag"; return 0; fi
    if grep -q "OutOfMemory" "$OUT/$tag.log" 2>/dev/null; then
      echo "[$tag] OOM，等 2 分钟后重试"; sleep 120
    else
      echo "[$tag] 非 OOM 失败:"; tail -3 "$OUT/$tag.log"; return 1
    fi
  done
  echo "[$tag] 重试 $RETRIES 次仍失败"; return 1
}

# 构建待跑列表（跳过已完成的）
jobs=()
for cond in $CONDS; do for arm in $ARMS; do for seed in $SEEDS; do
  tag="${cond}__${arm}__s${seed}"
  [ -f "$OUT/$tag.json" ] && { echo "[skip] $tag"; continue; }
  jobs+=("$tag|$(arm_overrides "$arm") $(cond_overrides "$cond") seed=$seed")
done; done; done

echo "待跑 ${#jobs[@]} 个 job | 并行度 $MAX_PAR | 候选 GPU: $GPUS | 单卡至少需 ${MIN_FREE_MB}MB"
for j in "${jobs[@]}"; do
  tag="${j%%|*}"; ov="${j#*|}"
  while [ "$(jobs -rp | wc -l)" -ge "$MAX_PAR" ]; do wait -n; done
  run_with_retry "$tag" $ov &
  sleep 240         # 错开启动:4096 envs 下 Isaac 约 2 分钟才完成显存分配,错开必须比这长,否则 pick_gpu 会把两个 job 派到同一张卡
done
wait

ok=0; fail=0
for j in "${jobs[@]}"; do
  tag="${j%%|*}"
  if [ -f "$OUT/$tag.json" ]; then ok=$((ok+1)); else
    fail=$((fail+1)); echo "[FAIL] $tag"; fi
done
echo "=========== 完成 $ok/${#jobs[@]}（失败 $fail）==========="

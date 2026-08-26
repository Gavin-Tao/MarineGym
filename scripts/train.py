import logging
import math
import os
import time
import json

import hydra
import torch
import numpy as np
import pandas as pd
import wandb
import matplotlib.pyplot as plt

from torch.func import vmap
from tqdm import tqdm
from omegaconf import OmegaConf

from marinegym import init_simulation_app
from torchrl.data import CompositeSpec
from torchrl.envs.utils import set_exploration_type, ExplorationType
from marinegym.utils.torchrl import SyncDataCollector
from marinegym.utils.torchrl.transforms import (
    FromMultiDiscreteAction,
    FromDiscreteAction,
    ravel_composite,
    AttitudeController,
    RateController,
)
from marinegym.utils.wandb import init_wandb
from marinegym.utils.torchrl import RenderCallback, EpisodeStats
from marinegym.learning import ALGOS

from setproctitle import setproctitle
from torchrl.envs.transforms import TransformedEnv, InitTracker, Compose


def compute_avg_power(actions: torch.Tensor) -> float:
    """Compute mean electrical power (W) across all steps/envs/rotors from action tensor."""
    throttle = actions  # T200: signed throttle in [-1, 1]
    rpm = torch.where(
        throttle > 0.075,  3.6599e3 * throttle + 3.4521e2,
        torch.where(throttle < -0.075, 3.4944e3 * throttle - 4.3350e2,
            torch.zeros_like(throttle))
    ).clamp(-3900, 3900)
    thrust = (9.81 * torch.where(
        rpm > 0,
        4.7368e-7 * rpm.pow(2) - 1.9275e-4 * rpm + 8.4452e-2,
        -3.8442e-7 * rpm.pow(2) - 1.6186e-4 * rpm - 3.9139e-2,
    )).clamp(-40.22, 51.50)  # clamp in N (after ×9.81), matching policy _compute_power_cost
    power_fwd = 0.758 * thrust.abs().pow(1.574)
    power_rev = 0.851 * thrust.abs().pow(1.654)
    power = torch.where(thrust >= 0, power_fwd, power_rev)
    return power.sum(-1).mean().item()  # sum over rotors, mean over steps/envs


@hydra.main(version_base=None, config_path=".", config_name="train")
def main(cfg):
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    simulation_app = init_simulation_app(cfg)
    run = init_wandb(cfg)
    setproctitle(run.name)
    print(OmegaConf.to_yaml(cfg))

    from marinegym.envs.isaac_env import IsaacEnv

    env_class = IsaacEnv.REGISTRY[cfg.task.name]
    base_env = env_class(cfg, headless=cfg.headless)

    transforms = [InitTracker()]

    # a CompositeSpec is by default processed by a entity-based encoder
    # ravel it to use a MLP encoder instead
    if cfg.task.get("ravel_obs", False):
        transform = ravel_composite(base_env.observation_spec, ("agents", "observation"))
        transforms.append(transform)
    if cfg.task.get("ravel_obs_central", False):
        transform = ravel_composite(base_env.observation_spec, ("agents", "observation_central"))
        transforms.append(transform)

    # optionally discretize the action space or use a controller
    action_transform: str = cfg.task.get("action_transform", None)
    if action_transform is not None:
        if action_transform.startswith("multidiscrete"):
            nbins = int(action_transform.split(":")[1])
            transform = FromMultiDiscreteAction(nbins=nbins)
            transforms.append(transform)
        elif action_transform.startswith("discrete"):
            nbins = int(action_transform.split(":")[1])
            transform = FromDiscreteAction(nbins=nbins)
            transforms.append(transform)
        else:
            raise NotImplementedError(f"Unknown action transform: {action_transform}")

    env = TransformedEnv(base_env, Compose(*transforms)).train()
    env.set_seed(cfg.seed)

    # Inject drone model name for actuator-aware power cost computation
    if hasattr(cfg.algo, "drone_model_name"):
        cfg.algo.drone_model_name = cfg.task.drone_model.name

    try:
        policy = ALGOS[cfg.algo.name.lower()](
            cfg.algo,
            env.observation_spec,
            env.action_spec,
            env.reward_spec,
            device=base_env.device
        )
    except KeyError:
        raise NotImplementedError(f"Unknown algorithm: {cfg.algo.name}")

    # Optional: load a pre-trained checkpoint for eval-only mode
    load_ckpt = cfg.get("load_ckpt", None)
    if load_ckpt:
        ckpt = torch.load(load_ckpt, map_location=base_env.device)
        policy.load_state_dict(ckpt)
        logging.info(f"Loaded checkpoint from {load_ckpt}")

    frames_per_batch = env.num_envs * int(cfg.algo.train_every)
    total_frames = cfg.get("total_frames", -1) // frames_per_batch * frames_per_batch
    max_iters = cfg.get("max_iters", -1)
    eval_interval = cfg.get("eval_interval", -1)
    save_interval = cfg.get("save_interval", -1)

    stats_keys = [
        k for k in base_env.observation_spec.keys(True, True)
        if isinstance(k, tuple) and k[0]=="stats"
    ]
    episode_stats = EpisodeStats(stats_keys)
    collector = SyncDataCollector(
        env,
        policy=policy,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        device=cfg.sim.device,
        return_same_td=True,
    )

    @torch.no_grad()
    def evaluate(
        seed: int=0,
        exploration_type: ExplorationType=ExplorationType.MODE
    ):

        if not cfg.headless:
            base_env.enable_render(True)
        base_env.eval()
        env.eval()
        env.set_seed(seed)

        render_callback = RenderCallback(interval=2) if not cfg.headless else None

        with set_exploration_type(exploration_type):
            trajs = env.rollout(
                max_steps=base_env.max_episode_length,
                policy=policy,
                callback=render_callback,
                auto_reset=True,
                break_when_any_done=False,
                return_contiguous=False,
            )
        base_env.enable_render(not cfg.headless)
        env.reset()

        done = trajs.get(("next", "done"))
        first_done = torch.argmax(done.long(), dim=1).cpu()

        def take_first_episode(tensor: torch.Tensor):
            indices = first_done.reshape(first_done.shape+(1,)*(tensor.ndim-2))
            return torch.take_along_dim(tensor, indices, dim=1).reshape(-1)

        traj_stats = {
            k: take_first_episode(v)
            for k, v in trajs[("next", "stats")].cpu().items()
        }

        info = {
            "eval/stats." + k: torch.mean(v.float()).item()
            for k, v in traj_stats.items()
        }

        actions = trajs.get(("agents", "action"))
        info["eval/avg_power_W"] = compute_avg_power(actions.cpu())

        # log video (only when rendering is available)
        if render_callback is not None:
            info["recording"] = wandb.Video(
                render_callback.get_video_array(axes="t c h w"),
                fps=0.5 / (cfg.sim.dt * cfg.sim.substeps),
                format="mp4"
            )

        # log distributions
        # df = pd.DataFrame(traj_stats)
        # table = wandb.Table(dataframe=df)
        # info["eval/return"] = wandb.plot.histogram(table, "return")
        # info["eval/episode_len"] = wandb.plot.histogram(table, "episode_len")

        return info

    # eval-only: load a checkpoint, roll out the deterministic policy under the current (e.g. OOD) config,
    # aggregate keep-out metrics, print, and exit — no training. (filter on/off is just a keepout.* config)
    # ---- 轨迹采集（仅在显式指定 +save_traj=<path.npz> 时启用，不影响既有流程）----
    if cfg.get("save_traj", None):
        from torchrl.envs.utils import step_mdp
        logging.info("CAPTURE-TRAJ: rolling out and recording positions...")
        base_env.eval(); env.eval()
        n_steps = int(cfg.get("traj_steps", 240))
        rec = {k: [] for k in ("pos", "ref", "obst", "quat", "vel", "engage", "clearance",
                               "u_rl", "u_applied", "done")}
        td = env.reset()
        be = base_env
        # 完整参考轨迹（整条 lemniscate，从 reset 时刻起算 max_episode_length 步）
        # —— 用于在轨迹图上显示载具在整条路径上走到了哪
        try:
            full_ref = be._compute_traj(be.max_episode_length, step_size=1.0).cpu().numpy()
        except Exception as ex:
            logging.warning(f"full_ref 采集失败: {ex}")
            full_ref = None
        with set_exploration_type(ExplorationType.MODE), torch.no_grad():
            const_a = cfg.get("const_action", None)
            for _ in range(n_steps):
                td = policy(td)
                if const_a is not None:      # 诊断：用固定动作覆盖策略，测载具真实能力
                    td.set(("agents", "action"),
                           torch.full_like(td[("agents", "action")], float(const_a)))
                td = env.step(td)
                rec["pos"].append(be.drone_state[..., :3].reshape(-1, 3).cpu().numpy().copy())
                rec["ref"].append(be.target_pos[:, 0].reshape(-1, 3).cpu().numpy().copy())
                _ob = getattr(be, "obstacle_pos", None)
                rec["obst"].append(_ob.reshape(be.num_envs, -1, 3).cpu().numpy().copy()
                                   if _ob is not None else np.zeros((be.num_envs, 1, 3), dtype=np.float32))
                # drone_state = [pos(3), quat(4), vel(6), heading(3), up(3), throttle(6)]
                rec["quat"].append(be.drone_state[..., 3:7].reshape(-1, 4).cpu().numpy().copy())
                rec["vel"].append(be.drone_state[..., 7:13].reshape(-1, 6).cpu().numpy().copy())
                eng = getattr(be, "_filter_engage", None)
                rec["engage"].append(eng.reshape(-1).float().cpu().numpy().copy()
                                     if eng is not None else np.zeros(be.num_envs, dtype=np.float32))
                for key, attr in (("u_rl", "_u_rl"), ("u_applied", "_u_applied")):
                    v = getattr(be, attr, None)
                    rec[key].append(v.reshape(be.num_envs, -1).cpu().numpy().copy()
                                    if v is not None else np.zeros((be.num_envs, 6), dtype=np.float32))
                cl = getattr(be, "_clearance", None)
                rec["clearance"].append(cl.reshape(-1).cpu().numpy().copy()
                                        if cl is not None else np.full(be.num_envs, np.nan, dtype=np.float32))
                rec["done"].append(td["next", "done"].reshape(-1).cpu().numpy().copy())
                td = step_mdp(td)
        out = {k: np.asarray(v) for k, v in rec.items()}
        if full_ref is not None: out["full_ref"] = full_ref
        out["keepout_r"] = np.asarray([float(getattr(be, "keepout_radius", 0.0))
                                       + float(getattr(be, "vehicle_radius", 0.0))])
        np.savez_compressed(cfg.save_traj, **out)
        logging.info(f"CAPTURE-TRAJ: saved {out['pos'].shape} to {cfg.save_traj}")
        wandb.finish(); simulation_app.close(); return

    if cfg.get("eval_only", False):
        logging.info("EVAL-ONLY: rolling out loaded policy (no training)...")
        base_env.eval(); env.eval()
        n_eval_ep = int(cfg.get("eval_episodes", 200))
        max_batches = int(cfg.get("eval_max_batches", 45))   # wall-clock guard (episodes can be long)
        with set_exploration_type(ExplorationType.MODE):
            for bi, data in enumerate(collector):
                episode_stats.add(data.to_tensordict())
                if bi % 5 == 0:
                    print(f"[eval] batch {bi} episodes {episode_stats._episodes}", flush=True)
                if episode_stats._episodes >= n_eval_ep or bi >= max_batches:
                    break
        stats = episode_stats.pop()
        agg = {(".".join(k) if isinstance(k, tuple) else k): v.float().mean().item()
               for k, v in stats.items(True, True)}
        print(f"=== EVAL-ONLY RESULTS (episodes={episode_stats._episodes}) ===", flush=True)
        for k in sorted(agg):
            print(f"  {k}: {agg[k]:.4f}", flush=True)
        wandb.finish()
        simulation_app.close()
        return

    # 论文③：把每次迭代的标量指标累积下来，最后连同确定性评测一起写 JSON
    p3_out = cfg.get("p3_out", None)
    p3_curve = []
    p3_t0 = time.time()

    pbar = tqdm(collector, total=total_frames//frames_per_batch)
    env.train()
    for i, data in enumerate(pbar):
        info = {"env_frames": collector._frames, "rollout_fps": collector._fps}
        episode_stats.add(data.to_tensordict())

        if len(episode_stats) >= base_env.num_envs:
            stats = {
                "train/" + (".".join(k) if isinstance(k, tuple) else k): torch.mean(v.float()).item()
                for k, v in episode_stats.pop().items(True, True)
            }
            info.update(stats)

        info.update(policy.train_op(data.to_tensordict()))

        train_actions = data.get(("agents", "action"))
        if train_actions is not None:
            info["train/stats.avg_power_W"] = compute_avg_power(train_actions.cpu())

        if eval_interval > 0 and i % eval_interval == 0:
            logging.info(f"Eval at {collector._frames} steps.")
            info.update(evaluate())
            env.train()
            base_env.train()

        if save_interval > 0 and i % save_interval == 0:
            try:
                ckpt_path = os.path.join(run.dir, f"checkpoint_{collector._frames}.pt")
                torch.save(policy.state_dict(), ckpt_path)
                logging.info(f"Saved checkpoint to {str(ckpt_path)}")
            except AttributeError:
                logging.warning(f"Policy {policy} does not implement `.state_dict()`")

        run.log(info)
        if p3_out:
            p3_curve.append({k: v for k, v in info.items()
                             if isinstance(v, (int, float))} | {"iter": i,
                             "wall_s": time.time() - p3_t0})
        print(OmegaConf.to_yaml({k: v for k, v in info.items() if isinstance(v, float)}))

        pbar.set_postfix({"rollout_fps": collector._fps, "frames": collector._frames})

        if max_iters > 0 and i >= max_iters - 1:
            break

    # ---- 论文③：确定性评测 + 结果落盘 ----
    if p3_out:
        # 用 collector 聚合**所有完整 episode**（而不是 evaluate() 那样一次 600 步
        # rollout 只取每个 env 的第一条 episode）。episode 平均 ~60 步而
        # max_episode_length=600，取第一条会浪费 ~90% 的数据，且慢 5 倍。
        # ---- 多评测 seed（与论文①② 口径一致：每格 N 个独立评测 seed，mean ± std）----
        # 单次训练 + N 个独立评测 seed。每个 seed 是一次独立的确定性 rollout，
        # 各 arm 用**同一组 seed**（共同随机数）⇒ 面对相同的初始条件与轨迹参数序列，
        # 消掉这部分方差，arm 之间才可比。
        n_seeds = int(cfg.get("p3_eval_seeds", 10))
        n_ep = int(cfg.get("p3_eval_episodes", 2000))       # 每个 seed 的目标 episode 数
        # 每批 = train_every 步(默认 64)。episode 上限 600 步 ⇒ 至少要 10 批
        # 才可能有 episode 跑满结束；留足余量取 24。
        max_b = int(cfg.get("p3_eval_max_batches", 24))     # 每个 seed 的 batch 上限
        seed0 = int(cfg.get("p3_eval_seed", 12345))
        logging.info(f"[paper3] deterministic eval: {n_seeds} 个评测 seed × "
                     f"~{n_ep} episode (共同随机数, 基 seed={seed0}) ...")
        base_env.eval(); env.eval()
        t_eval = time.time()

        # 用**独立的** collector 做评测：训练用的那个在训练循环 break 之后
        # 迭代器状态已终止，再迭代会立刻返回空（实测每个 seed 只花 6 s、
        # 收不到任何 episode）。评测批小一些(train_every//2)以便细粒度控制。
        eval_fpb = env.num_envs * max(int(cfg.algo.train_every) // 2, 8)
        # ⚠ marinegym 的 SyncDataCollector.iterator() 末尾是
        #   `if self._frames >= self.total_frames: break`
        #   传 -1 会在第 1 批之后立刻 break（只有 32 步，慢速轨迹下一条
        #   episode 都结束不了）。必须给足额度：每个 seed 最多 max_b 批，
        #   而 iterator() 每次 __iter__ 都把 _frames 归零，故按单 seed 算即可。
        eval_collector = SyncDataCollector(
            env, policy=policy, frames_per_batch=eval_fpb,
            total_frames=eval_fpb * (max_b + 2),
            device=cfg.sim.device, return_same_td=True,
        )

        per_seed = {}          # key -> [每个 seed 的均值]
        total_ep = 0
        for si in range(n_seeds):
            env.set_seed(seed0 + si)
            env.reset()
            # ⚠ EpisodeStats.pop() 只清 _stats，**不重置 _episodes**（它从训练
            # 开始就一直累加）。若用 _episodes 作停止判据，评测第一批之后就会
            # 立刻满足 `>= n_ep` 而退出 —— 慢速轨迹下一条 episode 都收不到。
            # 这里显式清零，并改用 len(episode_stats) 作判据。
            episode_stats._stats.clear()
            episode_stats._episodes = 0
            _pw_seed = []
            _nb = 0
            with set_exploration_type(ExplorationType.MODE), torch.no_grad():
                for bi, data in enumerate(eval_collector):
                    _nb = bi + 1
                    episode_stats.add(data.to_tensordict())
                    _a = data.get(("agents", "action"), None)
                    if _a is not None:
                        _pw_seed.append(compute_avg_power(_a.cpu()))
                    if len(episode_stats) >= n_ep or bi >= max_b:
                        break
            # 任务可跟踪时 episode 会跑满 max_episode_length(600 步)，
            # 若批数不够则一条都不会结束 ⇒ pop() 会抛 "non-empty TensorList"。
            # 这里显式跳过并告警，而不是让整个 run 崩掉。
            logging.info(f"[paper3] 评测 seed {seed0+si}: 实际迭代 {_nb} 批, "
                         f"累计 episode {len(episode_stats)}")
            if len(episode_stats) == 0:
                logging.warning(f"[paper3] 评测 seed {seed0+si}: {max_b} 批(每批 "
                                f"{eval_fpb} frames)内无 episode 结束"
                                f"。任务可能已可持续跟踪，请加大 p3_eval_max_batches。跳过该 seed。")
                continue
            total_ep += len(episode_stats)
            st_i = episode_stats.pop()
            row = {}
            for k, v in st_i.items(True, True):
                name = "eval/" + (".".join(k) if isinstance(k, tuple) else k)
                row[name] = float(v.float().mean())

            # ---- 派生指标（比原始 stats 更可读，也更适合当论文主指标）----
            _el = st_i.get(("stats", "episode_len")).float().reshape(-1).clamp_min(1.0)
            _te = st_i.get(("stats", "tracking_error")).float().reshape(-1)
            # tracking_error 是逐步累加的 -distance ⇒ 平均跟踪误差(米) = -te/len
            _tem = (-_te / _el)
            row["eval/stats.tracking_err_mean_m"] = float(_tem.mean())
            row["eval/stats.tracking_err_p90_m"] = float(_tem.quantile(0.90))
            # 成功率：跑满时长而没有跟丢（episode 因跟踪失败提前终止）
            _maxlen = float(base_env.max_episode_length)
            row["eval/stats.success_rate"] = float((_el >= _maxlen * 0.99).float().mean())
            row["eval/stats.episode_len_p50"] = float(_el.median())
            row["eval/stats.episode_len_p10"] = float(_el.quantile(0.10))
            # 能耗（AUV 上很关键）：本 seed 各 batch 的平均电功率
            if _pw_seed:
                row["eval/stats.avg_power_W"] = float(np.mean(_pw_seed))
            for name, val in row.items():
                per_seed.setdefault(name, []).append(val)

        # 聚合：mean ± std **跨评测 seed**（n = n_seeds），与论文①② 一致
        if not per_seed:
            logging.error("[paper3] 所有评测 seed 都没有完整 episode —— "
                          "p3_eval_max_batches 太小。不写结果。")
            raise RuntimeError("eval collected zero episodes; increase p3_eval_max_batches")
        agg = {}
        for name, vals in per_seed.items():
            a = np.asarray(vals, dtype=float)
            agg[name] = {"mean": float(a.mean()), "std": float(a.std(ddof=1) if len(a) > 1 else 0.0),
                         "n": len(a), "values": [float(x) for x in a]}
        n_done = total_ep

        # 发散检测：value_loss 爆掉 → NaN → 各指标全 0，会静默污染结果表。
        # 显式标记，汇总脚本据此剔除并在论文里如实报告。
        diverged = False
        for c in p3_curve:
            for k in ("policy_loss", "value_loss", "entropy"):
                v = c.get(k)
                if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                    diverged = True
        _el_agg = agg.get("eval/stats.episode_len", {})
        if (not math.isfinite(_el_agg.get("mean", 1.0))) or _el_agg.get("mean", 1.0) <= 0.0:
            diverged = True
        if diverged:
            logging.warning("[paper3] 该 run 训练发散(NaN/全零)，已标记 diverged=true")

        rec = {
            "diverged": diverged,
            "arm": str(cfg.algo.encoder.get("name", "mlp")),
            "encoder": getattr(policy, "encoder_cfg", None),
            "context_len": int(cfg.task.get("context_len", 1)),
            # 上游原始配置(TrackOrig)里没有 pomdp 段，get 会返回普通 dict，
            # to_container 只吃 OmegaConf 对象 —— 必须容错，否则训练白跑。
            "pomdp": (OmegaConf.to_container(cfg.task.pomdp, resolve=True)
                      if "pomdp" in cfg.task and cfg.task.pomdp is not None else {}),
            "seed": int(cfg.seed),
            "total_frames": int(collector._frames),
            "param_count": getattr(policy, "param_count", None),
            "wandb_dir": getattr(run, "dir", None),      # 便于日后只重跑评测
            "eval_seeds": [seed0 + i for i in range(n_seeds)],
            "eval_protocol": (f"{n_seeds} independent eval seeds, common random "
                              f"numbers across arms, deterministic (MODE)"),
            "train_wall_s": time.time() - p3_t0,
            "eval_wall_s": time.time() - t_eval,
            "eval_episodes": n_done,
            "eval": agg,
            "curve": p3_curve,
        }
        os.makedirs(os.path.dirname(os.path.abspath(p3_out)), exist_ok=True)
        with open(p3_out, "w") as f:
            json.dump(rec, f, indent=1, default=str)
        logging.info(f"[paper3] wrote {p3_out}  ({n_done} episodes)")
        for k in ("eval/stats.episode_len", "eval/stats.return",
                  "eval/stats.tracking_err_mean_m", "eval/stats.success_rate",
                  "eval/stats.avg_power_W", "eval/stats.action_smoothness"):
            if k in agg:
                print(f"[paper3] {k}: {agg[k]['mean']:.4f} ± {agg[k]['std']:.4f}"
                      f"  (n={agg[k]['n']})", flush=True)

    logging.info(f"Final Eval at {collector._frames} steps.")
    info = {"env_frames": collector._frames}
    # info.update(evaluate())
    # run.log(info)

    try:
        ckpt_path = os.path.join(run.dir, "checkpoint_final.pt")
        torch.save(policy.state_dict(), ckpt_path)

        model_artifact = wandb.Artifact(
            f"{cfg.task.name}-{cfg.algo.name.lower()}-{cfg.task.drone_model.name}",
            type="model",
            description=f"{cfg.task.name}-{cfg.algo.name.lower()}",
            metadata=dict(cfg))

        model_artifact.add_file(ckpt_path)
        wandb.save(ckpt_path)
        run.log_artifact(model_artifact)

        logging.info(f"Saved checkpoint to {str(ckpt_path)}")
    except AttributeError:
        logging.warning(f"Policy {policy} does not implement `.state_dict()`")

    wandb.finish()

    simulation_app.close()


if __name__ == "__main__":
    main()

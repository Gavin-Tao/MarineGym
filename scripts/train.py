import logging
import os
import time

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
                               "u_rl", "u_applied", "done", "reward", "flow",
                               "wall_clear", "wall_hw", "lam", "d_hat",
                               "pred_clear", "pred_clear_zero", "pred_clear_oracle")}
        td = env.reset()
        be = base_env
        # 完整参考轨迹（整条 lemniscate，从 reset 时刻起算 max_episode_length 步）
        # —— 用于在轨迹图上显示载具在整条路径上走到了哪
        try:
            # 上限 2000 步：max_episode_length 被调大时（长漂移诊断会设到上万），
            # 这里的 vmap 会退化成一个巨大的逐步计算并把整个采集卡住。
            full_ref = be._compute_traj(min(int(be.max_episode_length), 2000), step_size=1.0).cpu().numpy()
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
                # V0 回归指纹：reward 是 env 全部改动的最敏感汇总量（奖励项一动就变）
                rec["reward"].append(td["next", "agents", "reward"].reshape(-1).cpu().numpy().copy())
                # 论文②走廊：侧壁裕度与该 episode 的半宽（V6 违约计数验证用）
                for key, attr in (("wall_clear", "_cor_clear"), ("wall_hw", "_cor_hw"), ("lam", "_cor_lam"),
                                  ("pred_clear", "_pred_clear"),          # K4: 三种 d̂ 下的预测最小裕度
                                  ("pred_clear_zero", "_pred_clear_zero"),
                                  ("pred_clear_oracle", "_pred_clear_oracle")):
                    v = getattr(be, attr, None)
                    rec[key].append(v.reshape(-1).cpu().numpy().copy()
                                    if v is not None else np.full(be.num_envs, np.nan, dtype=np.float32))
                _dh = getattr(be, "_d_hat", None)   # 观测器输出（体系等效外力残差，6 维）
                rec["d_hat"].append(_dh.reshape(be.num_envs, -1)[:, :6].cpu().numpy().copy()
                                    if _dh is not None else np.full((be.num_envs, 6), np.nan, dtype=np.float32))
                # V1/V2/V3/V5 洋流验证：世界系流速 [lin(3), ang(3)]，每步记录
                fv = getattr(be.drone, "flow_vels", None)
                rec["flow"].append(fv.reshape(be.num_envs, -1)[:, :6].cpu().numpy().copy()
                                   if fv is not None else np.zeros((be.num_envs, 6), dtype=np.float32))
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

        # 论文②：训练曲线落盘 + 控制台摘要。
        # 原实现只把统计发给 wandb(offline)，无人值守时既看不到进展，也没法直接画训练曲线。
        _tl = cfg.get("train_log", None)
        # 前几个迭代还没有任何完整 episode，info 里没有 train/stats.*。若此时就定表头，
        # wall_violation 这些关键列会永远缺席。等到 stats 真的出现再开始记。
        if _tl and any(k.startswith("train/stats.") for k in info):
            import csv as _csv
            # 表头必须**只定一次**：前几个迭代还没有完整 episode，info 的键更少，
            # 逐行重算 fieldnames 会写出列数不一致的 CSV（pandas 直接解析失败）。
            # 因此第一次写入时定下表头并复用；缺的键留空，多出来的丢弃。
            if not hasattr(main, "_tl_fields"):
                if os.path.exists(_tl):
                    with open(_tl) as _f:
                        main._tl_fields = next(_csv.reader(_f))
                else:
                    main._tl_fields = ["iter"] + sorted(
                        k for k in info if isinstance(info[k], (int, float)))
                    with open(_tl, "w", newline="") as _f:
                        _csv.DictWriter(_f, fieldnames=main._tl_fields).writeheader()
            with open(_tl, "a", newline="") as _f:
                _csv.DictWriter(_f, fieldnames=main._tl_fields, extrasaction="ignore",
                                restval="").writerow({"iter": i, **info})
            if i % 10 == 0:
                _pick = [(k.split("stats.")[-1], info[k]) for k in
                         ("train/stats.wall_violation", "train/stats.wall_viol_frac",
                          "train/stats.tracking_error_ema", "train/stats.min_wall_dist",
                          "train/stats.episode_len", "train/stats.return") if k in info]
                if _pick:
                    print("[it %4d] " % i + "  ".join(f"{k}={v:.4f}" for k, v in _pick), flush=True)

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
        print(OmegaConf.to_yaml({k: v for k, v in info.items() if isinstance(v, float)}))

        pbar.set_postfix({"rollout_fps": collector._fps, "frames": collector._frames})

        if max_iters > 0 and i >= max_iters - 1:
            break

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

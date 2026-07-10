#!/usr/bin/env python3
"""OOD evaluation script. Run as Hydra app."""
# Pre-import to register Hydra ConfigStore entries
from marinegym.learning import ALGOS

import hydra, torch, json, time, os, sys
import numpy as np
from omegaconf import OmegaConf, DictConfig

OOD_PRESETS = {
    "train": dict(enable=True, n_obstacles=1, radius=0.8, reward_weight=2.0,
                  lateral_offset_std=0.0),
    "b1":    dict(enable=True, n_obstacles=1, radius=2.0, reward_weight=2.0,
                  lateral_offset_std=0.0),
    "b2":    dict(enable=True, n_obstacles=3, radius=0.8, reward_weight=2.0,
                  lateral_offset_std=0.0),
    "b3":    dict(enable=True, n_obstacles=1, radius=0.8, reward_weight=2.0,
                  lateral_offset_std=1.0),
    "b4":    dict(enable=True, n_obstacles=1, radius=0.8, reward_weight=2.0,
                  lateral_offset_std=0.0),
}

def run_eval(env, policy, base_env, num_seeds):
    from torchrl.envs.utils import set_exploration_type, ExplorationType
    stats = {"collision": [], "min_obstacle_dist": [], "success": [],
             "tracking_error": []}
    for seed in range(num_seeds):
        env.set_seed(seed); base_env.eval(); env.eval()
        with set_exploration_type(ExplorationType.MODE):
            trajs = env.rollout(max_steps=base_env.max_episode_length, policy=policy,
                                auto_reset=True, break_when_any_done=False,
                                return_contiguous=False)
        done = trajs.get(("next", "done"))
        first_done = torch.argmax(done.long(), dim=1).cpu()
        def take_first(t):
            idx = first_done.reshape(first_done.shape + (1,)*(t.ndim-2))
            return torch.take_along_dim(t, idx, dim=1).reshape(-1)
        ts = {k: take_first(v) for k, v in trajs[("next", "stats")].cpu().items()}
        stats["collision"].append(ts.get("collision", torch.zeros(1)).float().mean().item())
        stats["min_obstacle_dist"].append(ts.get("min_obstacle_dist", torch.tensor([100.])).float().mean().item())
        stats["success"].append(1 - stats["collision"][-1])
        stats["tracking_error"].append(ts.get("tracking_error", torch.zeros(1)).float().mean().item())
        env.reset()
    return {k: {"mean": float(np.mean(v)), "std": float(np.std(v))} for k, v in stats.items()}

@hydra.main(version_base=None, config_path=".", config_name="train")
def main(cfg: DictConfig):
    from marinegym import init_simulation_app
    simulation_app = init_simulation_app(cfg)

    from marinegym.envs.isaac_env import IsaacEnv
    from torchrl.envs.transforms import TransformedEnv, InitTracker, Compose

    ood_name = cfg.get("eval_ood", "train")
    mppi_mode = cfg.get("eval_mppi", "off")
    num_seeds = cfg.get("eval_seeds", 20)
    ckpt_path = cfg.get("eval_ckpt", "")

    kp = OOD_PRESETS.get(ood_name, OOD_PRESETS["train"])
    task_cfg = cfg.task
    if "keepout" not in task_cfg:
        task_cfg["keepout"] = {}
    task_cfg["keepout"].update(kp)
    task_cfg["keepout"]["mppi"]["enable"] = (mppi_mode != "off")
    task_cfg["keepout"]["mppi"]["exact"] = (mppi_mode == "exact")
    task_cfg["keepout"]["shield"]["enable"] = False

    print(f"OOD: {ood_name} | MPPI: {mppi_mode} | Seeds: {num_seeds}")
    if not ckpt_path:
        print("ERROR: --cfg eval_ckpt=PATH required"); simulation_app.close(); return

    env_class = IsaacEnv.REGISTRY[cfg.task.name]
    base_env = env_class(cfg, headless=cfg.headless)
    env = TransformedEnv(base_env, Compose(InitTracker())).train()
    env.set_seed(42)

    policy = ALGOS[cfg.algo.name.lower()](cfg.algo, env.observation_spec,
                                            env.action_spec, env.reward_spec,
                                            device=base_env.device)
    ckpt = torch.load(ckpt_path, map_location=base_env.device)
    policy.load_state_dict(ckpt); policy.eval()
    print(f"Loaded: {ckpt_path}")

    t0 = time.time()
    summary = run_eval(env, policy, base_env, num_seeds)
    elapsed = time.time() - t0

    name = f"{ood_name}_mppi-{mppi_mode}"
    print(f"\n{'='*50}")
    print(f"Results: {name} ({num_seeds} seeds, {elapsed:.1f}s)")
    for k in ["collision","min_obstacle_dist","success","tracking_error"]:
        print(f"  {k:20s}: {summary[k]['mean']:.4f} +/- {summary[k]['std']:.4f}")

    output = {"ood": ood_name, "mppi": mppi_mode, "n_seeds": num_seeds,
              "elapsed_s": elapsed, "results": summary}
    out_path = f"/tmp/ood_eval_{ood_name}_mppi-{mppi_mode}.json"
    with open(out_path, "w") as f: json.dump(output, f, indent=2)
    print(f"Saved to {out_path}")
    simulation_app.close()

if __name__ == "__main__":
    main()

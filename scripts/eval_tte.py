#!/usr/bin/env python3
"""TTE 多 seed 评估脚本 — 加载 checkpoint, N test seeds rollout, 统计 + 显著性检验"""
import hydra, torch, json, time, os, sys, argparse
import numpy as np
from omegaconf import OmegaConf, DictConfig

# Must register algos before hydra main
from marinegym.learning import ALGOS

CKPT_MAP = {}  # populated after training: label -> path

def find_latest_checkpoint(wandb_dir, pattern):
    """Find the latest checkpoint_final.pt matching pattern in wandb dirs."""
    import glob
    matches = []
    for d in glob.glob(os.path.join(wandb_dir, "offline-run-*")):
        ckpt = os.path.join(d, "files", "checkpoint_final.pt")
        if os.path.exists(ckpt):
            matches.append((os.path.getmtime(ckpt), ckpt))
    matches.sort(reverse=True)
    return [m[1] for m in matches]


def run_eval(env, policy, base_env, num_seeds, training_seed=0):
    """Run `num_seeds` evaluation rollouts, each with a different test seed."""
    from torchrl.envs.utils import set_exploration_type, ExplorationType
    stats = {
        "collision": [], "min_obstacle_dist": [], "return": [],
        "episode_len": [], "tracking_error": [], "tracking_error_ema": [],
        "filter_activation": [], "correction": [], "detour_ratio": [],
        "over_clearance": [], "action_smoothness": [], "internalize_w": [],
    }
    for seed in range(training_seed + 1, training_seed + 1 + num_seeds):
        env.set_seed(seed)
        base_env.eval()
        env.eval()
        with set_exploration_type(ExplorationType.MODE):
            trajs = env.rollout(
                max_steps=base_env.max_episode_length,
                policy=policy,
                auto_reset=True,
                break_when_any_done=False,
                return_contiguous=False,
            )
        done = trajs.get(("next", "done"))
        first_done = torch.argmax(done.long(), dim=1).cpu()

        def take_first(t):
            idx = first_done.reshape(first_done.shape + (1,) * (t.ndim - 2))
            return torch.take_along_dim(t, idx, dim=1).reshape(-1)

        ts = {k: take_first(v) for k, v in trajs[("next", "stats")].cpu().items()}
        for k in stats:
            if k in ts:
                stats[k].append(ts[k].float().mean().item())
            else:
                stats[k].append(float("nan"))
        env.reset()
    return stats


def compute_stats(values):
    """Compute mean, std, and 95% CI."""
    arr = np.array(values, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return {"mean": float("nan"), "std": float("nan"), "ci95": float("nan"), "n": 0}
    mean = arr.mean()
    std = arr.std(ddof=1) if len(arr) > 1 else 0.0
    ci95 = 1.96 * std / np.sqrt(len(arr)) if len(arr) > 1 else 0.0
    return {"mean": float(mean), "std": float(std), "ci95": float(ci95), "n": len(arr)}


def welch_t_test(a, b):
    """Welch's t-test (unequal variance). Returns (t_stat, p_value)."""
    from scipy import stats
    a = np.array(a, dtype=np.float64)
    b = np.array(b, dtype=np.float64)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan")
    t, p = stats.ttest_ind(a, b, equal_var=False)
    return float(t), float(p)


@hydra.main(version_base=None, config_path=".", config_name="train")
def main(cfg: DictConfig):
    from marinegym import init_simulation_app
    simulation_app = init_simulation_app(cfg)

    from marinegym.envs.isaac_env import IsaacEnv
    from torchrl.envs.transforms import TransformedEnv, InitTracker, Compose

    ckpt_path = cfg.get("eval_ckpt", "")
    num_seeds = cfg.get("eval_seeds", 10)
    training_seed = cfg.get("training_seed", 0)
    output_path = cfg.get("eval_output", "/tmp/tte_eval.json")
    label = cfg.get("eval_label", "unknown")

    if not ckpt_path:
        print("ERROR: eval_ckpt required")
        simulation_app.close()
        return

    print(f"[Eval] label={label} ckpt={ckpt_path} seeds={num_seeds}")
    env_class = IsaacEnv.REGISTRY[cfg.task.name]
    base_env = env_class(cfg, headless=cfg.headless)
    env = TransformedEnv(base_env, Compose(InitTracker())).train()
    env.set_seed(42)

    policy = ALGOS[cfg.algo.name.lower()](
        cfg.algo, env.observation_spec, env.action_spec, env.reward_spec,
        device=base_env.device,
    )
    ckpt = torch.load(ckpt_path, map_location=base_env.device)
    policy.load_state_dict(ckpt)
    policy.eval()
    print(f"  Loaded checkpoint: {ckpt_path}")

    t0 = time.time()
    raw_stats = run_eval(env, policy, base_env, num_seeds, training_seed)
    elapsed = time.time() - t0
    print(f"  Eval done in {elapsed:.1f}s")

    result = {}
    for k, v in raw_stats.items():
        result[k] = compute_stats(v)

    result["elapsed_s"] = elapsed
    result["label"] = label
    result["ckpt"] = ckpt_path
    result["num_seeds"] = num_seeds

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    # Pretty print
    print(f"\n{'='*60}")
    print(f"Results: {label}")
    print(f"{'='*60}")
    key_names = ["collision", "min_obstacle_dist", "return", "tracking_error",
                 "filter_activation", "detour_ratio", "over_clearance",
                 "episode_len", "correction"]
    for k in key_names:
        if k in result:
            r = result[k]
            print(f"  {k:25s}: {r['mean']:.4f} ± {r['std']:.4f}  (n={r['n']})")
    print(f"\nSaved to {output_path}")
    simulation_app.close()


if __name__ == "__main__":
    main()

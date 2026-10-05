"""Train from a config file and save the run.

    uv run python -m src.ml.train configs/ppo_gust_dummy.toml [--name NAME]

The first line printed is `RUN_DIR <path>` so a launcher (the UI) can find the run.
Create a file called STOP in the run directory to stop early; weights are still saved.
"""

import argparse
from datetime import datetime
from pathlib import Path

import torch

from src.config import CONFIG
from src.ml.registry import build_algo
from src.ml.run_config import RunConfig
from src.ml.runs import RunDir, make_env, make_policy

CHECKPOINT_EVERY = 10  # iterations


def train(cfg: RunConfig, runs_root: Path = CONFIG.runs_dir) -> RunDir:
    run = RunDir.create(cfg, runs_root)
    print(f"RUN_DIR {run.path}", flush=True)

    torch.manual_seed(cfg.seed)  # policy sampling and minibatch shuffling use the global RNG
    env = make_env(cfg, n_envs=cfg.algo.n_envs, seed=cfg.seed)
    policy = make_policy(cfg)
    algo = build_algo(cfg.algo)
    status = run.read_status()

    def on_iteration(it: int, metrics: dict[str, float]) -> bool:
        run.append_metrics(it, metrics)
        status.iterations_done = it + 1
        status.final_return = metrics.get("train_return")
        if (it + 1) % CHECKPOINT_EVERY == 0:
            policy.save(run.policy_path)
            run.write_status(status)
        print(f"iter {it:4d}  " + "  ".join(f"{k} {v:9.4f}" for k, v in metrics.items()), flush=True)
        return not run.stop_requested()

    try:
        algo.train(env, policy, on_iteration)
        status.state = "stopped" if run.stop_requested() else "done"
    except KeyboardInterrupt:
        status.state = "stopped"
    except Exception as e:
        status.state = "failed"
        status.error = repr(e)
        raise
    finally:
        policy.save(run.policy_path)
        status.finished = datetime.now()
        run.write_status(status)
        print(f"FINISHED {status.state}", flush=True)
    return run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path, help="TOML run config")
    parser.add_argument("--name", help="override the run name in the config")
    args = parser.parse_args()

    cfg = RunConfig.from_toml(args.config)
    if args.name:
        cfg = cfg.model_copy(update={"name": args.name})
    train(cfg)


if __name__ == "__main__":
    main()

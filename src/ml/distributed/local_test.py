"""Run the whole decoupled system on one machine: learner + actor as separate processes, as in the tunnel.

    uv run python -m src.ml.distributed.local_test      (edit the settings below first)

What it does, in the same order you would by hand:
    1. zero-deflection baseline episodes, so the result has something to compare against
    2. start the learner  -> it writes SHARED/config.json and waits for data
    3. start the actor    -> warmup episodes, then picks up weights between episodes
    4. after EPISODES, write SHARED/STOP and wait for the learner to finish
    5. summary: returns over time vs the baseline, weight versions used, staleness, loop timing

The shared folder goes in the system temp folder (not OneDrive) and is deleted afterwards unless KEEP_SHARED.
On one machine the learner competes with the actor for the CPU, so expect worse loop timing than in the tunnel,
where they run on different computers.
"""

import multiprocessing as mp
import shutil
import tempfile
import time
from pathlib import Path

import polars as pl

from src.config import CONFIG
from src.ml.distributed.actor import run_actor
from src.ml.distributed.files import config_path, new_episodes, read_episode, stop_path
from src.ml.distributed.learner import run_learner
from src.ml.evaluate import evaluate
from src.ml.run_config import RunConfig
from src.ml.runs import RunDir, make_env

# ----- settings -----
CONFIG_FILE = Path("configs/sac_tunnel_mock.toml")  # algo must be SAC
EPISODES = 70  # actor episodes before stopping
N_ENVS = 1  # actor parallel envs (sim envs only; the tunnel is always 1)
BASELINE_EPISODES = 3  # zero-deflection episodes to compare against (0 to skip)
KEEP_SHARED = False  # keep the shared folder afterwards to inspect the files

START_TIMEOUT_S = 60.0  # learner start-up (imports, building the networks)
STOP_TIMEOUT_S = 60.0  # learner finishing its current chunk of updates after STOP


def baseline(cfg: RunConfig, n: int) -> float | None:
    """Mean return of zero-deflection episodes: the number a controller has to beat."""
    if n <= 0:
        return None
    env = make_env(cfg, n_envs=1, seed=cfg.seed + 1000)
    returns = [evaluate(env, None, seed=cfg.seed + 1000 + i)[0].mean_return for i in range(n)]
    env.close()  # releases the (mock) rig
    return sum(returns) / len(returns)


def newest_run(name: str) -> RunDir | None:
    """The learner's run directory (it runs in another process, so find it by name)."""
    runs = sorted(CONFIG.runs_dir.glob(f"*_{name}"))
    return RunDir(runs[-1]) if runs else None


def summarise(shared: Path, base: float | None, run_dir: RunDir | None) -> None:
    eps = [read_episode(p) for p in new_episodes(shared, set())]
    if not eps:
        print("\nno episodes were written")
        return
    df = pl.DataFrame([
        {
            "kind": "eval" if m.evaluate else ("warmup" if m.policy_version < 0 else "train"),
            "version": m.policy_version,
            "return": float(f["rew"].sum()) / m.n_envs,
            "dt_max_ms": 1000 * m.loop_dt_max,
        }
        for m, f in eps
    ])
    print(f"\n===== summary: {df.height} episodes =====")
    if base is not None:
        print(f"zero-deflection baseline : {base:9.2f}")
    for kind in ("warmup", "train", "eval"):
        part = df.filter(pl.col("kind") == kind)
        if part.height:
            q = max(1, part.height // 4)
            first, last = part.head(q)["return"].mean(), part.tail(q)["return"].mean()
            print(f"{kind:7s} episodes {part.height:4d} : first quarter {first:9.2f}  ->  last quarter {last:9.2f}")
    versions = df.filter(pl.col("version") >= 0)["version"]
    if versions.len():
        print(f"weight versions used     : v{versions.min()} .. v{versions.max()} ({versions.n_unique()} distinct)")
    print(f"worst loop period        : {df['dt_max_ms'].max():.2f} ms")
    if run_dir is not None:
        m = run_dir.read_metrics()
        if m.height:
            last = m.row(m.height - 1, named=True)
            staleness = m["staleness"].cast(pl.Float64).fill_nan(None).mean()
            print(
                f"learner                  : {int(last['version'])} versions, {int(last['updates'])} updates, "
                f"mean staleness {staleness or 0.0:.2f} versions"
            )
        print(f"run directory            : {run_dir.path}")


def main() -> None:
    cfg = RunConfig.from_toml(CONFIG_FILE)
    cfg = cfg.model_copy(update={"name": f"{cfg.name}_local"})
    shared = Path(tempfile.mkdtemp(prefix="rl_shared_"))
    print(f"shared folder: {shared}")

    base = baseline(cfg, BASELINE_EPISODES)
    if base is not None:
        print(f"zero-deflection baseline: {base:.2f} per episode")

    # Separate processes, exactly like the tunnel: they share nothing but the folder.
    learner = mp.Process(target=run_learner, args=(cfg, shared), name="learner")
    actor = mp.Process(target=run_actor, args=(shared, N_ENVS, None, EPISODES), name="actor")
    learner.start()
    try:
        # The actor needs config.json, which the learner writes on start-up.
        deadline = time.monotonic() + START_TIMEOUT_S
        while not config_path(shared).exists():
            if not learner.is_alive() or time.monotonic() > deadline:
                raise SystemExit("learner did not start (see its output above)")
            time.sleep(0.2)
        actor.start()
        actor.join()
        stop_path(shared).touch()
        learner.join(timeout=STOP_TIMEOUT_S)
    except KeyboardInterrupt:
        print("\ninterrupted: stopping both processes")
        stop_path(shared).touch()
        actor.join(timeout=STOP_TIMEOUT_S)
        learner.join(timeout=STOP_TIMEOUT_S)
    finally:
        for proc in (actor, learner):
            if proc.is_alive():
                proc.terminate()

    summarise(shared, base, newest_run(cfg.name))
    if KEEP_SHARED:
        print(f"shared folder kept: {shared}")
    else:
        shutil.rmtree(shared, ignore_errors=True)


if __name__ == "__main__":
    main()

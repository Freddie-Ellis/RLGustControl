from pydantic import BaseModel

from src.ml.envs.base import ActFn, Env, Trace
from src.ml.envs.wind_tunnel import WindTunnelParams
from src.ml.run_config import RunConfig
from src.ml.runs import make_env


class EvalResult(BaseModel):
    """Summary of one batched evaluation rollout."""
    mean_return: float  # undiscounted episode return, averaged over parallel envs
    action_rms: float  # control effort
    env_metrics: dict[str, float]  # from Env.trace_metrics, e.g. lift error


def summarise(env: Env, trace: Trace) -> EvalResult:
    """Compute the metrics of an already recorded rollout."""
    return EvalResult(
        mean_return=float(trace.reward.sum(0).mean()),
        action_rms=float(trace.action.pow(2).mean().sqrt()),
        env_metrics=env.trace_metrics(trace),
    )


def evaluate(env: Env, act_fn: ActFn | None = None, seed: int = 999) -> tuple[EvalResult, Trace]:
    """Run one episode in every parallel env from a fixed seed, so different policies see the same gusts.

    `act_fn=None` evaluates the zero-action (uncontrolled) baseline.
    """
    env.seed(seed)
    trace = env.run(act_fn)
    return summarise(env, trace), trace


def replay_refusal(cfg: RunConfig) -> str | None:
    """Why a run cannot be replayed from here, or None if it can. A replay builds the run's env and steps it, which
    for a wind tunnel run means driving the rig: only the mock may be driven from a plot window."""
    params = cfg.env.params
    if isinstance(params, WindTunnelParams) and params.io != "mock":
        return (
            f"This run's env is the real wind tunnel ({params.io}). Replaying it would move the surfaces from here; "
            "run it as a session (SAC Session tab), where the actor, the safety layer and STOP are in charge."
        )
    return None


def replay(
    cfg: RunConfig, act_fn: ActFn, n_envs: int, seed: int, baseline: bool
) -> tuple[tuple[EvalResult, Trace], tuple[EvalResult, Trace] | None]:
    """Roll `act_fn` out on `cfg`'s env, then (optionally) the zero-action baseline from the same seed. A wind tunnel
    env is one rig, so it runs one episode whatever `n_envs` says, in real time. The env is always closed again."""
    if (reason := replay_refusal(cfg)) is not None:
        raise ValueError(reason)
    env = make_env(cfg, n_envs=1 if isinstance(cfg.env.params, WindTunnelParams) else n_envs)
    try:
        return evaluate(env, act_fn, seed), (evaluate(env, None, seed) if baseline else None)
    finally:
        env.close()

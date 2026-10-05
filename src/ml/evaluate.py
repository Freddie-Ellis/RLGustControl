from pydantic import BaseModel

from src.ml.envs.base import ActFn, Env, Trace


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

"""Reward functions, kept separate from the envs so the mock and the real tunnel compute exactly the same thing."""

from torch import Tensor


def pitch_reward(
    cm: Tensor, u: Tensor, prev_u: Tensor, cm_scale: float, action_cost: float, rate_cost: float
) -> Tensor:
    """Hold Cm at 0 with small, smooth deflections. cm [N]; u, prev_u [N, n_surfaces] normalised to [-1, 1].

    `cm_scale` is a typical uncontrolled Cm excursion. Dividing by it keeps the tracking term around 1 whatever
    the rig's absolute Cm, so it stays comparable to SAC's entropy bonus. The action terms use the mean over
    surfaces, so their weight does not change with the number of surfaces.
    """
    tracking = (cm / cm_scale) ** 2
    effort = u.pow(2).mean(-1)  # actuator load / power
    rate = (u - prev_u).pow(2).mean(-1)  # chatter wears servos and excites the model
    return -(tracking + action_cost * effort + rate_cost * rate)

"""Reward functions, kept separate from the envs so the mock and the real tunnel compute exactly the same thing."""

from torch import Tensor


def tracking_reward(
    error: Tensor, weights: Tensor, u: Tensor, prev_u: Tensor, action_cost: float, rate_cost: float
) -> Tensor:
    """Hold each tracked load at its target with small, smooth deflections.

    error [N, K]: (load - target) / scale for each of the K tracked loads, where `scale` is the load's typical
        uncontrolled excursion in gusts, so a typical miss costs about its weight whatever the load's absolute size,
        and the tracking term stays comparable to SAC's entropy bonus.
    weights [K]: how much each tracked load counts.
    u, prev_u [N, n_surfaces]: deflections now and one step before, normalised to [-1, 1] (0 = neutral). The action
        terms use the mean over surfaces, so their weight does not change with the number of surfaces.

    E.g. tracking C_L and C_m with scales 0.05 and 0.01, 0.05 too much lift and 0.01 too little pitch cost
    (0.05 / 0.05)^2 + (0.01 / 0.01)^2 = 2 before the action terms.
    """
    tracking = (weights * error.pow(2)).sum(-1)
    effort = u.pow(2).mean(-1)  # actuator load / power
    rate = (u - prev_u).pow(2).mean(-1)  # chatter wears servos and excites the model
    return -(tracking + action_cost * effort + rate_cost * rate)

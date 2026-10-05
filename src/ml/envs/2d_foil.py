'''Pitch disturbed 2D air foil environment for reinforcement learning.

This will be hooked up to real simulation in XFoil and use pressure as the observation and controls the angle of a flap.
'''

import torch
from pydantic import BaseModel
from torch import Tensor

from src.ml.envs.base import Env, EnvSpec


class XFOILPitchProblemParams(BaseModel):
    """Parameters for the pitch disturbed 2D air foil environment."""

    tau: float = 0.3
    slope: float = 1.0
    cl_ref: float = 0.5
    gust_len: float = 1.0
    act_limit: float = 1.5
    action_cost: float = 0.01
    naca: str = "0012"


class XFOILPitchProblemEnv(Env):

    no_sensors: int = 10
    obs_dim = no_sensors * 2 + 1 # [pressure readings from sensors, rate of change, previous action]
    act_dim = 1 # [flap angle]

    def __init__(self, spec: EnvSpec, params: XFOILPitchProblemParams, n_envs: int, seed: int | None = None) -> None:
        super().__init__(spec, n_envs, seed)
        self.params = params
        # Initialize state variables for the environment
        self.pressures = torch.zeros((n_envs, self.no_sensors))
        self.prev_u = torch.zeros(n_envs)

        self.setupXFOIL(naca=self.params.naca)

    def _obs(self, pressure_dot: torch.Tensor) -> torch.Tensor:
        """Construct the observation vector from the current state."""
        return torch.cat([self.pressures, 0.1 * pressure_dot.unsqueeze(-1), self.prev_u.unsqueeze(-1)], dim=-1)

    def reset(self) -> torch.Tensor:
        """Reset the environment to an initial state and return the initial observation."""
        n = self.n_envs
        self.t = 0
        # Reset pressures and previous actions
        self.pressures = torch.zeros((n, self.no_sensors))
        self.prev_u = torch.zeros(n)
        return self._obs(torch.zeros(n))

    def step(self, action: Tensor) -> tuple[Tensor, Tensor]:
        u = action.squeeze(-1).clamp(-self.params.act_limit, self.params.act_limit)

        self.runXFOIL()
        
        return 

    def setupXFOIL(self, naca: str) -> None:
        '''Create a config foil for XFOIl'''

    def parseNACA(self, naca: str) -> None:
        return 
    
    def runXFOIL(self) -> None:
        return

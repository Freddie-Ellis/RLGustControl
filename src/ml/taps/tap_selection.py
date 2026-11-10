"""Pressure tap placement from CFD surface pressure fields.

Taps can only go on parts of the wing that never move, so the candidates are a fixed set of points
there (e.g. the undeflected case's surface nodes away from the actuators). Every case is remeshed
around its deflected actuators, so its surface Cp is first interpolated onto these points with
`candidate_cp`; after that, column j means the same physical point in every case. The targets
(normally the moments the controller has to hold) come from the solver's own force report, which
integrates over each case's full deflected surface.

Taps are chosen greedily: each new tap goes at the candidate point that most reduces the
uncertainty of the targets given the taps already chosen, treating the pressures as Gaussian with
the covariance measured across the cases and taps as noisy sensors.

Inputs the controller already knows, i.e. its own actuator deflections, can be passed as `known`.
Their linear effect is accounted for before any tap is chosen, so taps go where the moment changes
the actuators do not explain (mainly the gust, and how it alters actuator effectiveness) show up.

The sensor noise also regularises the fit: with only a few hundred cases, a noise-free criterion
would explain the training cases perfectly once the number of taps approaches the number of
cases. Check a selection on held-out cases with `evaluate`.

Working in the space of cases keeps every step O(n_cases * n_candidates), so ~1e5 candidate
points are fine.
"""

import math
from dataclasses import dataclass

import torch
from torch import Tensor

N_TAPS: int = 50
SENSOR_NOISE_CP: float = 0.005  # placeholder: tap measurement noise std, in Cp
KNOWN_JITTER: float = 1e-10  # keeps the estimator solve well posed; known inputs are noise free
INTERP_NEIGHBOURS: int = 4  # case surface nodes used to interpolate Cp at each candidate point
INTERP_CHUNK: int = 2048  # candidate points per distance computation, bounds memory


@dataclass(frozen=True)
class CandidatePoints:
    """Fixed points on the non-moving wing surface where a tap may go, in body axes."""

    coords: Tensor  # (C, 3)
    surface: Tensor  # (C,) int surface id, e.g. 0 = upper, 1 = lower


@dataclass(frozen=True)
class CaseSurface:
    """One CFD case's surface solution on its own (remeshed) surface nodes."""

    coords: Tensor  # (M, 3) node positions
    surface: Tensor  # (M,) int surface id, same convention as CandidatePoints
    cp: Tensor  # (M,)


def interpolate_cp(case: CaseSurface, points: CandidatePoints, max_distance: float | None = None) -> Tensor:
    """Cp at each candidate point (C,), by inverse-distance weighting of the nearest case nodes.

    Only nodes on the same surface are used, so thin edges never mix upper and lower pressures.
    If `max_distance` is given, raise when a point's nearest node is further than that, which
    means the point is not on this case's surface (e.g. it sits on a moved actuator).
    """
    cp = torch.empty(len(points.coords), dtype=torch.float64)
    for surface in points.surface.unique().tolist():
        at_points = (points.surface == surface).nonzero().squeeze(1)
        on_case = case.surface == surface
        nodes = case.coords[on_case].to(torch.float64)
        node_cp = case.cp[on_case].to(torch.float64)
        k = min(INTERP_NEIGHBOURS, len(nodes))
        for chunk in at_points.split(INTERP_CHUNK):
            dist, idx = torch.cdist(points.coords[chunk].to(torch.float64), nodes).topk(k, largest=False)
            if max_distance is not None and (dist[:, 0] > max_distance).any():
                raise ValueError(f"candidate points on surface {surface} are not on this case's surface")
            weights = 1.0 / dist.clamp_min(1e-12)
            cp[chunk] = (weights * node_cp[idx]).sum(dim=1) / weights.sum(dim=1)
    return cp


def candidate_cp(cases: list[CaseSurface], points: CandidatePoints, max_distance: float | None = None) -> Tensor:
    """(N, C) Cp at every candidate point for each of N cases."""
    return torch.stack([interpolate_cp(case, points, max_distance) for case in cases])


@dataclass(frozen=True)
class TapSelection:
    taps: Tensor  # (n_taps,) candidate point indices, in the order they were chosen
    # (n_taps + 1, K) fraction of each target's variance left after the known inputs and the first k taps
    unexplained: Tensor
    pressure_mean: Tensor  # (n_taps,) mean Cp at each tap over the training cases
    known_mean: Tensor  # (J,) mean of each known input; J = 0 if none were given
    target_mean: Tensor  # (K,)
    estimator: Tensor  # (J + n_taps, K) linear map from [known inputs, tap pressures] to targets

    def estimate(self, tap_cp: Tensor, known: Tensor | None = None) -> Tensor:
        """Targets (n, K) from Cp at the taps (n, n_taps) and the known inputs (n, J)."""
        features = tap_cp.to(torch.float64) - self.pressure_mean
        if len(self.known_mean):
            if known is None:
                raise ValueError("this selection was made with known inputs; pass them to estimate()")
            features = torch.cat([known.to(torch.float64) - self.known_mean, features], dim=1)
        return self.target_mean + features @ self.estimator


def select_taps(
    cp: Tensor,
    targets: Tensor,
    points: CandidatePoints,
    n_taps: int = N_TAPS,
    known: Tensor | None = None,
    target_weights: Tensor | None = None,
    noise_std: float = SENSOR_NOISE_CP,
    min_spacing: float = 0.0,
    max_per_surface: dict[int, int] | None = None,
) -> TapSelection:
    """Greedily choose `n_taps` tap locations.

    cp: (N, C) Cp at every candidate point for each of N cases, from `candidate_cp`.
    targets: (N, K) quantities to estimate from the taps, normally the controlled moments from the
        solver's force report. Constant columns must be dropped.
    known: (N, J) inputs the controller already knows in each case (its actuator deflections).
    target_weights: (K,) relative importance; each target is first normalised by its variance.
    min_spacing: no two taps on the same surface closer than this (mesh units).
    max_per_surface: optional cap on taps per surface id, e.g. {0: 30, 1: 20}.
    """
    n_cases, n_points = cp.shape
    dtype = torch.float64
    scale = 1.0 / math.sqrt(n_cases - 1)
    # centred and scaled so that a^T b is the covariance across cases
    x = (cp.to(dtype) - cp.to(dtype).mean(dim=0)) * scale  # (N, C)
    t = (targets.to(dtype) - targets.to(dtype).mean(dim=0)) * scale  # (N, K)
    t_var = (t * t).sum(dim=0)
    if (t_var <= 0).any():
        constant = (t_var <= 0).nonzero().squeeze(1).tolist()
        raise ValueError(f"targets {constant} are constant across cases; drop them")
    weights = torch.ones(t.shape[1], dtype=dtype) if target_weights is None else target_weights.to(dtype)
    weights = weights / t_var
    noise = noise_std**2
    k = torch.empty(n_cases, 0, dtype=dtype) if known is None else known.to(dtype)
    k = (k - k.mean(dim=0)) * scale

    # Q is the N x N "what the known inputs and taps have not explained" operator; keep Q x and Q t
    # up to date with rank-1 updates instead of forming Q. It starts as the projection that removes
    # the linear effect of the known inputs.
    qx = x - k @ torch.linalg.lstsq(k, x).solution if k.shape[1] else x.clone()
    qt = t - k @ torch.linalg.lstsq(k, t).solution if k.shape[1] else t.clone()
    available = torch.ones(n_points, dtype=torch.bool)
    cand_coords = points.coords.to(dtype)
    cand_surface = points.surface
    per_surface: dict[int, int] = {}
    chosen: list[int] = []
    unexplained = [(t * qt).sum(dim=0) / t_var]

    for _ in range(n_taps):
        variance = noise + (x * qx).sum(dim=0)  # (C,) uncertainty of each candidate's reading
        explained = qt.T @ x  # (K, C) covariance of what is left of each target with each candidate
        score = (weights[:, None] * explained**2).sum(dim=0) / variance
        score[~available] = -torch.inf
        j = int(score.argmax())
        if not torch.isfinite(score[j]):
            break  # constraints leave no candidates
        qxj = qx[:, j].clone()
        qx -= torch.outer(qxj, qxj @ x) / variance[j]
        qt -= torch.outer(qxj, qxj @ t) / variance[j]
        chosen.append(j)
        unexplained.append((t * qt).sum(dim=0) / t_var)

        surface = int(cand_surface[j])
        same_surface = cand_surface == surface
        too_close = (cand_coords - cand_coords[j]).norm(dim=1) < min_spacing
        available &= ~(same_surface & too_close)
        available[j] = False
        per_surface[surface] = per_surface.get(surface, 0) + 1
        if max_per_surface is not None and per_surface[surface] >= max_per_surface.get(surface, n_taps):
            available &= ~same_surface

    picked = torch.tensor(chosen, dtype=torch.long)
    features = torch.cat([k, x[:, picked]], dim=1)
    penalty = torch.cat([torch.full((k.shape[1],), KNOWN_JITTER), torch.full((len(chosen),), noise)])
    gram = features.T @ features + torch.diag(penalty.to(dtype))
    estimator = torch.linalg.solve(gram, features.T @ t)
    return TapSelection(
        taps=picked,
        unexplained=torch.stack(unexplained),
        pressure_mean=cp[:, picked].to(dtype).mean(dim=0),
        known_mean=torch.empty(0, dtype=dtype) if known is None else known.to(dtype).mean(dim=0),
        target_mean=targets.to(dtype).mean(dim=0),
        estimator=estimator,
    )


def evaluate(
    selection: TapSelection,
    cp: Tensor,
    targets: Tensor,
    known: Tensor | None = None,
    noise_std: float = SENSOR_NOISE_CP,
) -> Tensor:
    """RMS estimation error of each target on (held-out) cases, as a fraction of its std. (K,)

    cp: (N, C) Cp at the candidate points, as passed to `select_taps`.

    Sensor noise of `noise_std` is added to the tap readings.
    """
    tap_cp = cp[:, selection.taps].to(torch.float64)
    tap_cp = tap_cp + noise_std * torch.randn_like(tap_cp)
    error = selection.estimate(tap_cp, known) - targets.to(torch.float64)
    return error.pow(2).mean(dim=0).sqrt() / targets.to(torch.float64).std(dim=0)

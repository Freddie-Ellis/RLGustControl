"""Pressure tap placement from CFD surface pressure fields.

Taps can only go on parts of the wing that never move, so the candidates are a fixed set of points
there (e.g. the undeflected case's surface nodes away from the actuators). Every case is remeshed
around its deflected actuators, so its surface Cp is first interpolated onto these points with an
`Interpolation` built once per mesh; after that, column j means the same physical point in every case. The targets
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
cases. Check a selection on held-out cases with `evaluate`, or with `heldout_curve` for every number of taps.

Working in the space of cases keeps every step O(n_cases * n_candidates), so ~1e5 candidate
points are fine.
"""

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from src.config import CONFIG

N_TAPS: int = CONFIG.rig.n_taps  # the wing itself is defined once, in src/config.py
SENSOR_NOISE_CP: float = 0.005  # placeholder: tap measurement noise std, in Cp
KNOWN_JITTER: float = 1e-10  # keeps the estimator solve well posed; known inputs are noise free
INTERP_NEIGHBOURS: int = 4  # case surface nodes used to interpolate Cp at each candidate point
INTERP_CHUNK: int = 512  # candidate points per distance computation, bounds memory (chunk x nodes doubles)


@dataclass(frozen=True)
class CandidatePoints:
    """Fixed points on the non-moving wing surface where a tap may go, in body axes."""

    coords: Tensor  # (C, 3)
    surface: Tensor  # (C,) int surface id, e.g. 0 = upper, 1 = lower
    # (C,) index of each point's mirror image across the symmetry plane, itself for a point on it; None if the wing is
    # not treated as symmetric. With it, taps are chosen and placed in mirror pairs.
    mirror: Tensor | None = None


@dataclass(frozen=True)
class Interpolation:
    """Linear map from Cp on one mesh's surface nodes to Cp at the candidate points.

    Every run of one mesh shares it, so the neighbour search is done once per mesh, not per run.
    """

    nodes: Tensor  # (C, k) mesh nodes used for each candidate point
    weights: Tensor  # (C, k) inverse-distance weights, each row sums to 1
    nearest: Tensor  # (C,) distance to the nearest node: large means the point is not on this mesh's surface

    def __call__(self, node_cp: Tensor) -> Tensor:
        """Cp on the mesh nodes (..., M) -> Cp at the candidate points (..., C)."""
        return (node_cp.to(torch.float64)[..., self.nodes] * self.weights).sum(dim=-1)


def interpolation(coords: Tensor, surface: Tensor, points: CandidatePoints) -> Interpolation:
    """Inverse-distance weighting of the `INTERP_NEIGHBOURS` nearest mesh nodes (M, 3) to each candidate point.

    Only nodes with the same surface id (M,) are used, so thin edges never mix upper and lower pressures.
    """
    n_points = len(points.coords)
    k = INTERP_NEIGHBOURS
    nodes = torch.zeros(n_points, k, dtype=torch.long)
    weights = torch.zeros(n_points, k, dtype=torch.float64)
    nearest = torch.full((n_points,), torch.inf, dtype=torch.float64)
    for s in points.surface.unique().tolist():
        on_mesh = (surface == s).nonzero().squeeze(1)
        if len(on_mesh) < k:
            continue  # no nodes on this surface: its points stay at infinite distance
        mesh_coords = coords[on_mesh].to(torch.float64)
        for chunk in (points.surface == s).nonzero().squeeze(1).split(INTERP_CHUNK):
            dist, idx = torch.cdist(points.coords[chunk].to(torch.float64), mesh_coords).topk(k, largest=False)
            inverse = 1.0 / dist.clamp_min(1e-12)
            nodes[chunk] = on_mesh[idx]
            weights[chunk] = inverse / inverse.sum(dim=1, keepdim=True)
            nearest[chunk] = dist[:, 0]
    return Interpolation(nodes=nodes, weights=weights, nearest=nearest)


@dataclass(frozen=True)
class TapSelection:
    taps: Tensor  # (n_taps,) candidate point indices, in the order they were chosen
    # (n_taps + 1, K) fraction of each target's variance left after the known inputs and the first k taps
    unexplained: Tensor
    pressure_mean: Tensor  # (n_taps,) mean Cp at each tap over the training cases
    known_mean: Tensor  # (J,) mean of each known input; J = 0 if none were given
    target_mean: Tensor  # (K,)
    target_std: Tensor  # (K,) over the training cases, the scale errors are reported in
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
    fixed: Tensor | None = None,
) -> TapSelection:
    """Greedily choose `n_taps` tap locations.

    fixed: candidate indices placed first, in this order, before any greedy choice (e.g. taps placed by hand). The
        greedy taps then go where these leave the targets unexplained. `n_taps` counts them, so with
        `n_taps == len(fixed)` this just scores a given layout: its `unexplained` curve and estimator.

    If `points.mirror` is given, each greedy choice is a mirror pair, scored jointly and placed together (a point on
    the symmetry plane is one tap on its own), so the layout is symmetric whenever `fixed` is.

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

    placed = [] if fixed is None else fixed.tolist()
    mirror = points.mirror
    # True for a point that is its own mirror image (on the symmetry plane), or every point without symmetry
    single = torch.ones(n_points, dtype=torch.bool) if mirror is None else mirror == torch.arange(n_points)

    def add(j: int) -> None:
        """Place a tap at candidate j: a rank-1 update of Q x and Q t, then the placement constraints."""
        variance_j = noise + qx[:, j] @ x[:, j]
        qxj = qx[:, j].clone()
        qx.sub_(torch.outer(qxj, qxj @ x) / variance_j)
        qt.sub_(torch.outer(qxj, qxj @ t) / variance_j)
        chosen.append(j)
        unexplained.append((t * qt).sum(dim=0) / t_var)
        surface = int(cand_surface[j])
        same_surface = cand_surface == surface
        too_close = (cand_coords - cand_coords[j]).norm(dim=1) < min_spacing
        available.logical_and_(~(same_surface & too_close))
        available[j] = False
        per_surface[surface] = per_surface.get(surface, 0) + 1
        if max_per_surface is not None and per_surface[surface] >= max_per_surface.get(surface, n_taps):
            available.logical_and_(~same_surface)

    while len(chosen) < n_taps:
        if len(chosen) < len(placed):
            add(int(placed[len(chosen)]))
            continue
        variance = noise + (x * qx).sum(dim=0)  # (C,) uncertainty of each candidate's reading
        explained = qt.T @ x  # (K, C) covariance of what is left of each target with each candidate
        score = (weights[:, None] * explained**2).sum(dim=0) / variance
        allowed = available.clone()
        if mirror is not None:
            # A pair {j, m} reduces the weighted target variance by e^T V^-1 e, with e = (e_j, e_m) the two readings'
            # covariances with what is left of a target and V their 2 x 2 covariance (noise on the diagonal). Scored
            # per tap, so pairs and taps on the symmetry plane compete fairly.
            cross = (x[:, mirror] * qx).sum(dim=0)  # covariance of each reading with its mirror's, given the taps so far
            v_m, e_m = variance[mirror], explained[:, mirror]
            det = variance * v_m - cross**2
            pair = (weights[:, None] * (v_m * explained**2 - 2 * cross * explained * e_m + variance * e_m**2)).sum(
                dim=0
            ) / det
            score = torch.where(single, score, pair / 2)
            allowed &= available[mirror]
            if n_taps - len(chosen) < 2:
                allowed &= single  # room for one more tap only
        score[~allowed] = -torch.inf
        j = int(score.argmax())
        if not torch.isfinite(score[j]):
            break  # constraints leave no candidates
        add(j)
        if not single[j]:
            add(int(mirror[j]))  # type: ignore[index]  # not single implies a mirror

    picked = torch.tensor(chosen, dtype=torch.long)
    return fit_estimator(cp, targets, picked, known, noise_std, unexplained=torch.stack(unexplained))


def farthest_point_fill(points: CandidatePoints, taps: Tensor, n_fill: int) -> Tensor:
    """`n_fill` more taps spread evenly over the wing, around the `taps` already placed. (n_fill,) candidate indices.

    Each new tap goes at the candidate furthest, in planform (x, y), from every tap on its own surface, so the gaps
    are filled largest first. A surface with no tap yet counts as infinitely far, so every surface gets covered.
    With `points.mirror`, each tap brings its mirror image (while `n_fill` leaves room for both).
    """
    planform = points.coords[:, :2].to(torch.float64)
    gap = torch.full((len(planform),), torch.inf, dtype=torch.float64)  # distance to the nearest same-surface tap
    for j in taps.tolist():
        _shrink_gap(gap, planform, points.surface, j)
    fill: list[int] = []
    while len(fill) < n_fill:
        j = int(gap.argmax())
        if gap[j] <= 0:
            break  # every candidate already holds a tap
        pair = [j] if points.mirror is None else list(dict.fromkeys([j, int(points.mirror[j])]))
        for k in pair[: n_fill - len(fill)]:
            fill.append(k)
            _shrink_gap(gap, planform, points.surface, k)
    return torch.tensor(fill, dtype=torch.long)


def _shrink_gap(gap: Tensor, planform: Tensor, surface: Tensor, tap: int) -> None:
    same = surface == surface[tap]
    gap[same] = torch.minimum(gap[same], (planform[same] - planform[tap]).norm(dim=1))


def fit_estimator(
    cp: Tensor,
    targets: Tensor,
    taps: Tensor,
    known: Tensor | None = None,
    noise_std: float = SENSOR_NOISE_CP,
    unexplained: Tensor | None = None,
) -> TapSelection:
    """The linear estimator of the targets from the known inputs and Cp at `taps`, fitted on these cases.

    `unexplained` is stored as given (empty if None): it describes how `taps` were chosen, not this fit.
    """
    dtype = torch.float64
    scale = 1.0 / math.sqrt(cp.shape[0] - 1)
    x = (cp[:, taps].to(dtype) - cp[:, taps].to(dtype).mean(dim=0)) * scale
    t = (targets.to(dtype) - targets.to(dtype).mean(dim=0)) * scale
    k = torch.empty(cp.shape[0], 0, dtype=dtype) if known is None else known.to(dtype)
    k = (k - k.mean(dim=0)) * scale
    features = torch.cat([k, x], dim=1)
    penalty = torch.cat([torch.full((k.shape[1],), KNOWN_JITTER), torch.full((len(taps),), noise_std**2)])
    gram = features.T @ features + torch.diag(penalty.to(dtype))
    return TapSelection(
        taps=taps,
        unexplained=torch.empty(0, targets.shape[1], dtype=dtype) if unexplained is None else unexplained,
        pressure_mean=cp[:, taps].to(dtype).mean(dim=0),
        known_mean=torch.empty(0, dtype=dtype) if known is None else known.to(dtype).mean(dim=0),
        target_mean=targets.to(dtype).mean(dim=0),
        target_std=targets.to(dtype).std(dim=0),
        estimator=torch.linalg.solve(gram, features.T @ t),
    )


def evaluate(
    selection: TapSelection,
    cp: Tensor,
    targets: Tensor,
    known: Tensor | None = None,
    noise_std: float = SENSOR_NOISE_CP,
) -> Tensor:
    """RMS estimation error of each target on (held-out) cases, as a fraction of its std over the training cases. (K,)

    cp: (N, C) Cp at the candidate points, as passed to `select_taps`. Using the training std keeps the scale
    meaningful however few held-out cases there are.

    Sensor noise of `noise_std` is added to the tap readings.
    """
    tap_cp = cp[:, selection.taps].to(torch.float64)
    tap_cp = tap_cp + noise_std * torch.randn_like(tap_cp)
    error = selection.estimate(tap_cp, known) - targets.to(torch.float64)
    return error.pow(2).mean(dim=0).sqrt() / selection.target_std


def heldout_curve(
    selection: TapSelection,
    train_cp: Tensor,
    train_targets: Tensor,
    test_cp: Tensor,
    test_targets: Tensor,
    train_known: Tensor | None = None,
    test_known: Tensor | None = None,
    noise_std: float = SENSOR_NOISE_CP,
) -> Tensor:
    """`evaluate`'s held-out error using only the first k taps chosen, for k = 0 .. n_taps. (n_taps + 1, K)

    The estimator is refitted on the training cases for each k, so row k is what a k-tap layout would achieve on
    unseen cases. Where it stops falling, further taps fit the training cases rather than add information.
    """
    return torch.stack([
        evaluate(
            fit_estimator(train_cp, train_targets, selection.taps[:k], train_known, noise_std),
            test_cp, test_targets, test_known, noise_std,
        )
        for k in range(len(selection.taps) + 1)
    ])

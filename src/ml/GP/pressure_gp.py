# Sparse Variational Guassian Process for 2D Pressure Prediction
# Inputs pressure distribution from CFD data, using m inducing points

# Sparse Variational Guassian Process for 2D Pressure Prediction
# Inputs pressure distribution from CFD data, using m inducing points
#
# The inducing point locations are free parameters, optimised together with the
# kernel hyperparameters by maximising the Titsias collapsed variational bound.
# After fitting, the inducing locations are the proposed pressure tap positions.

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class FitResult:
    """Training history of a `SparsePressureGP.fit` call."""

    elbo: list[float]


class SparsePressureGP(nn.Module):
    """Sparse GP regression (SGPR) with learnable inducing point locations.

    Inputs `x` have shape (N, d), e.g. surface coordinates (x/c, y/c) of the CFD
    points. Targets `y` have shape (N,) for one pressure distribution, or (N, T)
    for T snapshots (time steps / gust cases) that share the same inducing
    points and kernel, so the taps are chosen to work across all snapshots.

    For 1-D inputs the inducing points are parametrised by the gaps between them, so
    they stay ordered, inside the data range and at least `min_spacing` of the uniform
    spacing apart: they can neither collapse onto each other nor get stuck on an end.
    Higher-dimensional inputs just keep each point inside the data bounding box.

    With `linear_degree` set this is an LMGP: a polynomial linear model (degree 1 is a
    straight line) is first fitted to the data by least squares, and the GP is trained
    on the residual. Predictions are the linear model plus the GP mean.
    """

    def __init__(
        self,
        x: Tensor,
        num_inducing: int,
        *,
        noise: float = 1e-2,
        jitter: float = 1e-6,
        seed: int = 0,
        min_spacing: float = 0.2,
        linear_degree: int | None = None,
    ) -> None:
        super().__init__()
        if num_inducing > x.shape[0]:
            raise ValueError("num_inducing cannot exceed the number of data points")
        if not 0.0 <= min_spacing < 1.0:
            raise ValueError("min_spacing must be in [0, 1)")
        if linear_degree is not None and linear_degree < 1:
            raise ValueError("linear_degree must be at least 1 (or None for a plain GP)")

        x = x.to(torch.float64)
        self.jitter = jitter
        self.num_inducing = num_inducing
        self.min_spacing = min_spacing
        self.linear_degree = linear_degree
        self._ordered = x.shape[1] == 1
        self.register_buffer("lower", x.min(dim=0).values)
        self.register_buffer("upper", x.max(dim=0).values)

        # Kernel hyperparameters (log space keeps them positive)
        self.log_lengthscale = nn.Parameter(
            torch.log((self.upper - self.lower) / 5.0).clone()  # type: ignore[operator]
        )
        self.log_signal_var = nn.Parameter(torch.zeros((), dtype=torch.float64))
        self.log_noise_var = nn.Parameter(
            torch.tensor(math.log(noise), dtype=torch.float64)
        )

        # Inducing points: initialised from a random subset of the data, stored as
        # unconstrained values (see `_to_raw` / `inducing_points`)
        generator = torch.Generator().manual_seed(seed)
        idx = torch.randperm(x.shape[0], generator=generator)[:num_inducing]
        self.raw_z = nn.Parameter(self._to_raw(x[idx]))

        self._x_train: Tensor | None = None
        self._y_train: Tensor | None = None
        self._y_mean: Tensor | None = None
        self._linear_coef: Tensor | None = None

    # --- linear model ------------------------------------------------------------

    def _features(self, x: Tensor) -> Tensor:
        """Polynomial features [1, u, u^2, ..., u^degree] per input dimension, u in [0, 1]."""
        assert self.linear_degree is not None
        u = (x - self.lower) / (self.upper - self.lower)  # type: ignore[operator]
        powers = [u**k for k in range(1, self.linear_degree + 1)]
        return torch.cat([torch.ones_like(u[:, :1]), *powers], dim=1)

    def linear_part(self, x_star: Tensor) -> Tensor:
        """The fitted linear model at `x_star`, shape (M, T); zeros for a plain GP."""
        if self._y_train is None:
            raise RuntimeError("call fit() before linear_part()")
        x_star = x_star.to(torch.float64)
        if self._linear_coef is None:
            return torch.zeros(x_star.shape[0], self._y_train.shape[1], dtype=torch.float64)
        return self._features(x_star) @ self._linear_coef

    # --- parametrisation -------------------------------------------------------

    @property
    def _gap_floor(self) -> float:
        """Smallest allowed gap between neighbouring taps (and to the ends), as a fraction of the range."""
        return self.min_spacing / (self.num_inducing + 1)

    def _to_raw(self, z: Tensor) -> Tensor:
        span = self.upper - self.lower  # type: ignore[operator]
        if self._ordered:
            floor = self._gap_floor
            edges = torch.cat([self.lower, z[:, 0].sort().values, self.upper])  # type: ignore[list-item]
            gaps = (torch.diff(edges) / span[0]).clamp_min(floor)
            gaps = gaps / gaps.sum()
            weights = ((gaps - floor) / (1.0 - (self.num_inducing + 1) * floor)).clamp_min(1e-4)
            return torch.log(weights / weights.sum())
        frac = ((z - self.lower) / span).clamp(1e-3, 1 - 1e-3)  # type: ignore[operator]
        return torch.logit(frac)

    @property
    def inducing_points(self) -> Tensor:
        """Current inducing locations (m, d), constrained to the data bounds."""
        span = self.upper - self.lower  # type: ignore[operator]
        if self._ordered:
            floor = self._gap_floor
            free = 1.0 - (self.num_inducing + 1) * floor
            gaps = floor + free * torch.softmax(self.raw_z, dim=0)  # n + 1 gaps summing to 1
            positions = torch.cumsum(gaps, dim=0)[:-1]
            return self.lower + span * positions.unsqueeze(-1)  # type: ignore[operator]
        return self.lower + span * torch.sigmoid(self.raw_z)  # type: ignore[operator]

    @property
    def noise_var(self) -> Tensor:
        return self.log_noise_var.exp()

    def tap_locations(self) -> Tensor:
        """Optimised inducing points, i.e. the suggested pressure tap positions."""
        return self.inducing_points.detach().clone()

    # --- kernel ----------------------------------------------------------------

    def kernel(self, a: Tensor, b: Tensor) -> Tensor:
        """ARD squared-exponential kernel matrix K(a, b)."""
        ell = self.log_lengthscale.exp()
        sq = torch.cdist(a / ell, b / ell).pow(2)
        return self.log_signal_var.exp() * torch.exp(-0.5 * sq)

    # --- collapsed variational bound --------------------------------------------

    def _factorise(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Return (Luu, A, LB) used by both the bound and the predictions."""
        z = self.inducing_points
        m = z.shape[0]
        eye = torch.eye(m, dtype=x.dtype, device=x.device)
        kuu = self.kernel(z, z) + self.jitter * eye
        luu = torch.linalg.cholesky(kuu)
        kuf = self.kernel(z, x)
        sigma = self.noise_var.sqrt()
        a = torch.linalg.solve_triangular(luu, kuf, upper=False) / sigma
        b = eye + a @ a.T
        lb = torch.linalg.cholesky(b)
        return luu, a, lb

    def elbo(self, x: Tensor, y: Tensor) -> Tensor:
        """Titsias collapsed bound, summed over outputs. `y` must be centred."""
        if y.ndim == 1:
            y = y.unsqueeze(-1)
        n, t = y.shape
        _, a, lb = self._factorise(x)
        sigma2 = self.noise_var
        sigma = sigma2.sqrt()

        c = torch.linalg.solve_triangular(lb, a @ y, upper=False) / sigma
        log_det = n * torch.log(sigma2) + 2.0 * torch.log(torch.diagonal(lb)).sum()
        quad = (y * y).sum() / sigma2 - (c * c).sum()
        # tr(Knn - Qnn) / sigma2; Knn diagonal is the signal variance (stationary)
        trace = (n * self.log_signal_var.exp() - sigma2 * (a * a).sum()) / sigma2

        return -0.5 * (n * t * math.log(2.0 * math.pi) + t * log_det + quad + t * trace)

    # --- training -----------------------------------------------------------------

    def fit(
        self,
        x: Tensor,
        y: Tensor,
        *,
        steps: int = 500,
        lr: float = 0.05,
        verbose: bool = False,
        callback: Callable[[int, float], bool | None] | None = None,
    ) -> FitResult:
        """Optimise inducing locations and hyperparameters by maximising the ELBO.

        `callback(step, elbo_per_point)` runs after every optimiser step, when `predict`
        is already usable. Returning True from it stops training early.
        """
        def joint_callback(step: int, elbos: list[float]) -> bool | None:
            return callback(step, elbos[0]) if callback is not None else None

        return fit_jointly(
            [self], [(x, y)], steps=steps, lr=lr, verbose=verbose, callback=joint_callback
        )[0]

    def _prepare(self, x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
        """Convert the data, remove the linear model and centre it; cache it and return
        (x, residual targets) for the GP."""
        x = x.to(torch.float64)
        y = y.to(torch.float64)
        if y.ndim == 1:
            y = y.unsqueeze(-1)
        if self.linear_degree is not None:
            features = self._features(x)
            coef = torch.linalg.lstsq(features, y).solution
            self._linear_coef = coef
            y = y - features @ coef  # the GP models what the line cannot
        y_mean = y.mean()
        yc = y - y_mean
        # Cached up front so predict() works from inside a training callback
        self._x_train, self._y_train, self._y_mean = x, yc, y_mean
        return x, yc

    # --- prediction ---------------------------------------------------------------

    @torch.no_grad()
    def predict(self, x_star: Tensor, *, include_noise: bool = False) -> tuple[Tensor, Tensor]:
        """Posterior mean (M, T), linear model included, and GP variance (M,) after `fit`."""
        if self._x_train is None or self._y_train is None or self._y_mean is None:
            raise RuntimeError("call fit() before predict()")
        x_star = x_star.to(torch.float64)
        luu, a, lb = self._factorise(self._x_train)
        sigma = self.noise_var.sqrt()
        c = torch.linalg.solve_triangular(lb, a @ self._y_train, upper=False) / sigma

        kus = self.kernel(self.inducing_points, x_star)
        tmp1 = torch.linalg.solve_triangular(luu, kus, upper=False)
        tmp2 = torch.linalg.solve_triangular(lb, tmp1, upper=False)

        mean = tmp2.T @ c + self._y_mean + self.linear_part(x_star)
        var =self.log_signal_var.exp() + tmp2.pow(2).sum(0) - tmp1.pow(2).sum(0)
        if include_noise:
            var = var + self.noise_var
        return mean, var


def fit_jointly(
    models: Sequence[SparsePressureGP],
    data: Sequence[tuple[Tensor, Tensor]],
    *,
    steps: int = 500,
    lr: float = 0.05,
    verbose: bool = False,
    callback: Callable[[int, list[float]], bool | None] | None = None,
) -> list[FitResult]:
    """Train several independent GPs in lockstep with a single optimiser.

    The models share nothing, so the summed loss is the same as training them one after
    another, but they advance together, which lets a UI show them updating side by side.
    `callback(step, [elbo_per_point, ...])` runs after every step; returning True stops.
    """
    prepared = [model._prepare(x, y) for model, (x, y) in zip(models, data, strict=True)]
    optimiser = torch.optim.Adam([p for model in models for p in model.parameters()], lr=lr)
    histories: list[list[float]] = [[] for _ in models]

    for step in range(steps):
        optimiser.zero_grad()
        losses = [
            -model.elbo(x, yc) / yc.numel() for model, (x, yc) in zip(models, prepared, strict=True)
        ]
        torch.stack(losses).sum().backward()
        optimiser.step()
        for history, loss in zip(histories, losses, strict=True):
            history.append(-loss.item())
        if verbose and step % 50 == 0:
            print(f"step {step:4d}  elbo/pt {[round(h[-1], 4) for h in histories]}")
        if callback is not None and callback(step, [h[-1] for h in histories]):
            break

    return [FitResult(elbo=history) for history in histories]

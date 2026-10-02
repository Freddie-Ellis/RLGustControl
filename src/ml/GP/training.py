from dataclasses import dataclass

import numpy as np
import torch
from airfoil import SurfaceData
from numpy.typing import NDArray
from pressure_gp import SparsePressureGP, fit_jointly
from qtpy.QtCore import QThread, Signal

Array = NDArray[np.float64]


@dataclass(frozen=True)
class SurfaceProgress:
    """Training state of the GP for one surface."""

    elbo: float  # per data point
    tap_s: Array  # inducing locations (arc length from the leading edge)
    tap_cp: Array  # model mean at the inducing locations
    grid_linear: Array  # the linear model alone, on the plotting grid
    grid_mean: Array  # linear model + GP mean on the plotting grid
    grid_std: Array  # GP latent standard deviation on the plotting grid


@dataclass(frozen=True)
class Progress:
    """Snapshot of a training run, sent to the UI thread."""

    run_id: int
    step: int
    steps: int
    upper: SurfaceProgress
    lower: SurfaceProgress


class TrainWorker(QThread):
    """Trains one LMGP (linear model + sparse GP on the residual) per surface in the background,
    streaming snapshots to the UI."""

    progress = Signal(object)  # Progress
    training_done = Signal(int)  # run_id

    def __init__(
        self,
        run_id: int,
        surface: SurfaceData,
        grid_upper: Array,
        grid_lower: Array,
        num_inducing: int,
        steps: int,
        *,
        lr: float = 0.05,
        emit_every: int = 4,
        frame_delay_ms: int = 15,
        linear_degree: int | None = 1,
    ) -> None:
        super().__init__()
        self._linear_degree = linear_degree
        self._run_id = run_id
        self._surface = surface
        self._grid_upper = grid_upper
        self._grid_lower = grid_lower
        self._num_inducing = num_inducing
        self._steps = steps
        self._lr = lr
        self._emit_every = emit_every
        self._frame_delay_ms = frame_delay_ms
        self._stop_requested = False

    def stop(self) -> None:
        """Ask the training loop to finish after its current step."""
        self._stop_requested = True

    def run(self) -> None:
        sides = (self._surface.upper, self._surface.lower)
        xs = [torch.from_numpy(side.s).reshape(-1, 1) for side in sides]
        ys = [torch.from_numpy(side.cp) for side in sides]
        grids = [
            torch.from_numpy(grid).reshape(-1, 1) for grid in (self._grid_upper, self._grid_lower)
        ]
        # LMGP: a linear model is fitted first and the GP learns the residual
        gps = [
            SparsePressureGP(x, self._num_inducing, linear_degree=self._linear_degree) for x in xs
        ]

        def side_progress(gp: SparsePressureGP, grid: torch.Tensor, elbo: float) -> SurfaceProgress:
            taps = gp.tap_locations()
            mean, var = gp.predict(torch.cat([taps, grid]))
            k = taps.shape[0]
            return SurfaceProgress(
                elbo=elbo,
                tap_s=taps[:, 0].numpy(),
                tap_cp=mean[:k, 0].numpy(),
                grid_linear=gp.linear_part(grid)[:, 0].numpy(),
                grid_mean=mean[k:, 0].numpy(),
                grid_std=var[k:].sqrt().numpy(),
            )

        def snapshot(step: int, elbos: list[float]) -> Progress:
            upper, lower = (
                side_progress(gp, grid, elbo) for gp, grid, elbo in zip(gps, grids, elbos, strict=True)
            )
            return Progress(self._run_id, step, self._steps, upper, lower)

        last_emitted = -1

        def on_step(step: int, elbos: list[float]) -> bool:
            nonlocal last_emitted
            if self._stop_requested:
                return True
            if step % self._emit_every == 0 or step == self._steps - 1:
                self.progress.emit(snapshot(step, elbos))
                last_emitted = step
                self.msleep(self._frame_delay_ms)  # pace it so the training is watchable
            return False

        results = fit_jointly(gps, list(zip(xs, ys, strict=True)), steps=self._steps, lr=self._lr, callback=on_step)

        # Make sure the final state (also after an early stop) reaches the UI
        final_step = len(results[0].elbo) - 1
        if final_step != last_emitted and final_step >= 0:
            self.progress.emit(snapshot(final_step, [r.elbo[-1] for r in results]))
        self.training_done.emit(self._run_id)

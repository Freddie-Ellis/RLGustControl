"""Small pyqtgraph helpers shared by the tabs."""

import numpy as np
import pyqtgraph as pg
from qtpy.QtCore import Qt
from qtpy.QtGui import QPen
from torch import Tensor

pg.setConfigOptions(antialias=True)


def series_pen(i: int, width: float = 1.5, dashed: bool = False) -> QPen:
    """A distinct colour for the i-th series (e.g. the i-th run on an overlay plot)."""
    style = Qt.PenStyle.DashLine if dashed else Qt.PenStyle.SolidLine
    return pg.mkPen(pg.intColor(i, hues=9), width=width, style=style)


def plot_batch(plot: pg.PlotItem, t: Tensor, y: Tensor, pen: QPen, name: str | None = None) -> None:
    """Plot every column of y [T, N] against t [T] as ONE curve item (NaN-separated), which stays fast for many envs."""
    y_np = y.detach().cpu().numpy()
    if y_np.ndim == 1:
        y_np = y_np[:, None]
    t_np = np.append(t.detach().cpu().numpy(), np.nan)
    xs = np.tile(t_np, y_np.shape[1])
    ys = np.vstack([y_np, np.full((1, y_np.shape[1]), np.nan)]).T.ravel()
    plot.plot(xs, ys, pen=pen, name=name, connect="finite")

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]

# Thin airfoil theory has a 1/sqrt(x) leading-edge singularity; cap the loading there
DELTA_CP_LIMIT = 4.0


@dataclass(frozen=True)
class Airfoil:
    """Surface coordinates (chord-normalised) of a 4-digit NACA airfoil."""

    x: Array  # chordwise stations used to build the section
    thickness: Array  # half-thickness distribution yt(x)
    camber_line: Array  # mean camber line yc(x)
    camber_slope: Array  # dyc/dx
    x_upper: Array
    y_upper: Array
    x_lower: Array
    y_lower: Array


def naca4(camber: float, camber_loc: float, thickness: float, n: int = 100) -> Airfoil:
    """Build a NACA 4-digit section.

    Args:
        camber: maximum camber m as a fraction of chord (NACA 2412 -> 0.02).
        camber_loc: chordwise position p of maximum camber as a fraction of chord
            (NACA 2412 -> 0.4). Ignored when `camber` is zero.
        thickness: maximum thickness t as a fraction of chord (NACA 2412 -> 0.12).
        n: number of chordwise stations (cosine spaced, clustering at the edges).
    """
    beta = np.linspace(0.0, np.pi, n)
    x = 0.5 * (1.0 - np.cos(beta))

    yt = 5.0 * thickness * (
        0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4
    )

    if camber == 0.0:
        yc = np.zeros_like(x)
        dyc_dx = np.zeros_like(x)
    else:
        p = camber_loc
        front = x < p
        yc = np.where(
            front,
            camber / p**2 * (2.0 * p * x - x**2),
            camber / (1.0 - p) ** 2 * ((1.0 - 2.0 * p) + 2.0 * p * x - x**2),
        )
        dyc_dx = np.where(
            front,
            2.0 * camber / p**2 * (p - x),
            2.0 * camber / (1.0 - p) ** 2 * (p - x),
        )

    theta = np.arctan(dyc_dx)
    return Airfoil(
        x=x,
        thickness=yt,
        camber_line=yc,
        camber_slope=dyc_dx,
        x_upper=x - yt * np.sin(theta),
        y_upper=yc + yt * np.cos(theta),
        x_lower=x + yt * np.sin(theta),
        y_lower=yc - yt * np.cos(theta),
    )


def placeholder_cp(airfoil: Airfoil, alpha: float = 0.0, n_modes: int = 8) -> tuple[Array, Array]:
    """Approximate upper/lower surface Cp at the airfoil's chordwise stations.

    This is a stand-in for CFD data, built from two linear pieces:
    - thickness: symmetric suction, Cp_t ~ -8 yt (about -0.5 at 12 % thickness)
    - camber/incidence: thin airfoil theory loading, dCp = Cp_lower - Cp_upper = 2 gamma / U,
      with gamma(beta) = 2U [A0 cot(beta/2) + sum A_n sin(n beta)], x = (1 - cos(beta)) / 2

    Camber therefore carries lift even at alpha = 0. The leading-edge singularity is capped
    at |dCp| = DELTA_CP_LIMIT so the result stays finite. Returns (cp_upper, cp_lower).
    """
    beta = np.arccos(1.0 - 2.0 * airfoil.x)
    slope = airfoil.camber_slope

    a0 = alpha - np.trapezoid(slope, beta) / np.pi
    gamma_over_u = a0 / np.tan(np.maximum(beta, 1e-3) / 2.0)
    for n in range(1, n_modes + 1):
        a_n = 2.0 / np.pi * np.trapezoid(slope * np.cos(n * beta), beta)
        gamma_over_u = gamma_over_u + a_n * np.sin(n * beta)
    delta_cp = np.clip(4.0 * gamma_over_u, -DELTA_CP_LIMIT, DELTA_CP_LIMIT)  # 2 * gamma/U

    cp_thickness = -8.0 * airfoil.thickness
    return cp_thickness - 0.5 * delta_cp, cp_thickness + 0.5 * delta_cp


@dataclass(frozen=True)
class Surface:
    """Pressure samples along one surface, from the leading edge to the trailing edge."""

    s: Array  # arc length from the leading edge
    x: Array
    y: Array
    cp: Array


@dataclass(frozen=True)
class SurfaceData:
    """Upper and lower surfaces as separate datasets.

    The pressure jump between the two surfaces at the leading edge is not smooth, so a
    single GP over both is hard to fit; each surface gets its own GP instead.
    """

    upper: Surface
    lower: Surface


def surface_data(airfoil: Airfoil, cp_upper: Array, cp_lower: Array) -> SurfaceData:
    """Split the airfoil into upper and lower surface datasets, each indexed by arc length."""

    def arc_length(x: Array, y: Array) -> Array:
        return np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])

    return SurfaceData(
        upper=Surface(
            arc_length(airfoil.x_upper, airfoil.y_upper), airfoil.x_upper, airfoil.y_upper, cp_upper
        ),
        lower=Surface(
            arc_length(airfoil.x_lower, airfoil.y_lower), airfoil.x_lower, airfoil.y_lower, cp_lower
        ),
    )

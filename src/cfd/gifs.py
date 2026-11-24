"""Presentation GIFs from the thinned CFD cases (paraview_scripts/thin_cases.py), drawn off-screen exactly as the CFD
tab draws them: same colour maps, ranges, lighting and camera direction.

Two kinds, run from the repo root:

  walls   wall Cp through a set's cases, by default seen from a fixed wind-tunnel camera: each case is turned back
          so the freestream runs along +x, so the wing visibly pitches with alpha and yaws with beta.
            uv run python -m src.cfd.gifs walls --batch batch3_000
            uv run python -m src.cfd.gifs walls --batch batch3_000 --beta all --slices 551   (|U| in the wake too)

  sweep   one actuator moving through its deflections at one flow condition (a set with a setpoints.parquet,
          e.g. actuator_effects), wall Cp plus a field on chosen slices.
            uv run python -m src.cfd.gifs sweep --actuator "le (sym)" --alpha 9 --field U --axis x --slices 136
            uv run python -m src.cfd.gifs sweep --actuator "wingtip (right)" --alpha 15 --beta 10 --field Uz --axis y --slices 237
          Run with --actuator list to see every actuator and its conditions.

  -h on either for every option. GIFs go to GIF_DIR, named after what they show, unless --out is given.
"""

import argparse
import re
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pyvista as pv
from PIL import Image

from src.cfd import slice_data
from src.cfd.slice_data import CFD_VIEW_DIR, CfdCase

GIF_DIR: Path = Path(__file__).resolve().parents[2] / "gifs"
WINDOW: tuple[int, int] = (1600, 900)  # pixels; a 16:9 slide
WALLS_FPS: float = 6.0
SWEEP_FPS: float = 2.0  # few frames per sweep (3 - 5 deflections), so slow
SWEEP_SET: str = "actuator_effects"
WALLS_ZOOM: float = 1.0  # framed on the first (lowest alpha) pose, with room for the others to pitch and yaw
SWEEP_ZOOM: float = 1.6  # framed on the wing (slices span the whole domain, so they are cut off at the edges)
TEXT_SIZE: int = 14
VIEWS: dict[str, tuple[float, float, float]] = {  # camera direction from the wing, model frame (x aft, y stbd, z up)
    "iso": slice_data.VIEW_DIRECTION,  # as the CFD tab
    "tunnel": (0.6, -1.5, 1.2),  # port, above and a little aft: pitching up turns the upper skin towards the camera
    "front": (-1.0, 0.0, 0.25),  # from upstream, a little above: straight onto x slices, the LEVs in cross-section
    "side": (0.0, -1.0, 0.1),  # from port: straight onto y slices
    "top": (0.0, 0.0, 1.0),
}
SWEEP_VIEW: str = "front"
WALLS_VIEW: str = "iso"
WAKE_SLICE_MM: int = 551  # first x slice behind the trailing edge (x = 499 mm)


def tunnel_frame(mesh: pv.PolyData, alpha: float, beta: float, pivot: Sequence[float]) -> pv.PolyData:
    """`mesh` turned about `pivot` so the freestream runs along +x. The CFD freestream in the model frame is
    (cos a cos b, -sin b, sin a cos b); a right-hand turn by alpha about y, then by beta about z, takes it to x."""
    return mesh.rotate_y(alpha, point=pivot, inplace=False).rotate_z(beta, point=pivot, inplace=False)


def frame(plotter: pv.Plotter, case: CfdCase, walls: pv.PolyData, slices: list[pv.PolyData], field: str,
          label: str, camera: list[tuple[float, float, float]]) -> Image.Image:
    """One picture: `walls` in wall Cp and `slices` in `field`, with `label` top left."""
    plotter.clear()
    same_bar = field == slice_data.WALL_FIELD or not slices
    wall_title = "Cp" if same_bar else "Cp (walls)"
    slice_data.add_field(plotter, case, walls, slice_data.WALL_FIELD, slice_data.WALL_LIGHTING,
                         slice_data.bar_args(slice_data.WALL_FIELD, wall_title, 0.6))
    for i, mesh in enumerate(slices):
        bar = slice_data.bar_args(field, slice_data.FIELDS[field].title, 0.1) if i == 0 and not same_bar else None
        slice_data.add_field(plotter, case, mesh, field, slice_data.SLICE_LIGHTING, bar)
    plotter.add_text(label, position="upper_left", color="black", font_size=TEXT_SIZE)
    plotter.camera_position = camera
    return Image.fromarray(plotter.screenshot(return_img=True))


def save_gif(images: list[Image.Image], path: Path, fps: float, bounce: bool) -> None:
    """Write `images` as a looping GIF; `bounce` plays them forward then back."""
    if bounce and len(images) > 2:
        images = images + images[-2:0:-1]
    paletted = [im.convert("RGB").quantize(colors=256, method=Image.Quantize.MEDIANCUT) for im in images]
    path.parent.mkdir(parents=True, exist_ok=True)
    paletted[0].save(path, save_all=True, append_images=paletted[1:], duration=round(1000 / fps), loop=0,
                     optimize=False, disposal=2)
    print(f"saved {path} ({len(paletted)} frames)")


def plotter() -> pv.Plotter:
    p = pv.Plotter(off_screen=True, window_size=list(WINDOW))
    p.set_background("white")
    return p


def walls_gif(set_name: str, batch: str, beta: str, tunnel: bool, field: str, axis: str, positions_mm: list[str],
              view: str, zoom: float, fps: float, bounce: bool, out: Path | None) -> None:
    set_dir = CFD_VIEW_DIR / set_name
    names = []
    for name in slice_data.list_cases(set_dir):
        condition = slice_data.case_condition(name)
        if condition is None or not re.fullmatch(batch, condition[0]):
            continue
        if beta != "all" and condition[2] != float(beta):
            continue
        names.append(name)
    if not names:
        raise SystemExit(f"no cases in {set_dir} with batch matching {batch!r} and beta {beta}")
    print(f"{len(names)} cases")

    p = plotter()
    images: list[Image.Image] = []
    camera: list[tuple[float, float, float]] | None = None
    pivot: tuple[float, float, float] | None = None
    for name in names:
        batch_name, alpha, b = slice_data.case_condition(name) or ("", 0.0, 0.0)
        case = slice_data.load_case(set_dir / name, with_slices=bool(positions_mm))
        if pivot is None:  # one pivot and camera for every frame, from the first case as it sits in the model frame
            pivot = tuple(np.array(case.walls.center, dtype=float))
            camera = slice_data.iso_camera([case.walls], zoom=zoom, direction=VIEWS[view])
        meshes = [case.walls] + [s.mesh for s in nearest_slices(case, axis, positions_mm)]
        if tunnel:  # the slices turn with the wing, so they stay where they are relative to it
            meshes = [tunnel_frame(m, alpha, b, pivot) for m in meshes]
        label = f"{batch_name}    AoA {alpha:+.1f} deg    sideslip {b:+.1f} deg"
        images.append(frame(p, case, meshes[0], meshes[1:], field, label, camera or []))
        print(f"  {name}")
    p.close()
    beta_tag = "all" if beta == "all" else f"b{float(beta):+.0f}"
    s_tag = f"_{field}_{slices_tag(axis, positions_mm)}" if positions_mm else ""
    save_gif(images, out or GIF_DIR / f"walls_{batch.replace('.*', 'all')}_{beta_tag}{s_tag}.gif", fps, bounce)


def slice_spec(spec: str, axis: str) -> tuple[str, int]:
    """(axis, mm) of a --slices entry: "551" is at 551 mm along `axis`; "y119" or "y-118" names its own axis."""
    if spec[:1] in slice_data.AXES:
        return spec[0], int(spec[1:])
    return axis, int(spec)


def nearest_slices(case: CfdCase, axis: str, specs: list[str]) -> list[slice_data.Slice]:
    """The case's slices nearest each of `specs` (see slice_spec)."""
    chosen = []
    for spec in specs:
        spec_axis, mm = slice_spec(spec, axis)
        available = case.slices[spec_axis]
        if available:
            chosen.append(min(available, key=lambda s: abs(s.position_mm - mm)))
    return chosen


def slices_tag(axis: str, specs: list[str]) -> str:
    return "_".join(f"{a}{mm}" for a, mm in (slice_spec(spec, axis) for spec in specs))


def list_actuators(sweeps: slice_data.Sweeps) -> str:
    lines = []
    for actuator, conditions in sweeps.items():
        conds = ", ".join(f"({a:+g}, {b:+g})" for a, b in conditions)
        lines.append(f"  {actuator!r}: (alpha, beta) = {conds}")
    return "\n".join(lines)


def matches(value: float, wanted: str) -> bool:
    """`value` is `wanted` (a number), or `wanted` is "all"."""
    return wanted == "all" or value == float(wanted)


def sweep_gif(set_name: str, actuator: str, alpha: str, beta: str, field: str, axis: str,
              positions_mm: list[str], view: str, zoom: float, fps: float, bounce: bool, out: Path | None) -> None:
    """One actuator's sweep at one condition, or (actuator, alpha or beta "all") every matching sweep one after
    another: actuator by actuator, each through its conditions. `bounce` plays each sweep forward then back."""
    set_dir = CFD_VIEW_DIR / set_name
    sweeps = slice_data.actuator_sweeps(set_dir)
    if not sweeps:
        raise SystemExit(f"{set_dir} has no {slice_data.SETPOINTS}, so no actuator sweeps")
    if actuator != "all" and actuator not in sweeps:
        raise SystemExit(f"actuators in {set_name}:\n{list_actuators(sweeps)}")
    chosen_sweeps = [(name, condition, points)
                     for name, conditions in sweeps.items() if actuator in ("all", name)
                     for condition, points in conditions.items()
                     if matches(condition[0], alpha) and matches(condition[1], beta)]
    if not chosen_sweeps:
        raise SystemExit(f"no sweep at alpha {alpha}, beta {beta}; there are:\n"
                         f"{list_actuators({a: c for a, c in sweeps.items() if actuator in ('all', a)})}")
    print(f"{len(chosen_sweeps)} sweeps, {sum(len(pts) for _, _, pts in chosen_sweeps)} cases")

    p = plotter()
    images: list[Image.Image] = []
    camera: list[tuple[float, float, float]] | None = None
    for name, (a, b), points in chosen_sweeps:
        print(f"{name} at alpha {a:+g}, beta {b:+g}: {[pt.deflection_deg for pt in points]} deg")
        sweep_images = []
        for point in points:
            case = slice_data.load_case(set_dir / point.case)
            chosen = nearest_slices(case, axis, positions_mm)
            if camera is None:  # one camera for every frame
                print(f"slices: {[f'{s.axis} = {s.position_mm} mm' for s in chosen]}")
                camera = slice_data.iso_camera([case.walls], zoom=zoom, direction=VIEWS[view])
            label = f"{name}  {point.deflection_deg:+g} deg\nAoA {a:+.1f} deg    sideslip {b:+.1f} deg"
            sweep_images.append(frame(p, case, case.walls, [s.mesh for s in chosen], field, label, camera))
        if bounce and len(sweep_images) > 2:
            sweep_images += sweep_images[-2:0:-1]
        images += sweep_images
    p.close()
    tag = re.sub(r"[^\w+-]+", "_", actuator).strip("_")
    a_tag = "all" if alpha == "all" else f"{float(alpha):+g}"
    b_tag = "all" if beta == "all" else f"{float(beta):+g}"
    s_tag = slices_tag(axis, positions_mm)
    save_gif(images, out or GIF_DIR / f"sweep_{tag}_a{a_tag}_b{b_tag}_{field}_{s_tag}.gif", fps, bounce=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    kinds = parser.add_subparsers(dest="kind", required=True)

    w = kinds.add_parser("walls", help="wall Cp through a set's cases")
    w.add_argument("--set", default="", help='set (subfolder of the CFD view folder); "" for the main cases')
    w.add_argument("--batch", default=".*", help="regex the batch name must match, e.g. batch3_000 (default: all)")
    w.add_argument("--beta", default="0", help='only cases at this sideslip, or "all" (default 0: the alpha sweep)')
    w.add_argument("--model-frame", action="store_true", help="keep the wing still instead of turning it to the wind")
    w.add_argument("--slices", nargs="*", default=[],
                   help=f"slice positions in mm along --axis, e.g. {WAKE_SLICE_MM} (just behind the trailing edge); "
                        "prefix one with its axis to mix, e.g. 551 y119")
    w.add_argument("--field", default="U", choices=list(slice_data.FIELDS), help="field on the slices")
    w.add_argument("--axis", default="x", choices=list(slice_data.AXES))
    w.add_argument("--view", default=WALLS_VIEW, choices=list(VIEWS))
    w.add_argument("--zoom", type=float, default=WALLS_ZOOM, help="> 1 moves in")
    w.add_argument("--fps", type=float, default=WALLS_FPS)
    w.add_argument("--bounce", action="store_true", help="play forward then back")
    w.add_argument("--out", type=Path, help="output .gif (default: under GIF_DIR)")

    s = kinds.add_parser("sweep", help="one actuator through its deflections")
    s.add_argument("--set", default=SWEEP_SET)
    s.add_argument("--actuator", required=True,
                   help='e.g. "le (sym)"; "all" for every actuator in turn; "list" to see them all')
    s.add_argument("--alpha", default="9", help='a condition\'s alpha, or "all" for every alpha in turn')
    s.add_argument("--beta", default="0", help='a condition\'s beta, or "all"')
    s.add_argument("--field", default="U", choices=list(slice_data.FIELDS), help="field on the slices")
    s.add_argument("--axis", default="x", choices=list(slice_data.AXES))
    s.add_argument("--slices", nargs="*", default=["136"],
                   help="slice positions in mm along --axis (nearest slice used), or with their own axis, e.g. "
                        "551 y119; none for walls only")
    s.add_argument("--view", default=SWEEP_VIEW, choices=list(VIEWS),
                   help="camera: front looks along x (x slices), side along y (y slices)")
    s.add_argument("--zoom", type=float, default=SWEEP_ZOOM, help="> 1 moves in")
    s.add_argument("--fps", type=float, default=SWEEP_FPS)
    s.add_argument("--no-bounce", action="store_true", help="play each sweep forward only (default: forward then back)")
    s.add_argument("--out", type=Path)

    args = parser.parse_args()
    if args.kind == "walls":
        walls_gif(args.set, args.batch, args.beta, not args.model_frame, args.field, args.axis, args.slices, args.view,
                  args.zoom, args.fps, args.bounce, args.out)
    else:
        sweep_gif(args.set, args.actuator, args.alpha, args.beta, args.field, args.axis, args.slices, args.view,
                  args.zoom, args.fps, not args.no_bounce, args.out)


if __name__ == "__main__":
    main()

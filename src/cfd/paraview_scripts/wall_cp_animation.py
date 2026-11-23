r"""Render wall Cp as animation frames, seen from a fixed wind-tunnel camera.

The CFD solves every case in the model frame (x aft, y starboard, z up) and turns the freestream instead of the
wing. Here each case is rotated back so the freestream always runs along +x: the wing visibly pitches up with
alpha and yaws with beta while the camera stays still. Control surfaces are already deflected in each case's
geometry, so actuator cases show the surfaces moving.

Run with ParaView's own Python (NOT normal python):
  Windows:  & "C:\Program Files\ParaView 6.2.1\bin\pvpython.exe" wall_cp_animation.py C:\path\to\runs C:\path\to\frames
  Iridis:   <osmesa ParaView>/bin/pvbatch wall_cp_animation.py /path/to/runs /path/to/frames

<runs> is the folder holding the batch*/ case folders. Frames are written as frame_0000.png, ... in order:
undeflected cases first, then each set of control-surface deflections; within each, the alpha sweep at zero
sideslip, then the sideslip conditions (see animation_order). Conditions with no walls.vtp yet are skipped. Stitch the frames into a video afterwards (see the bottom of this docstring).

Stitching at 0.1 s per frame (150 frames -> 15 s), either:
  ffmpeg -framerate 10 -i frame_%04d.png -c:v libx264 -pix_fmt yuv420p -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" wall_cp.mp4
  uv run python -c "from PIL import Image; import glob; f=[Image.open(p) for p in sorted(glob.glob('frame_*.png'))]; f[0].save('wall_cp.gif', save_all=True, append_images=f[1:], duration=100, loop=0)"
"""
import glob
import json
import math
import os
import re
import sys

from paraview.simple import *

# ---- settings ---------------------------------------------------------------
CP_RANGE = (-4.0, 1.0)       # keep in step with slices.py and src/cfd/slice_data.py: one Cp colour scale everywhere
SIZE = [1600, 900]
VIEW_DIRECTION = (-1.4, -1.5, 1.0)   # camera sits along this direction from the moment reference point
ZOOM = 0.8                   # < 1 backs the camera off so every pitched / yawed pose stays in shot
# ------------------------------------------------------------------------------


def condition_name(alpha: float, beta: float) -> str:
    """Solve folder name for a flow condition, as FlowCondition.name in src/cfd/case.py."""
    return "a%+05.1f_b%+05.1f" % (alpha, beta)


def toml_list(section: str, key: str) -> list:
    """A numeric list `key = [...]` from one section of case.toml (written by batch_cases.py, so plain JSON-like)."""
    m = re.search(r"^\s*%s\s*=\s*(\[.*?\])\s*(#.*)?$" % key, section, re.MULTILINE)
    return json.loads(m.group(1)) if m else []


def read_case(case_dir: str) -> tuple[list[tuple[float, float]], dict[str, float]]:
    """The case's flow conditions in case.toml order (as CaseSpec.conditions()) and its non-zero deflections."""
    text = open(os.path.join(case_dir, "case.toml")).read()
    sections = dict(re.findall(r"^\[(\w+)\]\s*$(.*?)(?=^\[|\Z)", text, re.MULTILINE | re.DOTALL))

    flow = sections.get("flow", "")
    alphas, betas = toml_list(flow, "alpha_deg"), toml_list(flow, "beta_deg") or [0.0]
    pairs = [(a, b) for b in betas for a in alphas] + [tuple(p) for p in toml_list(flow, "pairs")]
    conditions = list(dict.fromkeys((float(a), float(b)) for a, b in pairs))

    deflections = {k: float(v) for k, v in
                   re.findall(r"^\s*(\w+)\s*=\s*([-+0-9.eE]+)", sections.get("deflections_deg", ""), re.MULTILINE)}
    return conditions, {k: v for k, v in deflections.items() if v != 0.0}


def latest_walls(solve_dir: str) -> str | None:
    """walls.vtp from the last time folder of a solved condition, or None if it has not been written."""
    files = glob.glob(os.path.join(solve_dir, "postProcessing", "wallSurfaces", "*", "walls.vtp"))
    return max(files, key=lambda f: float(os.path.basename(os.path.dirname(f)))) if files else None


def frames(root: str) -> list[dict]:
    """Every frame to render, in animation order."""
    out = []
    for case_dir in sorted(d for d in glob.glob(os.path.join(root, "batch*")) if os.path.isdir(d)):
        conditions, deflections = read_case(case_dir)
        for alpha, beta in conditions:
            solve_dir = os.path.join(case_dir, "solve", condition_name(alpha, beta))
            walls = latest_walls(solve_dir)
            if walls is None:
                print("skip %s/%s (no walls.vtp)" % (os.path.basename(case_dir), condition_name(alpha, beta)))
                continue
            info = json.load(open(os.path.join(solve_dir, "solve_case.json")))
            speed = math.sqrt(sum(u * u for u in info["freestream_velocity_m_s"]))
            out.append({"case": os.path.basename(case_dir), "alpha": alpha, "beta": beta, "walls": walls,
                        "deflections": deflections, "q": 0.5 * speed ** 2,
                        "ref": info["reference"]["point_m"]})
    return sorted(out, key=animation_order)


def animation_order(frame: dict) -> tuple:
    """Undeflected cases first, then each set of deflections in turn. Within a set: the whole alpha sweep at zero
    sideslip, then the sideslip conditions (by alpha, then beta). Independent of how cases were split into batches."""
    deflections = tuple(sorted(frame["deflections"].items()))
    return (len(deflections) > 0, deflections, frame["beta"] != 0.0, frame["alpha"], frame["beta"])


def label(frame: dict) -> str:
    text = "%s    AoA %+.1f deg    sideslip %+.1f deg" % (frame["case"], frame["alpha"], frame["beta"])
    if frame["deflections"]:
        text += "\n" + "    ".join("%s %+g deg" % (k, v) for k, v in sorted(frame["deflections"].items()))
    return text


def to_tunnel_frame(source, alpha: float, beta: float, ref: list[float]) -> list:
    """Rotate the model about the moment reference point so the freestream lies along +x.

    The freestream in the model frame is (cos a cos b, -sin b, sin a cos b) (src/cfd/solve.py). A right-hand
    rotation by alpha about y, then by beta about z, takes it to (1, 0, 0). Kept as separate one-step filters
    so the order of operations is explicit. Returns the filters, last one is the output."""
    steps = [{"Translate": [-c for c in ref]}, {"Rotate": [0.0, alpha, 0.0]},
             {"Rotate": [0.0, 0.0, beta]}, {"Translate": list(ref)}]
    filters = []
    for step in steps:
        t = Transform(Input=filters[-1] if filters else source)
        t.Transform = "Transform"
        for prop, value in step.items():
            setattr(t.Transform, prop, value)
        filters.append(t)
    return filters


root, outdir = sys.argv[1], sys.argv[2]
os.makedirs(outdir, exist_ok=True)
todo = frames(root)
print("rendering %d frames" % len(todo))

view = CreateRenderView()
view.ViewSize = SIZE
try:
    view.UseColorPaletteForBackground = 0   # ParaView >= 5.10 ignores Background unless this is off
except AttributeError:
    pass
view.Background = [1, 1, 1]
view.OrientationAxesVisibility = 1

lut = GetColorTransferFunction("Cp")
lut.AutomaticRescaleRangeMode = "Never"
lut.RescaleTransferFunction(*CP_RANGE)
lut.ColorSpace = "Lab"   # blue - white - red, white at Cp = 0 (keep in step with slices.py DIVERGING)
lut.RGBPoints = [CP_RANGE[0], 0.23, 0.30, 0.75, 0.0, 1.0, 1.0, 1.0, CP_RANGE[1], 0.71, 0.02, 0.15]
GetOpacityTransferFunction("Cp").RescaleTransferFunction(*CP_RANGE)
bar = GetScalarBar(lut, view)
bar.Title = "Cp"
bar.ComponentTitle = ""
bar.TitleColor = [0, 0, 0]   # default white text vanishes on the white background
bar.LabelColor = [0, 0, 0]

text = Text(Text="")
text_disp = Show(text, view)
text_disp.WindowLocation = "Upper Left Corner"
text_disp.FontSize = 14
text_disp.Color = [0, 0, 0]

camera_set = False
for i, frame in enumerate(todo):
    reader = XMLPolyDataReader(FileName=[frame["walls"]])
    calc = Calculator(Input=reader)
    calc.AttributeType = "Cell Data"
    calc.ResultArrayName = "Cp"
    calc.Function = "p/%g" % frame["q"]
    moved = to_tunnel_frame(calc, frame["alpha"], frame["beta"], frame["ref"])

    disp = Show(moved[-1], view)
    ColorBy(disp, ("CELLS", "Cp"))
    disp.LookupTable = lut
    lut.RescaleTransferFunction(*CP_RANGE)
    disp.SetScalarBarVisibility(view, True)
    text.Text = label(frame)

    if not camera_set:   # frame once, then never move the camera again
        ref = frame["ref"]
        view.CameraFocalPoint = list(ref)
        view.CameraPosition = [r + d for r, d in zip(ref, VIEW_DIRECTION)]
        view.CameraViewUp = [0, 0, 1]
        view.ResetCamera(False)
        view.CameraFocalPoint = list(ref)   # ResetCamera recentres on the bounds; pin back to the pivot
        GetActiveCamera().Dolly(ZOOM)
        camera_set = True

    Render(view)
    SaveScreenshot(os.path.join(outdir, "frame_%04d.png" % i), view, ImageResolution=SIZE)
    print("frame %04d  %s %s" % (i, frame["case"], condition_name(frame["alpha"], frame["beta"])))

    Hide(moved[-1], view)
    for f in reversed(moved):
        Delete(f)
    Delete(calc)
    Delete(reader)

"""Slices + wall-surface pressure (or yPlus / Cf) for CFD cases, with an optional slice sweep.

Run with ParaView's own Python, e.g. on Windows:
  "C:\\Program Files\\ParaView 5.13\\bin\\pvpython.exe" render_slices.py <path> <outdir> [options]

<path> can be:
  - one case folder   (.../solve/a+25.0_b+00.0)
  - a solve folder or a batch folder or the root (every case found is rendered)

Modes
  --mode all     one image per case: walls + all slices together (default)
  --mode sweep   walls + ONE slice at a time, stepping through the slices in order
                 -> sweep_x_000.png, sweep_x_001.png ... (stitch with ffmpeg, see below)

Examples
  render_slices.py C:\\dev\\test_cfd\\batch1_000\\solve\\a+25.0_b+00.0 out --mode all --axis x
  render_slices.py C:\\dev\\test_cfd\\batch1_000\\solve\\a+25.0_b+00.0 out --mode sweep --axis y --camera top
  render_slices.py C:\\dev\\test_cfd out --slice-field Cp --wall yPlus

Stitch a sweep into a video (ffmpeg):
  ffmpeg -framerate 4 -i sweep_x_%03d.png -pix_fmt yuv420p sweep_x.mp4

Data assumptions (checked on a+25.0_b+00.0):
  postProcessing/wallSurfaces/<time>/walls.vtp   cell data: p, yPlus, wallShearStress
  postProcessing/flowSlices/<time>/x_136mm.vtp, y_m118mm.vtp, y_p60mm.vtp ...  cell data: k, nut, p, U
  p and wallShearStress are kinematic (incompressible), so Cp = p / (0.5*U_inf^2).

UNTESTED against a real ParaView install (smoke-tested only with a stubbed paraview module).
Property names are for ParaView 5.12/5.13. Tweak the camera and colour ranges once to taste.
"""
import argparse
import glob
import math
import os
import re
import sys

from paraview.simple import *

# ---------------------------------------------------------------------------
# arguments
# ---------------------------------------------------------------------------
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("path", help="case folder, or a folder containing cases")
ap.add_argument("outdir", help="where PNGs are written (one subfolder per case)")
ap.add_argument("--mode", choices=["all", "sweep"], default="all")
ap.add_argument("--axis", choices=["x", "y", "all"], default="all", help="which slices to use")
ap.add_argument("--slice-field", default="U", help="U (magnitude), Cp, k, nut or p")
ap.add_argument("--slice-range", nargs=2, type=float, help="colour range for slices (default: sensible / data range)")
ap.add_argument("--wall", choices=["Cp", "yPlus", "Cf"], default="Cp", help="wall surface colouring")
ap.add_argument("--wall-range", nargs=2, type=float, help="colour range for walls")
ap.add_argument("--camera", choices=["iso", "top", "side", "front", "back"], default="iso")
ap.add_argument("--zoom", type=float, default=1.0, help=">1 zooms in")
ap.add_argument("--uinf", type=float, default=20.0, help="freestream speed in m/s")
ap.add_argument("--size", nargs=2, type=int, default=[1600, 900], metavar=("W", "H"))
args = ap.parse_args()

Q = 0.5 * args.uinf ** 2   # kinematic dynamic pressure; assumes p_inf = 0

DEFAULT_WALL_RANGE = {"Cp": (-2.0, 1.0), "yPlus": (0.0, 100.0), "Cf": (0.0, 0.01)}
DEFAULT_SLICE_RANGE = {"U": (0.0, 1.5 * args.uinf), "Cp": (-2.0, 1.0)}

# direction from focal point to camera, and view-up
CAMERAS = {
    "iso":   ((-0.8, -1.0, 0.6), (0, 0, 1)),
    "top":   ((0, 0, 1), (1, 0, 0)),
    "side":  ((0, -1, 0), (0, 0, 1)),
    "front": ((-1, 0, 0), (0, 0, 1)),   # from upstream
    "back":  ((1, 0, 0), (0, 0, 1)),    # from downstream
}

SLICE_RE = re.compile(r"^([xyz])_(m|p)?(\d+)mm\.vtp$")


# ---------------------------------------------------------------------------
# file discovery
# ---------------------------------------------------------------------------
def latest_time_dir(parent):
    subs = [d for d in glob.glob(os.path.join(parent, "*")) if os.path.isdir(d)]
    def key(d):
        try:
            return float(os.path.basename(d))
        except ValueError:
            return -1.0
    return max(subs, key=key) if subs else None


def find_cases(path):
    path = os.path.abspath(path)
    cands = [path] + glob.glob(os.path.join(path, "*")) \
        + glob.glob(os.path.join(path, "solve", "*")) \
        + glob.glob(os.path.join(path, "*", "solve", "*"))
    cases = []
    for c in sorted(set(cands)):
        if glob.glob(os.path.join(c, "postProcessing", "wallSurfaces", "*", "walls.vtp")):
            cases.append(c)
    return cases


def list_slices(case, axis):
    tdir = latest_time_dir(os.path.join(case, "postProcessing", "flowSlices"))
    out = []
    if not tdir:
        return out
    for f in glob.glob(os.path.join(tdir, "*.vtp")):
        m = SLICE_RE.match(os.path.basename(f))
        if not m:
            continue
        ax, sign, mm = m.group(1), m.group(2), int(m.group(3))
        if axis != "all" and ax != axis:
            continue
        val = -mm if sign == "m" else mm
        out.append((ax, val, f))
    return sorted(out)        # by axis, then position


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def data_range(src, array):
    src.UpdatePipeline()
    arr = src.CellData[array]
    return arr.GetRange(-1 if arr.GetNumberOfComponents() > 1 else 0)


def fixed_lut(name, rng):
    lut = GetColorTransferFunction(name)
    lut.AutomaticRescaleRangeMode = "Never"
    lut.RescaleTransferFunction(rng[0], rng[1])
    return lut


def union_bounds(sources):
    b = None
    for s in sources:
        s.UpdatePipeline()
        sb = s.GetDataInformation().GetBounds()
        if b is None:
            b = list(sb)
        else:
            for i in (0, 2, 4):
                b[i] = min(b[i], sb[i])
            for i in (1, 3, 5):
                b[i] = max(b[i], sb[i])
    return b


def set_camera(view, bounds, name, zoom):
    c = [(bounds[0] + bounds[1]) / 2, (bounds[2] + bounds[3]) / 2, (bounds[4] + bounds[5]) / 2]
    diag = math.sqrt((bounds[1] - bounds[0]) ** 2 + (bounds[3] - bounds[2]) ** 2 + (bounds[5] - bounds[4]) ** 2)
    d, up = CAMERAS[name]
    n = math.sqrt(sum(x * x for x in d))
    dist = 1.5 * diag / zoom
    view.CameraFocalPoint = c
    view.CameraPosition = [c[i] + dist * d[i] / n for i in range(3)]
    view.CameraViewUp = list(up)


def try_set(obj, attr, value):
    """Property names shift a little between ParaView versions; don't die on cosmetics."""
    try:
        setattr(obj, attr, value)
    except Exception as e:
        print("  (skipped %s: %s)" % (attr, e))


# ---------------------------------------------------------------------------
# view
# ---------------------------------------------------------------------------
view = CreateRenderView()
view.ViewSize = args.size
try_set(view, "UseColorPaletteForBackground", 0)
try_set(view, "Background", [1, 1, 1])
view.OrientationAxesVisibility = 1

cases = find_cases(args.path)
if not cases:
    sys.exit("No cases with postProcessing/wallSurfaces/*/walls.vtp found under %s" % args.path)
print("found %d case(s)" % len(cases))

for case in cases:
    name = os.path.basename(case)
    batch = os.path.basename(os.path.dirname(os.path.dirname(case)))   # ...<batch>/solve/<case>
    tag = "%s_%s" % (batch, name) if batch.startswith("batch") else name
    outdir = os.path.join(args.outdir, tag)
    os.makedirs(outdir, exist_ok=True)
    created = []          # for cleanup, creation order

    # ---- walls ---------------------------------------------------------
    wdir = latest_time_dir(os.path.join(case, "postProcessing", "wallSurfaces"))
    wreader = XMLPolyDataReader(FileName=[os.path.join(wdir, "walls.vtp")])
    created.append(wreader)
    if args.wall == "yPlus":
        wsrc, warray = wreader, "yPlus"
    else:
        calc = Calculator(Input=wreader)
        calc.AttributeType = "Cell Data"
        if args.wall == "Cp":
            calc.ResultArrayName, calc.Function = "Cp", "p/%g" % Q
        else:
            calc.ResultArrayName, calc.Function = "Cf", "mag(wallShearStress)/%g" % Q
        created.append(calc)
        wsrc, warray = calc, args.wall
    wall_range = tuple(args.wall_range) if args.wall_range else DEFAULT_WALL_RANGE[args.wall]
    wlut = fixed_lut(warray, wall_range)
    wdisp = Show(wsrc, view)
    ColorBy(wdisp, ("CELLS", warray))
    wdisp.LookupTable = wlut
    wdisp.SetScalarBarVisibility(view, True)
    try_set(GetScalarBar(wlut, view), "WindowLocation", "Lower Left Corner")

    # ---- slices --------------------------------------------------------
    slices = list_slices(case, args.axis)
    if not slices:
        print("%s: no matching slices, rendering walls only" % tag)
    slice_objs = []       # (axis, mm, display, source)
    sf = args.slice_field
    for ax, val, f in slices:
        r = XMLPolyDataReader(FileName=[f])
        created.append(r)
        if sf == "Cp":
            c = Calculator(Input=r)
            c.AttributeType, c.ResultArrayName, c.Function = "Cell Data", "Cp", "p/%g" % Q
            created.append(c)
            src, arr, comp = c, "Cp", None
        elif sf == "U":
            src, arr, comp = r, "U", "Magnitude"
        else:
            src, arr, comp = r, sf, None
        slice_objs.append((ax, val, src, arr, comp))

    if slice_objs:
        if args.slice_range:
            srange = tuple(args.slice_range)
        elif sf in DEFAULT_SLICE_RANGE:
            srange = DEFAULT_SLICE_RANGE[sf]
        else:   # k, nut, p: use the data range across this case's slices
            los, his = zip(*[data_range(s[2], s[3]) for s in slice_objs])
            srange = (min(los), max(his))
        slut = fixed_lut(sf if sf != "U" else "U", srange)

    slice_disps = []
    for ax, val, src, arr, comp in slice_objs:
        d = Show(src, view)
        ColorBy(d, ("CELLS", arr, comp) if comp else ("CELLS", arr))
        d.LookupTable = slut
        d.Visibility = 0
        slice_disps.append(d)
    if slice_disps:
        slice_disps[0].SetScalarBarVisibility(view, True)
        bar = GetScalarBar(slut, view)
        try_set(bar, "Title", "|U| (m/s)" if sf == "U" else sf)
        try_set(bar, "ComponentTitle", "")
        try_set(bar, "WindowLocation", "Lower Right Corner")

    # ---- label ---------------------------------------------------------
    label = Text(Text=tag)
    created.append(label)
    ldisp = Show(label, view)
    try_set(ldisp, "WindowLocation", "Upper Left Corner")
    try_set(ldisp, "FontSize", 18)
    try_set(ldisp, "Color", [0, 0, 0])

    # ---- camera (fixed for the whole case) ------------------------------
    set_camera(view, union_bounds([wreader] + [s[2] for s in slice_objs]), args.camera, args.zoom)

    # ---- render --------------------------------------------------------
    if args.mode == "all":
        for d in slice_disps:
            d.Visibility = 1
        Render(view)
        path = os.path.join(outdir, "%s_slices_%s_wall%s.png" % (tag, sf, args.wall))
        SaveScreenshot(path, view, ImageResolution=args.size)
        print("saved", path)
    else:
        counters = {}
        for (ax, val, src, arr, comp), d in zip(slice_objs, slice_disps):
            i = counters.get(ax, 0)
            counters[ax] = i + 1
            for other in slice_disps:
                other.Visibility = 0
            d.Visibility = 1
            label.Text = "%s   %s = %d mm" % (tag, ax, val)
            Render(view)
            path = os.path.join(outdir, "sweep_%s_%03d.png" % (ax, i))
            SaveScreenshot(path, view, ImageResolution=args.size)
            print("saved", path)

    # ---- cleanup before the next case ----------------------------------
    for obj in reversed(created):
        Delete(obj)
    del created
r"""Flow slices + wall surface (Cp, yPlus or Cf) for CFD cases: one image per slice, stepping through the slices.

Every image shares one camera and one colour range per field, so cases can be compared image to image (as in
wall_cp_animation.py). Slices are drawn one at a time because, shown together, the nearest one hides
all the others.

Run with ParaView's own Python (NOT normal python):
  Windows:  & "C:\Program Files\ParaView 6.2.1\bin\pvpython.exe" slices.py <path> <outdir> [options]
  Iridis:   <osmesa ParaView>/bin/pvbatch slices.py <path> <outdir> [options]

<path> can be one case folder (.../solve/a+25.0_b+00.0), a solve folder, a batch folder, or the root holding the
batch folders: every case found under it is rendered.

Output: in each case's folder, <field>_sweep_x_000.png, <field>_sweep_x_001.png ... (x slices from upstream to downstream), then
<field>_sweep_y_000.png ... (y slices from port to starboard).

Examples
  slices.py C:\dev\test_cfd\batch1_000\solve\a+25.0_b+00.0 out --axis x
  slices.py C:\dev\test_cfd\batch1_000\solve\a+25.0_b+00.0 out --axis y --camera side
  slices.py C:\dev\test_cfd out --slice-field Cp --wall yPlus
  slices.py C:\dev\test_cfd out --slice-field Uz --axis x     (crossflow on the x slices: the vortex swirl)

Colour ranges: Cp uses CP_RANGE (the same as the wall scripts), |U| 0 to 1.5 U_inf, Ux, Uy and Uz +-0.5 U_inf,
yPlus and Cf fixed defaults. For k, nut and p, which have no natural range, the range is taken over the
slices of EVERY case found, so it is still one range for the whole set. --slice-range / --wall-range override any of
these. Cp, Ux, Uy and Uz are blue (negative) - white (0) - red (positive), so white is always exactly 0.

Ux, Uy and Uz are the disturbance the wing makes, U - U_inf, in the model frame, so undisturbed flow is white: on Ux,
blue is a velocity deficit (wake, vortex core); on an x slice, Uy and Uz show a vortex as a blue/red pair either side
of its core.

Stitch a sweep into a video:  ffmpeg -framerate 4 -i U_sweep_x_%03d.png -pix_fmt yuv420p U_sweep_x.mp4

Data (as written by src/cfd/solve.py):
  postProcessing/wallSurfaces/<time>/walls.vtp     cell data: p, yPlus, wallShearStress
  postProcessing/flowSlices/<time>/x_136mm.vtp, y_m118mm.vtp, ...   cell data: k, nut, p, U
  solve_case.json                                  freestream_velocity_m_s, for the dynamic pressure
  p and wallShearStress are kinematic (incompressible), so Cp = p / (0.5 U_inf^2) and Cf = |tau| / (0.5 U_inf^2).
"""
import argparse
import glob
import json
import math
import os
import re
import sys

from paraview.simple import *

# ---- settings ---------------------------------------------------------------
U_INF = 20.0                 # m/s, used only when a case has no solve_case.json
CP_RANGE = (-4.0, 1.0)       # keep in step with wall_cp_animation.py and src/cfd/slice_data.py
DEFAULT_WALL_RANGE = {"Cp": CP_RANGE, "yPlus": (0.0, 100.0), "Cf": (0.0, 0.01)}
DEFAULT_SLICE_RANGE = {
    "U": (0.0, 1.5 * U_INF), "Cp": CP_RANGE,
    "Ux": (-0.5 * U_INF, 0.5 * U_INF), "Uy": (-0.5 * U_INF, 0.5 * U_INF), "Uz": (-0.5 * U_INF, 0.5 * U_INF),
}
U_COMPONENTS = {"Ux": "X", "Uy": "Y", "Uz": "Z"}   # disturbance U - U_inf, model frame: x aft, y starboard, z up
SIGNED_FIELDS = ("Cp", "Ux", "Uy", "Uz")   # drawn blue - white - red with white at exactly 0
DIVERGING = ((0.23, 0.30, 0.75), (1.0, 1.0, 1.0), (0.71, 0.02, 0.15))   # keep in step with the wall scripts
FIELD_TITLES = {"U": "|U| (m/s)", "Ux": "Ux - U_inf,x (m/s)", "Uy": "Uy - U_inf,y (m/s)", "Uz": "Uz - U_inf,z (m/s)"}
SLICE_LIGHTING = (1.0, 0.0)   # (ambient, diffuse): slices are flat, so unlit, showing their true colours
WALL_LIGHTING = (0.6, 0.4)    # mostly unlit so white reads as white, with a little shading to show the shape
TEXT_COLOUR = [0, 0, 0]      # labels and colour bars, on the white background
# ------------------------------------------------------------------------------

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("path", help="case folder, or a folder containing cases")
ap.add_argument("outdir", help="where PNGs are written (one subfolder per case)")
ap.add_argument("--axis", choices=["x", "y", "all"], default="all", help="which slices to use")
ap.add_argument("--slice-field", default="U",
                help="U (magnitude), Ux, Uy, Uz (U - U_inf components, model frame), Cp, k, nut or p")
ap.add_argument("--slice-range", nargs=2, type=float, help="colour range for the slices")
ap.add_argument("--wall", choices=["Cp", "yPlus", "Cf"], default="Cp", help="wall surface colouring")
ap.add_argument("--wall-range", nargs=2, type=float, help="colour range for the walls")
ap.add_argument("--camera", choices=["iso", "top", "side", "front", "back"], default="iso")
ap.add_argument("--zoom", type=float, default=1.0, help=">1 zooms in")
ap.add_argument("--size", nargs=2, type=int, default=[1600, 900], metavar=("W", "H"))
args = ap.parse_args()

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
    return [c for c in sorted(set(cands))
            if glob.glob(os.path.join(c, "postProcessing", "wallSurfaces", "*", "walls.vtp"))]


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
        if os.path.getsize(f) == 0:   # e.g. an interrupted copy
            print("  skipping empty slice file %s" % f)
            continue
        out.append((ax, -mm if sign == "m" else mm, f))
    return sorted(out)        # by axis, then position


def freestream(case):
    """The case's freestream velocity vector in the model frame (U_INF along +x if it has no solve_case.json)."""
    path = os.path.join(case, "solve_case.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)["freestream_velocity_m_s"]
    print("  %s: no solve_case.json, assuming U_inf = %g m/s along x" % (os.path.basename(case), U_INF))
    return [U_INF, 0.0, 0.0]


def dynamic_pressure(u_inf):
    """Kinematic 0.5 U_inf^2."""
    return 0.5 * sum(c * c for c in u_inf)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def try_set(obj, attr, value):
    """Property names shift a little between ParaView versions; don't die on cosmetics."""
    try:
        setattr(obj, attr, value)
    except Exception as e:
        print("  (skipped %s: %s)" % (attr, e))


def fixed_lut(name, rng):
    """The colour map for `name`, pinned to `rng`: ParaView never rescales it to a case's data. Signed fields
    (SIGNED_FIELDS) go blue - white - red with white at 0, wherever 0 falls in the range."""
    lut = GetColorTransferFunction(name)
    lut.AutomaticRescaleRangeMode = "Never"
    lut.RescaleTransferFunction(rng[0], rng[1])
    GetOpacityTransferFunction(name).RescaleTransferFunction(rng[0], rng[1])
    if name in SIGNED_FIELDS and rng[0] < 0 < rng[1]:
        lo, mid, hi = DIVERGING
        lut.ColorSpace = "Lab"
        lut.RGBPoints = [rng[0], *lo, 0.0, *mid, rng[1], *hi]
    return lut


def style_bar(lut, title, corner):
    bar = GetScalarBar(lut, view)
    try_set(bar, "Title", title)
    try_set(bar, "ComponentTitle", "")
    try_set(bar, "TitleColor", TEXT_COLOUR)
    try_set(bar, "LabelColor", TEXT_COLOUR)
    try_set(bar, "WindowLocation", corner)


def slice_source(reader, field, q, u_inf):
    """(source, array, component) to colour a slice by. Cp and the velocity components get a Calculator, which is
    returned for cleanup too. The components are the disturbance U - U_inf: the freestream itself (e.g. 8.5 m/s of Uz
    at 25 deg alpha in the model frame) would otherwise swamp the flow around the wing."""
    if field == "Cp":
        function = "p/%g" % q
    elif field in U_COMPONENTS:
        axis = U_COMPONENTS[field]
        function = "U_%s-(%g)" % (axis, u_inf["XYZ".index(axis)])
    elif field == "U":
        return reader, "U", "Magnitude"
    else:
        return reader, field, None
    c = Calculator(Input=reader)
    c.AttributeType, c.ResultArrayName, c.Function = "Cell Data", field, function
    return c, field, None


def data_range(src, array):
    """(min, max) of a cell array (magnitude for a vector), or None if the source does not have it."""
    src.UpdatePipeline()
    arr = src.CellData[array]
    if arr is None:
        return None
    return arr.GetRange(-1 if arr.GetNumberOfComponents() > 1 else 0)


def global_slice_range(cases, field, axis):
    """The range of `field` over the slices of every case, for fields with no natural range (k, nut, p)."""
    lo, hi = math.inf, -math.inf
    for case in cases:
        for _, _, f in list_slices(case, axis):
            r = XMLPolyDataReader(FileName=[f])
            rng = data_range(r, field)
            Delete(r)
            if rng is None:
                print("  %s has no '%s'; left out of the colour range" % (f, field))
                continue
            lo, hi = min(lo, rng[0]), max(hi, rng[1])
    if lo > hi:
        sys.exit("no slice has a cell array '%s'" % field)
    return lo, hi


def set_camera(bounds, name, zoom):
    c = [(bounds[0] + bounds[1]) / 2, (bounds[2] + bounds[3]) / 2, (bounds[4] + bounds[5]) / 2]
    diag = math.sqrt((bounds[1] - bounds[0]) ** 2 + (bounds[3] - bounds[2]) ** 2 + (bounds[5] - bounds[4]) ** 2)
    d, up = CAMERAS[name]
    n = math.sqrt(sum(x * x for x in d))
    dist = 1.5 * diag / zoom
    view.CameraFocalPoint = c
    view.CameraPosition = [c[i] + dist * d[i] / n for i in range(3)]
    view.CameraViewUp = list(up)


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


# ---------------------------------------------------------------------------
# view, colour maps and camera: set once, shared by every case
# ---------------------------------------------------------------------------
cases = find_cases(args.path)
if not cases:
    sys.exit("No cases with postProcessing/wallSurfaces/*/walls.vtp found under %s" % args.path)
print("found %d case(s)" % len(cases))

view = CreateRenderView()
view.ViewSize = args.size
try_set(view, "UseColorPaletteForBackground", 0)   # ParaView >= 5.10 ignores Background unless this is off
try_set(view, "Background", [1, 1, 1])
view.OrientationAxesVisibility = 1

sf = args.slice_field
wall_array = args.wall
wall_range = tuple(args.wall_range) if args.wall_range else DEFAULT_WALL_RANGE[wall_array]
if args.slice_range:
    slice_range = tuple(args.slice_range)
elif sf in DEFAULT_SLICE_RANGE:
    slice_range = DEFAULT_SLICE_RANGE[sf]
else:
    slice_range = global_slice_range(cases, sf, args.axis)
print("colour ranges: walls %s %s, slices %s %s" % (wall_array, wall_range, sf, slice_range))

def expected_images(case, tag, outdir):
    """Every PNG the render loop below writes for `case` (same names, same numbering)."""
    slices = list_slices(case, args.axis)
    if not slices:
        return [os.path.join(outdir, "%s_wall%s.png" % (tag, wall_array))]
    counters = {}
    paths = []
    for ax, _, _ in slices:
        i = counters.get(ax, 0)
        counters[ax] = i + 1
        paths.append(os.path.join(outdir, "%s_sweep_%s_%03d.png" % (sf, ax, i)))
    return paths


wlut = fixed_lut(wall_array, wall_range)
slut = fixed_lut(sf, slice_range)   # Cp on both: one map, one range (CP_RANGE)
camera_set = False

for case in cases:
    name = os.path.basename(case)
    batch = os.path.basename(os.path.dirname(os.path.dirname(case)))   # ...<batch>/solve/<case>
    tag = "%s_%s" % (batch, name) if batch.startswith("batch") else name
    outdir = os.path.join(args.outdir, tag)
    os.makedirs(outdir, exist_ok=True)
    if camera_set and all(os.path.exists(p) for p in expected_images(case, tag, outdir)):
        # Resume after an interrupted run. The first case is always rendered, as it frames the camera.
        print("%s: already rendered, skipping" % tag)
        continue
    u_inf = freestream(case)
    q = dynamic_pressure(u_inf)
    created = []          # for cleanup, creation order

    # ---- walls ---------------------------------------------------------
    wdir = latest_time_dir(os.path.join(case, "postProcessing", "wallSurfaces"))
    wreader = XMLPolyDataReader(FileName=[os.path.join(wdir, "walls.vtp")])
    created.append(wreader)
    if wall_array == "yPlus":
        wsrc = wreader
    else:
        calc = Calculator(Input=wreader)
        calc.AttributeType = "Cell Data"
        calc.ResultArrayName = wall_array
        calc.Function = "p/%g" % q if wall_array == "Cp" else "mag(wallShearStress)/%g" % q
        created.append(calc)
        wsrc = calc
    wdisp = Show(wsrc, view)
    try_set(wdisp, "Ambient", WALL_LIGHTING[0])
    try_set(wdisp, "Diffuse", WALL_LIGHTING[1])
    ColorBy(wdisp, ("CELLS", wall_array))
    wdisp.LookupTable = wlut
    wlut.RescaleTransferFunction(*wall_range)   # re-pin in case anything rescaled it
    wdisp.SetScalarBarVisibility(view, True)
    style_bar(wlut, "Cp (walls)" if wall_array == "Cp" and sf != "Cp" else wall_array, "Upper Right Corner")

    # ---- slices --------------------------------------------------------
    slices = list_slices(case, args.axis)
    if not slices:
        print("%s: no matching slices, rendering walls only" % tag)
    slice_objs = []       # (axis, mm, source)
    slice_disps = []
    for ax, val, f in slices:
        r = XMLPolyDataReader(FileName=[f])
        created.append(r)
        src, arr, comp = slice_source(r, sf, q, u_inf)
        if src is not r:
            created.append(src)
        d = Show(src, view)
        try_set(d, "Ambient", SLICE_LIGHTING[0])
        try_set(d, "Diffuse", SLICE_LIGHTING[1])
        ColorBy(d, ("CELLS", arr, comp) if comp else ("CELLS", arr))
        d.LookupTable = slut
        d.Visibility = 0
        slice_objs.append((ax, val, src))
        slice_disps.append(d)
    slut.RescaleTransferFunction(*slice_range)
    if slice_disps and sf != wall_array:   # the same field shares the walls' bar
        slice_disps[0].SetScalarBarVisibility(view, True)
        style_bar(slut, FIELD_TITLES.get(sf, sf), "Lower Right Corner")

    # ---- label ---------------------------------------------------------
    label = Text(Text=tag)
    created.append(label)
    ldisp = Show(label, view)
    try_set(ldisp, "WindowLocation", "Upper Left Corner")
    try_set(ldisp, "FontSize", 18)
    try_set(ldisp, "Color", TEXT_COLOUR)

    # ---- camera: framed on the first case, then never moved ----------------
    if not camera_set:
        set_camera(union_bounds([wreader] + [s[2] for s in slice_objs]), args.camera, args.zoom)
        camera_set = True

    # ---- render --------------------------------------------------------
    if not slice_disps:   # walls only
        Render(view)
        path = os.path.join(outdir, "%s_wall%s.png" % (tag, wall_array))
        SaveScreenshot(path, view, ImageResolution=args.size)
        print("saved", path)
    counters = {}
    for (ax, val, _), d in zip(slice_objs, slice_disps):
        i = counters.get(ax, 0)
        counters[ax] = i + 1
        for other in slice_disps:
            other.Visibility = 0
        d.Visibility = 1
        label.Text = "%s   %s = %d mm" % (tag, ax, val)
        Render(view)
        path = os.path.join(outdir, "%s_sweep_%s_%03d.png" % (sf, ax, i))
        SaveScreenshot(path, view, ImageResolution=args.size)
        print("saved", path)

    # ---- cleanup before the next case ----------------------------------
    for obj in reversed(created):
        Delete(obj)

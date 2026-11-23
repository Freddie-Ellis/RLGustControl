r"""Thin CFD results down to what the CFD tab's 3D viewer needs, small enough to copy off the HPC.

Keeps, for each solved condition, the last time folder's flow slices and wall surface with only the arrays the viewer
colours by, stored as float32 and zlib-compressed:
  slices  cell data p, U       (Cp, |U|, Ux, Uy, Uz are all computed from these in the viewer)
  walls   cell data p          (wall Cp)
Everything else (k, nut, yPlus, wallShearStress, point data, the 3D fields) is left behind.

Run with ParaView's Python on Iridis (or any Python that has vtk):
  <osmesa ParaView>/bin/pvbatch thin_cases.py <path> <outdir>

<path> can be one case folder (.../solve/a+25.0_b+00.0), a solve folder, a batch folder, or the root holding the
batch folders, as in slices.py. Output, one folder per condition, named as slices.py names them:
  <outdir>/<batch>_<condition>/walls.vtp
  <outdir>/<batch>_<condition>/x_136mm.vtp, y_m118mm.vtp, ...
  <outdir>/<batch>_<condition>/case.json      freestream_velocity_m_s (for Cp and U - U_inf)
case.json is written last, so a folder that has one is complete: re-running skips those (resume after a timeout).
"""
import glob
import json
import os
import re
import sys

from vtkmodules.vtkCommonCore import vtkFloatArray, vtkPoints
from vtkmodules.vtkCommonDataModel import vtkPolyData
from vtkmodules.vtkIOXML import vtkXMLPolyDataReader, vtkXMLPolyDataWriter

SLICE_ARRAYS = ("p", "U")
WALL_ARRAYS = ("p",)
SLICE_RE = re.compile(r"^([xyz])_(m|p)?(\d+)mm\.vtp$")   # as slices.py


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


def case_tag(case):
    """<batch>_<condition>, as slices.py names its output folders."""
    name = os.path.basename(case)
    batch = os.path.basename(os.path.dirname(os.path.dirname(case)))   # ...<batch>/solve/<case>
    return "%s_%s" % (batch, name) if batch.startswith("batch") else name


def thin(src, dst, arrays):
    """Copy `src` to `dst` with only the cell `arrays`, everything as float32, compressed."""
    reader = vtkXMLPolyDataReader()
    reader.SetFileName(src)
    reader.Update()
    full = reader.GetOutput()

    points = vtkPoints()
    points.SetDataTypeToFloat()
    points.DeepCopy(full.GetPoints())   # converts to float32

    out = vtkPolyData()
    out.SetPoints(points)
    out.SetPolys(full.GetPolys())
    for name in arrays:
        array = full.GetCellData().GetArray(name)
        if array is None:
            raise ValueError("%s has no cell array '%s'" % (src, name))
        small = vtkFloatArray()
        small.DeepCopy(array)
        small.SetName(name)
        out.GetCellData().AddArray(small)

    writer = vtkXMLPolyDataWriter()
    writer.SetFileName(dst)
    writer.SetInputData(out)
    writer.SetDataModeToAppended()
    writer.EncodeAppendedDataOff()   # raw binary, not base64
    writer.SetCompressorTypeToZLib()
    if not writer.Write():
        raise IOError("could not write %s" % dst)


def thin_case(case, outdir):
    tag = case_tag(case)
    target = os.path.join(outdir, tag)
    if os.path.exists(os.path.join(target, "case.json")):
        print("%s: already thinned, skipping" % tag)
        return
    os.makedirs(target, exist_ok=True)

    walls = os.path.join(latest_time_dir(os.path.join(case, "postProcessing", "wallSurfaces")), "walls.vtp")
    thin(walls, os.path.join(target, "walls.vtp"), WALL_ARRAYS)

    slices_dir = latest_time_dir(os.path.join(case, "postProcessing", "flowSlices"))
    n = 0
    for f in sorted(glob.glob(os.path.join(slices_dir, "*.vtp")) if slices_dir else []):
        if not SLICE_RE.match(os.path.basename(f)):
            continue
        if os.path.getsize(f) == 0:   # e.g. an interrupted copy
            print("  skipping empty slice file %s" % f)
            continue
        thin(f, os.path.join(target, os.path.basename(f)), SLICE_ARRAYS)
        n += 1

    info_path = os.path.join(case, "solve_case.json")
    info = {}
    if os.path.exists(info_path):
        with open(info_path) as fh:
            info = {"freestream_velocity_m_s": json.load(fh)["freestream_velocity_m_s"]}
    else:
        print("  %s: no solve_case.json; the viewer will assume its default freestream" % tag)
    with open(os.path.join(target, "case.json"), "w") as fh:   # last: marks the folder complete
        json.dump(info, fh, indent=2)
    print("%s: walls + %d slices" % (tag, n))


if len(sys.argv) != 3:
    sys.exit(__doc__)
path, outdir = sys.argv[1], sys.argv[2]
cases = find_cases(path)
if not cases:
    sys.exit("No cases with postProcessing/wallSurfaces/*/walls.vtp found under %s" % path)
print("found %d case(s)" % len(cases))
for case in cases:
    thin_case(case, outdir)

"""Render wall Cp (and optionally yPlus / Cf) for every case, same view + colour range.

Run with ParaView's own Python (NOT normal python):
  "C:\\Program Files\\ParaView 5.xx\\bin\\pvpython.exe" render_walls.py C:\\path\\to\\cases C:\\path\\to\\outdir

Expects: <root>/batch*/solve/<case>/postProcessing/wallSurfaces/<time>/walls.vtp
walls.vtp holds CELL data: p (kinematic), yPlus, wallShearStress.
UNTESTED against your ParaView version - tweak the settings block if needed.
"""
import glob
import os
import sys

from paraview.simple import *

root, outdir = sys.argv[1], sys.argv[2]
os.makedirs(outdir, exist_ok=True)

# ---- settings ---------------------------------------------------------------
U_INF = 20.0                 # m/s (speed_m_s in results.json)
Q = 0.5 * U_INF ** 2         # kinematic dynamic pressure; assumes p is kinematic, p_inf = 0
CP_RANGE = (-2.0, 1.0)       # fixed colour range so cases are comparable
SIZE = [1600, 900]
# ------------------------------------------------------------------------------

files = sorted(glob.glob(os.path.join(
    root, "batch*", "solve", "*", "postProcessing", "wallSurfaces", "*", "walls.vtp")))
print("found %d wall files" % len(files))

view = CreateRenderView()
view.ViewSize = SIZE
view.Background = [1, 1, 1]
view.OrientationAxesVisibility = 1

lut = GetColorTransferFunction("Cp")
lut.RescaleTransferFunction(*CP_RANGE)
camera_set = False

for f in files:
    d = f
    for _ in range(4):
        d = os.path.dirname(d)          # -> .../<case>
    case = os.path.basename(d)
    batch = os.path.basename(os.path.dirname(os.path.dirname(d)))
    name = "%s_%s" % (batch, case)

    reader = XMLPolyDataReader(FileName=[f])
    calc = Calculator(Input=reader)
    calc.AttributeType = "Cell Data"
    calc.ResultArrayName = "Cp"
    calc.Function = "p/%g" % Q

    disp = Show(calc, view)
    ColorBy(disp, ("CELLS", "Cp"))
    disp.LookupTable = lut
    disp.SetScalarBarVisibility(view, True)

    if not camera_set:
        view.CameraPosition = [-1.0, -1.5, 1.0]   # tweak once, reused for every case
        view.CameraFocalPoint = [0.4, 0.0, 0.0]
        view.CameraViewUp = [0, 0, 1]
        view.ResetCamera()
        camera_set = True

    Render(view)
    SaveScreenshot(os.path.join(outdir, name + "_Cp.png"), view, ImageResolution=SIZE)
    print("saved", name)

    Hide(calc, view)
    Delete(calc)
    Delete(reader)
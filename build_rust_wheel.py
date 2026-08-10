import sys
import subprocess

cwd = "rust/rust_wrapper"

def run_module():
	cmd = [sys.executable, "-m", "maturin", "develop"]
	try:
		return subprocess.run(cmd, cwd=cwd, check=False)
	except FileNotFoundError:
		return None

def run_executable():
	cmd = ["maturin", "develop"]
	try:
		return subprocess.run(cmd, cwd=cwd, check=False)
	except FileNotFoundError:
		return None

res = run_module()
if res is None or (hasattr(res, "returncode") and res.returncode != 0):
	# Try the executable fallback
	res2 = run_executable()
	if res2 is None:
		print("Error: 'maturin' not found. Install it into the active Python (pip install maturin) or ensure 'maturin' is on PATH.", file=sys.stderr)
		print("Invoking Python:", sys.executable, file=sys.stderr)
		sys.exit(1)
	elif res2.returncode != 0:
		print("'maturin' executable returned", res2.returncode, file=sys.stderr)
		sys.exit(res2.returncode)
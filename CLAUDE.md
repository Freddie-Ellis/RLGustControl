# CLAUDE.md

## Project
University Part IV GDP: online estimation and reinforcement-learning-based control for unsteady aerodynamic systems hit by large-amplitude gusts. Goal: detect disturbances from surface pressure sensors on a 3D delta wing model (~50 taps), estimate unsteady forces, and synthesise controllers that hold a desired aerodynamic performance (e.g. zero pitching moment) during gust encounters.

## Modelling approach
Combine Physics Augmented Auto Encoders (PA-AE), LSTM networks and RL to control the wing in extreme aerodynamic conditions.

## Code structure
- Python is the main language (`main.py`, Python >= 3.14, managed with uv; `pyproject.toml` / `uv.lock`).
- `ruststuff/` is a Rust backend compiled into a Python module with maturin. It holds performance-critical code while keeping access to Python libraries. Type stubs live in `ruststuff.pyi`; keep them in sync with the Rust API.
- `benchmarks/` holds benchmarks.
- `ruststuff` is installed from the wheel in `ruststuff/target/wheels/`, so rebuild the wheel after changing Rust code.

## Libraries
Use these, and don't substitute alternatives:
- Polars (not Pandas)
- qtpy (any custom UI)
- PyTorch
- ruststuff (custom Rust backend)
- Matplotlib (one-off plotting only)
- Pydantic (serialisable, safe typing)

## Rules
- All Python code must be fully type hinted.
- Use Polars, never Pandas.
- Put performance-critical code in the Rust backend rather than slow Python loops.
- Ask before adding new dependencies.

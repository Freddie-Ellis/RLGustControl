use pyo3::prelude::*;

#[pyfunction]
fn step(x: f64, v: f64, action: f64) -> (f64, f64) {
    rust_logic::step(x, v, action)
}

#[pymodule]
fn rust_wrapper(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(step, m)?)?;
    Ok(())
}
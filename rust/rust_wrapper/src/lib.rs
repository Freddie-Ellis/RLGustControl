use pyo3::prelude::*;

#[pyfunction]
fn step(x: f64, v: f64, action: f64) -> (f64, f64) {
    rust_logic::step(x, v, action)
}

#[pyfunction]
fn bench_iter(iters: usize) -> PyResult<(f64, f64, f64, Vec<(f64, f64)>)> {
    let (secs, x, v, his) = rust_logic::bench_iter(iters);
    Ok((secs, x, v, his))
}

#[pymodule]
fn rust_wrapper(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(step, m)?)?;
    m.add_function(wrap_pyfunction!(bench_iter, m)?)?;
    Ok(())
}
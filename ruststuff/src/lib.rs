pub mod step;
pub mod bench_iters;

use pyo3::prelude::*;

#[pymodule]
fn ruststuff(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(step::step, m)?)?;
    m.add_function(wrap_pyfunction!(bench_iters::bench_iter, m)?)?;
    Ok(())
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn step_moves_state() {
        let (x, v) = step::step(0.0, 0.0, 1.0);
        assert!(x != 0.0 || v != 0.0);
    }

    #[test]
    fn bench_iter_runs() {
        let iters: usize = 20000000;
        let (secs, x, v, his) = bench_iters::bench_iter(iters);
        assert!(secs >= 0.0);
        // Make a plot of the history to visualize the trajectory
        use plotters::prelude::*;

        let root = BitMapBackend::new("bench_iter_plot.png", (1600, 1200)).into_drawing_area();
        root.fill(&WHITE).unwrap();

        let x_min = his.iter().map(|(x, _)| *x).fold(f64::INFINITY, f64::min);
        let x_max = his.iter().map(|(x, _)| *x).fold(f64::NEG_INFINITY, f64::max);
        
        let v_min = his.iter().map(|(_, v)| *v).fold(f64::INFINITY, f64::min);
        let v_max = his.iter().map(|(_, v)| *v).fold(f64::NEG_INFINITY, f64::max);

        let mut chart = ChartBuilder::on(&root)
            .caption("Position Trajectory", ("sans-serif", 40))
            .margin(10)
            .x_label_area_size(30)
            .y_label_area_size(30)
            .build_cartesian_2d(x_min..x_max, v_min..v_max)
            .unwrap();
        chart.configure_mesh().draw().unwrap();
        chart.draw_series(LineSeries::new(
                his.iter().map(|(x, v)| (*x, *v)),
                &RED,
            ))
            .unwrap()
            .label("Trajectory")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));
        chart.configure_series_labels().background_style(&WHITE.mix(0.8)).draw().unwrap();
        println!("bench_iter: elapsed={:.6}s, final_x={:.6}, final_v={:.6}, history_len={}", secs, x, v, his.len());
    }
}

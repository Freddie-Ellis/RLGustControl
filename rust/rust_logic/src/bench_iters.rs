use crate::step::step;

/// Run `step` for `iters` iterations and return (elapsed_seconds, final_x, final_v, history).
pub fn bench_iter(iters: usize) -> (f64, f64, f64, Vec<(f64, f64)>) {
    use std::time::Instant;
    let mut x: f64 = 0.0;
    let mut v: f64 = 0.0;
    let mut his: Vec<(f64, f64)> = Vec::with_capacity(iters);
    let action: f64 = 1.0;
    let start = Instant::now();
    for _ in 0..iters {
        let (nx, nv) = step(x, v, action);
        x = nx;
        v = nv;
        his.push((x, v));
    }
    let dur = start.elapsed();
    (dur.as_secs_f64(), x, v, his)
}

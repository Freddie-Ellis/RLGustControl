pub fn step(x: f64, v: f64, action: f64) -> (f64, f64) {
    let a = -0.5 * x - 0.1 * v + action;
    let v_new = v + a * 0.01;
    let x_new = x + v_new * 0.01;
    (x_new, v_new)
}

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn step_moves_state() {
        let (x, v) = step(0.0, 0.0, 1.0);
        assert!(x != 0.0 || v != 0.0);
    }
}
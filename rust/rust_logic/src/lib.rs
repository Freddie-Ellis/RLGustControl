pub fn step(x: f64, v: f64, action: f64) -> (f64, f64) {
    let a = -0.5 * x - 0.1 * v + action;
    let v_new = v + a * 0.01;
    let x_new = x + v_new * 0.01;
    (x_new, v_new)
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
import time

import matplotlib.pyplot as plt

import ruststuff


def step_py(x: float, v: float, action: float):
    a = -0.5 * x - 0.1 * v + action
    v_new = v + a * 0.01
    x_new = x + v_new * 0.01
    return x_new, v_new

def bench_py(iters: int) -> tuple[float, float, float, list[tuple[float, float]]]:
    x = 0.0
    v = 0.0
    his = []
    action = 1.0
    start = time.perf_counter()
    for _ in range(iters):
        x, v = step_py(x, v, action)
        his.append((x, v))
    elapsed = time.perf_counter() - start
    return elapsed, x, v, his

def bench_rust(iters: int) -> tuple[float, float, float, list[tuple[float, float]]]:
    # If ruststuff exposes a bench_iter that runs the loop inside Rust, use it
    secs, x, v, his = ruststuff.bench_iter(iters)
    return secs, x, v, his
    
def main() -> None:
    iters = 20000000  # 20 million iterations
    print(f"Running benchmark with {iters} iterations...")
    t_py, _, _, his_py = bench_py(iters)
    t_rust, _, _, his_rust = bench_rust(iters)

    print(f"Python: elapsed={t_py:.6f}s, calls/s={iters/t_py:.2f}, ns/call={t_py*1e9/iters:.2f}")
    print(f"Rust (pyo3): elapsed={t_rust:.6f}s, calls/s={iters/t_rust:.2f}, ns/call={t_rust*1e9/iters:.2f}")
    plt.figure(figsize=(10, 5))
    plt.plot([x for x, v in his_py], [v for x, v in his_py], label='Python', color='blue')
    plt.plot([x for x, v in his_rust], [v for x, v in his_rust], label='Rust', color='orange')
    plt.xlabel('Velocity (v)')
    plt.ylabel('Position (x)')
    plt.title(f'Position Trajectory: {iters} iterations')
    plt.legend()
    plt.savefig('benchmarks\\trajectory.png')

if __name__ == "__main__":
    main()

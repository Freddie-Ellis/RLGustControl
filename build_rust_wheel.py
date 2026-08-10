import subprocess

subprocess.run(["maturin", "develop"], cwd="rust/rust_wrapper", check=False)
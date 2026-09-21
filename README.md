# Prym–Green ancillary code

Code for *The Prym–Green conjecture in even genera up to 30*, by Sam Payne and
Thomas Willwacher. It reproduces the computations in even genera 20–30 using
the paper's fixed parameters.

Each run computes its Wiedemann sequences from scratch on an NVIDIA GPU,
recovers rank lower bounds on the CPU, and performs the kernel and quadratic
deformation checks when needed. The six small mathematical inputs are included.
The original computations are summarized in [RESULTS.md](RESULTS.md).

The CPU tests, a small independent CUDA comparison, and complete reruns of
genera 20, 22, and 24 pass on Linux with CUDA. [Validation notes](docs/validation.md)
record the tested software versions and fresh computations.

## Build

Required:

- Linux with an NVIDIA GPU and CUDA toolkit (`nvcc`), plus a compatible host
  C++ compiler. The paper used RTX PRO 6000 Blackwell GPUs.
- Rust 1.93.0 and Cargo, normally installed through `rustup`; the toolchain is
  pinned in `rust-toolchain.toml`.
- Python 3.12 or 3.13, with the pinned NumPy and SciPy dependencies below.
  NumPy reconstructs the deformation inputs; SciPy is used by developer tests.
- The zstd development library/headers and GNU Make.

From the repository directory:

```sh
python3.13 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
make
```

The default CUDA target is `sm_120`, matching the paper's Blackwell GPUs. To
select another supported architecture or host compiler, for example:

```sh
make CUDA_ARCH=89 CUDA_CXX=/path/to/g++
```

Set `NVCC=/path/to/nvcc` if the toolkit is not on PATH. Changing compiler or
architecture settings requires removing the four generated CUDA executables
before rebuilding. Their names appear in `.gitignore`.

Both CPU programs are bundled; no sibling repository is required. On a
machine without CUDA, build and test the CPU components with:

```sh
make cpu
make test
```

## Run

Run one genus from input verification through its final rank/residual checks:

```sh
./reproduce --genus 20
./reproduce --genus 24
```

Select GPUs, the total CPU thread budget, and a scratch directory if needed:

```sh
./reproduce --genus 22 --gpus 0,1 --threads 64 --output /scratch/my-prym-runs
```

Run all six genera sequentially with:

```sh
./reproduce --all --gpus 0,1 --threads 64 --output /scratch/my-prym-runs
```

The numerical settings are fixed. GPU and CPU work overlap, with one CUDA job
per selected GPU and up to two CPU rank jobs per GPU. The total CPU thread
budget is divided between those rank workers. Kernel and deformation jobs
have priority and start once their inputs are ready, while other base sectors
continue in the background. Different genera run sequentially.

A successful run prints `PASS g=...` and the path to `summary.json`. The
summary contains all required rank bounds and deformation-check outcomes;
per-stage logs and working files are in the same fresh directory. A failure
exits nonzero, stops active child processes, and retains diagnostics. Repeating
the command starts a new computation in a new directory. There is no
saved-WDM input, download, or resume command.

## Resources

The original campaigns used two NVIDIA RTX PRO 6000 Blackwell GPUs with 96 GB
each, 64 CPU cores, and 1 TB host RAM. Approximate elapsed times were:

| Genus | Original elapsed time |
|---:|---:|
| 20 | 1 minute |
| 22 | 2.1 minutes |
| 24 | 7.5 minutes |
| 26 | 49 minutes |
| 28 | 15.9 hours |
| 30 | 10.6 days |

Start with genus 20 or 22. Minimum resource requirements for the larger cases
have not been established. Large runs generate WDM files and vectors locally;
the original genus-30 campaign alone retained about 6.8 GB of compressed WDM
data. Preserve enough disk space for intermediate states and logs.

## Code and checks

- `src/`: exact section-space algebra, cyclic verification, and deformation
  applications on the CPU.
- `cuda/`: matrix-free sequence generation and kernel recovery.
- `third_party/bcw/`: the attributed CPU sequence-to-rank solver.
- `scripts/reproduce.py`: the complete fixed pipeline.
- `scripts/prepare_deformation.py`: reconstruction of paired Taylor inputs.
- `data/`: six small frozen inputs and the fixed paper specifications.
- `tests/`: exact small reference calculations and regression checks.

[Implementation notes](docs/implementation.md) connect the paper's notation
to the functions, tensor layouts, sign conventions, and verification steps.
`make test` checks all six small inputs and includes an independent explicit
genus-12 deformation calculation. CPU tests do not substitute for GPU testing.

After building, the small CUDA integration test can be run with:

```sh
python tests/check_cuda.py --device 0 --output runs/cuda-tests
```

It checks all stages on a tiny genus-12 instance against independent exact
matrices, including recovered kernel, correction, and obstruction vectors.
It takes a few seconds. Full paper computations use `./reproduce` as above.

Licensing and origins are recorded in [THIRD_PARTY.md](THIRD_PARTY.md).

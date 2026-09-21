# Validation of the ancillary implementation

The extracted code has been built and tested on Linux with CUDA. The paper
reruns below start from the included mathematical inputs and generate new
sequences. Historical WDM files are not used.

## Build and test

The tested software versions are CUDA 13.4 (nvcc V13.4.92), GCC 15.3.0,
Rust/Cargo 1.93.0, Python 3.13.15, NumPy 2.3.2, and SciPy 1.18.1. The CUDA
build target was `sm_120`. Create and activate the Python environment using
the instructions in [README.md](../README.md), then run:

```sh
make
make test
python tests/check_cuda.py --device 0 --output runs/cuda-tests
./reproduce --genus 20 --output runs
```

If the compiler tools are not on PATH, specify `NVCC=/path/to/nvcc` and
`CUDA_CXX=/path/to/g++` in the `make` command. Select an appropriate scratch
directory with `--output` for larger runs.

All 44 CPU tests pass: 18 rank-solver tests, 15 geometry/deformation Rust tests,
and 11 Python tests. All four CUDA executables compile successfully.

## Independent GPU comparison

The GPU integration test uses g12 over F_661. It runs all six base sectors,
recovers the sector-zero kernel, solves both augmented correction systems,
and verifies the replacement rank. Three deterministic blocks per base sector
compare CPU and CUDA Gram applications. Recovered kernel, correction, and
obstruction vectors also agree with an independent exact matrix calculation
after accounting for kernel basis choice.

The initial CUDA build found two extraction errors: missing WDM residue
normalization and duplicate template declarations. Both were corrected in
`eb08ee2`; no numerical algorithm changed. The first tiny GPU trial used
F_109, where the fixed preconditioner lowered the augmented sector-5 Gram
rank from 84 to 83. Independent elimination confirmed both ranks, and all
93 CUDA moments matched exact matrix powers. The integration test therefore
uses F_661; the CPU fixture still uses F_109. Paper primes and seeds were
unchanged. A deficient rank bound is rejected by the runner.

## Complete paper reruns

| Genus | Required rank computations | Result |
|---:|---:|---|
| 20 | 13 | All ranks, kernel checks, and correction residuals passed. |
| 22 | 11 | Every base sector has full column rank. |

[validation-results.json](../data/validation-results.json) records every fresh
rank and residual outcome. The tested production source is `eb08ee2`; later
commits add the integration test and documentation without changing those
executables. Each execution also retains its detailed `summary.json` and
logs locally in the selected output directory.

The larger genera await complete fresh reruns. Their historical results
remain in [RESULTS.md](../RESULTS.md).

# Production CUDA operators

These four translation units are extracted from `prymgreene` revision
`eb70fbf55dd3e9fa29b4397d2617086cff490e2b`. The root Makefile builds them;
the root `reproduce` script coordinates them. They require CUDA and zstd.
The extracted code builds on CUDA 13.4 and has passed the small independent
GPU integration test; see [validation notes](../docs/validation.md) for the
complete paper cases tested and the compiler environment.

| Executable | Purpose |
|---|---|
| `cuprym_cyclic` | Generate one eliminated cyclic-sector sequence. |
| `cuprym_cyclic_kernel_vectors` | Recover original-coordinate base kernel candidates. |
| `cuprym_deformation` | Generate an augmented or replacement sequence. |
| `cuprym_deformation_kernel_vectors` | Recover augmented-system kernel candidates. |

The cyclic interface accepts only `--operator eliminated`. The deformation
interface additionally accepts `augmented` and `replacement`, with
`--dense-file` and, for replacement, `--drop-columns`. Every invocation uses
one explicit `--sector`. The caller chooses `--device` and output paths.

Sequence generation uses block width 4, numerical seed 1, two target-mixing
rounds, mixing seed 10001, and no intermediate saves. The old explicit flags
`-v 4 --seed 1 --rowmix-rounds 2 --rowmix-seed 10001 --saveafter 0`
are accepted for the internal runner; other values are rejected. The sequence
length and safe dot-product chunk are chosen internally. The
`--validate-rowmix` flag retains the developer CPU/CUDA comparison of the
base eliminated operator before a run; it is not a shortened benchmark.

Example internal sequence command:

```sh
cuprym_cyclic data/g22.json --operator eliminated --sector 0 \
  --out runs/example --device 0 -v 4 --seed 1 \
  --rowmix-rounds 2 --rowmix-seed 10001
```

It writes
`eliminated-sector00-v4-rowmix-r2-s10001.wdm.zst` and a matching
`.rowmix.json` sidecar. Deformation filenames replace `eliminated` with
`augmented` or `replacement`. Existing WDM files and sidecars are rejected.
The CPU `prym-rank` command consumes the completed WDM and writes its rank
result and, where needed, polynomial generator.

Recovery takes the same fixture, operator, sector, and device, plus
`-f FILE.wdm.zst -g FILE_generators.txt`. Its default output names append
`_nullvectors_1.txt` (preconditioned coordinates), `_nullvectors_2.txt`
(original coordinates), and `_nullvectors_report.json` to the **whole WDM
filename**. Preserve the sidecar: recovery checks its dimensions, mixing
recipe, seeds, and hashes. Only original-coordinate candidates enter the
subsequent exact Rust residual checks.

## Representation

The multiplication tensor, right inverse, kernel inclusion, and subset
incidences define the operator. Forward application computes
`T=D_m X`, `Y=B Z-R T`, then `D_(m-1) Y`; the transpose reverses the factors.
Neither the rectangular operator nor its Gram matrix is assembled.
The Gram iteration applies the fixed source and target preconditioners and
`paired-sl2-target-v1` row mixing. Dense vector blocks are row-major;
WDM stores each vector column on its own line and each upper-triangular
moment entry as a sequence. `matrices.h` performs that layout conversion.

`cyclic_cuda_helpers.h` contains the factor kernels and layouts.
`prym_cuda_helpers.h` retains only shared dense buffers, modular arithmetic,
and moment kernels. `wdmfiles.h` and `zstd_compat.h` preserve the WDM format.
The CLI11 header and its license notice are retained unchanged.

The unused ordinary-Phi route, generic CSR matrix readers, random-point
kernels, and public benchmark/tuning options were removed. Production
factor kernels, modular overflow checks, row-mixing recipes, and recovery
formulas were retained. The GPU test in `tests/check_cuda.py` compares the
small full pipeline with exact matrices and checks CPU/CUDA base-sector
applications before sequence generation.

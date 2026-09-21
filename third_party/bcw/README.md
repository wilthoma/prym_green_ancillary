# CPU rank recovery

This directory contains the production subset of Thomas Willwacher's
`bcw_rank` at revision `df34d858e9704146befc1db856e8141c7fc70dd4`, under the
included MIT license. Build it through the root Cargo workspace. Its executable
is `prym-rank`; the repository's `reproduce` script supplies its inputs.

```sh
prym-rank path/to/sector.wdm.zst --threads 8 -g
```

The positional input must be a **completed** WDM file. The program loads it
once and fails if its sequence is insufficient for the production stopping
condition. It neither polls the CUDA producer nor creates a `.stop` file.
The optional `-g` writes the polynomial generator needed to recover kernel
vectors on CUDA. The original names and contents of the outputs are retained:

- `sector_result.txt`: matrix dimensions and the exact rank lower bound.
- `sector_generators.txt`: one coefficient matrix per line, in the convention
  consumed by the CUDA kernel-recovery executables.

The suffixes `.wdm.zst` and `.wdm` are removed before appending these names.
WDM compression is detected from its zstd magic bytes, rather than its suffix.

## Mathematical conventions

The CUDA producer records the upper triangle of the symmetric moments
`V^T G^k V`, including the initial `V^T V` term. `rank_pipeline` removes that
initial term before constructing the order-basis problem. Polynomial matrices
are indexed `[row][column][degree]` throughout. The shifted-degree rank bound,
stopping threshold (10), NTT arithmetic bounds, and generator reversal and
transpose conventions are inherited from the production implementation.

The bound concerns the supplied matrix through its preconditioned Gram
operator. A successful invocation need not prove full rank: the outer runner
must compare the bound to the required column count. A deficient bound can
reflect insufficient information from the chosen projections. Kernel vectors
and correction vectors are checked separately against the original operators.

## Files and extraction changes

`rank_pipeline.rs` coordinates the order basis and rank bound;
`sigma_basis.rs` computes and exports it; `poly_mat_mul.rs` and `ntt.rs`
implement exact polynomial products; `modular_linalg.rs` performs small exact
finite-field matrix operations; `wdm_files.rs` preserves the file layout.
`nalgebra` supplies integer matrix storage and products here, not floating
rank estimates.

The narrow command replaces the upstream live-producer coordinator. Unrelated
matrix formats, floating rank comparisons, SIMD implementations, benchmark
tests, unused diagnostics, and inactive timing counters were excluded.
Production algebra and generator encoding were retained. Arithmetic tests
use fixed random seeds. `tests/complete_file.rs` checks a known rank-two
sequence, the exported generator, and rejection of insufficient input.

Run `cargo test -p prym-bcw` from the repository root.

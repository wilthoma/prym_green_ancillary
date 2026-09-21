# Journal ancillary repository plan

Agreed scope, 21 September 2026. The base version has now been extracted and
passes local CPU checks. CUDA validation awaits the author's push and clone
on ada-32. This document preserves the design decisions; README.md describes
the implemented commands and current status.

## Scope

Ship the necessary code, six small fixed inputs, good documentation, and a
compact summary of the paper's computations. Readers select a genus and run
its complete computation on a CUDA machine. An optional all-genera command
runs the same six computations.

There is no downloadable computation archive, saved-Wiedemann replay
workflow, or option to start from supplied WDM files. WDM files are generated
internally during each fresh computation.

Use the single method in Sections 3–4 of *The Prym–Green conjecture in even
genera up to 30*: cyclic decomposition, the eliminated matrix-free operator,
CUDA block Wiedemann sequences, and CPU rank recovery. Include the paired
quadratic deformation for genera 20, 24, 28. Fix all mathematical and numerical
parameters to the paper values.

The base version has passed local CPU checks. The author next pushes it to
GitHub and clones it on ada-32, where CUDA validation will take place.

## Layout

```text
README.md                 Prerequisites, build, run, resource requirements
RESULTS.md                Compact summary of the original computations
Makefile                  Build and developer test targets
reproduce                 Run one genus or all six, end to end
Cargo.toml / Cargo.lock    Rust code and locked dependencies
rust-toolchain.toml        Tested toolchain
requirements.txt          Python dependencies
LICENSE / THIRD_PARTY.md   Project license and third-party attribution

src/                      Geometry, cyclic decomposition, deformation
cuda/                     Required GPU sequence and kernel-recovery code
third_party/bcw/           Required CPU rank solver, with its license
scripts/                  Internal preparation and execution helpers
data/                     Six small inputs and fixed parameter specifications
tests/                    Small reference calculations and correctness checks
docs/implementation.md    Paper equations, code structure, conventions
runs/                     Generated working files; ignored by Git
```

Use plain Markdown and code comments/docstrings. Keep attribution and source
revisions in THIRD_PARTY.md and original campaign references in RESULTS.md.
The repository must build without either sibling repository.

## Reader workflow

After installing the documented prerequisites:

```sh
make
./reproduce --genus 20
./reproduce --genus 24
./reproduce --all
```

These commands are implemented in the base version. Each selected genus runs
from the fixed mathematical input through input checks, CUDA sequence
generation, CPU rank recovery, kernel/deformation calculations where needed,
and the final mathematical checks. The runner prints a concise result and
automatically saves a small summary and logs.

CUDA is required. Optional settings should be limited to GPU allocation,
a total CPU thread budget, and the working directory. Parameters, block width,
preconditioners, and seeds are fixed internally. Calling the runner without
a genus or --all prints usage.

Large WDM files and vectors are internal working files generated locally
under runs/ or the selected scratch directory. They are ignored by Git and
are not distributed. Preserve the necessary existing execution logic without
adding a generic workflow or checkpoint framework. A failed run must stop
clearly and retain its diagnostic files. Do not offer automatic resumption
as a reader workflow in the initial version.

Developer tests can run without CUDA where possible. There are no separate
reader-facing check, replay, download, or report subcommands.

## Inputs and results summary

The six prepared inputs total 1,416,205 bytes, about 1.42 MB. Include these
small files and their exact parameter specifications. All cases use orbit
representatives (1,2),(4,10), Prym signs -1, a pencil of characters 1,m-1,
and elimination section of character 2, where m=g/2.

| g | p | zeta | Sectors | Columns per sector | Required result | Original elapsed time |
|---:|---:|---:|---:|---:|---|---:|
| 20 | 661 | 190 | 10 | 19,448 | Nonzero sectors full; sector 0 nullity 2; replacement full | 61 s |
| 22 | 661 | 9 | 11 | 75,582 | Every sector full | 125 s |
| 24 | 1009 | 160 | 12 | 293,930 | Nonzero sectors full; sector 0 nullity 2; replacement full | 449 s |
| 26 | 1093 | 11 | 13 | 1,144,066 | Every sector full | 2,941 s |
| 28 | 1009 | 74 | 14 | 4,457,400 | Nonzero sectors full; sector 0 nullity 2; replacement full | 57,077 s |
| 30 | 1051 | 136 | 15 | 17,383,860 | Every sector full | 919,735 s |

For the deformation cases the first orbit has velocities
dotP_j=zeta^(2j), dotQ_j=1, with zero-based j; the second orbit is stationary.
Correction sectors are 1,m-1. All selected runs use block width 4, seed 1,
two row-mix rounds, row-mix seed 10001, and paired-sl2-target-v1.

RESULTS.md should contain parameters, dimensions, per-sector ranks, kernel
dimensions, residual-check outcomes, replacement ranks, timings, hardware,
and campaign references. Summarize the existing records rather than importing
collections of old manifests and logs. Identify the table as a record of the
authors' computations; fresh executions perform the rank and residual checks
again.

The recorded runs used two RTX PRO 6000 Blackwell GPUs with 96 GB each, 64 CPU
cores, and 1 TB RAM. Timings include overlapped CPU work and waiting, but
exclude initial fixture preparation. Measure resource requirements on the
extracted version. Genus 30 historically took about 10.6 days.

Selected campaigns, relative to
/scratch/userdata/wilthoma/prymgreene_data/:

| g | Host | Campaign |
|---:|---|---|
| 20 | ada-32 | paper_uniform_20260921_103216/g20-paired |
| 22 | ada-32 | paper_uniform_20260921_103216/g22 |
| 24 | ada-31 | deformation_timing_g24_20260907_100829/g24 |
| 26 | ada-31 | paper_uniform_20260921_104029/g26v4 |
| 28 | ada-31 | deformation_g28_20260907_102247/g28 |
| 30 | ada-32 | cyclic_profile_g30_20260907_113440 |

Read-only SSH inspection confirmed the expected rank files and values. Exact
g20 and g26 inputs were retrieved to the temporary local audit folder with
matching hashes; g22, g24, g28, g30 inputs were already available locally.
The old checked-in g20 fixture uses an alternating deformation and must not
be used. The original server outputs remain untouched.

## Source extraction

Retain the finite-field and section algebra, cyclic construction, paired
Taylor setup, deformation applications and residual checks from prymgreene.
Keep only the necessary parts of points.rs, sections.rs and artinian.rs, plus
field.rs, linear.rs, binary_form.rs, subsets.rs, sha256.rs, cyclic.rs and
deformation.rs.

The CUDA path needs cuprym_cyclic.cu, cuprym_cyclic_kernel_vectors.cu,
cuprym_deformation.cu, cuprym_deformation_kernel_vectors.cu and their required
headers. Reuse the necessary execution logic from the existing cyclic and
deformation schedulers.

Bundle the required bcw_rank CPU modules: rank_pipeline, wdm_files,
sigma_basis, poly_mat_mul, ntt, modular_linalg, and a reduced entry point.
Retain the independent small geometry checks and deformation reference for
developer validation.

Remove experimental algorithms, random-point campaigns, parameter searches,
alternative deformations, old CPU Wiedemann backends, matrix exporters,
benchmarks, failed runs, and personal server paths. Preserve tested algebra
and kernels while removing unused code; avoid a broad GPU rewrite.

Current extraction source references:

- prymgreene: eb70fbf55dd3e9fa29b4397d2617086cff490e2b.
- bcw_rank: df34d858e9704146befc1db856e8141c7fc70dd4.

These are extraction snapshots, not established historical build revisions
for every recorded run. Retain the BCW MIT notice and CLI11 license. The Prym
crate declares MIT OR Apache-2.0; preserve that starting license choice unless
the authors change it.

Pin a tested Rust toolchain and dependencies. Stable Rust appears sufficient
after removing old SIMD backends, but must be checked. CUDA requires nvcc,
a compatible host compiler, and zstd. Taylor preparation uses NumPy; the
independent reference uses NumPy/SciPy. Document prerequisites and make GPU
architecture configurable at build time.

## Documentation and correctness

README.md covers installation, run examples, expected output, resources,
and troubleshooting. docs/implementation.md follows the paper from section
spaces to Artinian reduction, elimination, cyclic sectors, deformation, and
rank recovery, linking each step to its functions.

Use module-level explanations and comments/docstrings to specify bases,
tensor layouts, weights, Koszul signs, Taylor conventions, and arithmetic
bounds. Explain which work happens on CPU and GPU. Keep format details with
the relevant implementation.

Preserve the essential verification conditions:

- Check nodal points, section constraints, the pencil including infinity,
  multiplication tensors, and quadratic normal generation.
- Check right-inverse and kernel-inclusion identities through Taylor order two.
- For odd m, attain the column count as a rank lower bound in every sector.
- For even m, attain full rank in every nonzero sector; combine a sector-zero
  lower bound N-2 with two independent exact kernel vectors; verify correction
  residuals and full rank N for the quadratic replacement matrix.

Document the BCW lower-bound guarantee and why these conditions suffice.
Preserve WDM layout, initial Gram-term removal, generator conventions,
preconditioning, original kernel-coordinate recovery, and overflow guards.
The code solves F0 S_code = F1 K and forms Z=F2 K-F1 S_code; the paper uses
S_paper=-S_code. Taylor coefficients mean coefficients of t^k, not kth
derivatives. Reproducibility concerns mathematical results; seeds alone do
not guarantee identical random vectors across compiler/STL versions.

## Delivery sequence

1. Extract the necessary code, include the six small inputs, and write the
   results summary, README, and implementation documentation.
2. Run local CPU tests: input/geometry checks, small independent operator and
   transpose comparisons, Taylor identities, and rank/interface examples.
3. Deliver the base version for the author's GitHub push and ada-32 clone.
4. On ada-32, check short CPU/CUDA comparisons, then run complete g20 and g22
   cases (both algorithms), followed by g24. Record resource use.
5. Validate larger cases explicitly before claiming they were rerun with the
   extracted code. Finalize documentation and tag the journal version.

The planning audit reviewed the paper, source repositories, corrected
paper_uniform records, and discussions in the ChatGPT project Prym Green
Conjecture. Independent geometric checks passed for all six cases. No large
rank computation or CUDA execution has yet been performed for this extraction.

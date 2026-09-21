# Fixed paper inputs

`g20.json` through `g30.json` are the six original input files, copied without reformatting. Their byte hashes, fixed numerical settings, matrix dimensions and expected outcomes are in `cases.json`. Input paths there are relative to this directory.

`recorded-results.json` contains the selected completed original rank results and recorded deformation residual summaries. Historical server paths are provenance only, not paths required to run this repository. See [RESULTS.md](../RESULTS.md) for interpretation and timings.

Some original input `status` flags are false: these immutable files were written before the expensive computation. Completed outcomes are recorded separately. Do not change those flags or the original files to claim completion.

The genus-20 input uses the paired deformation. The older alternating experiment, random-point inputs and unused genera are deliberately excluded.

Large WDM sequences, generator matrices, dense correction/obstruction vectors and historical logs are not included. The repository contains about 1.5 MB of small input/evidence data.

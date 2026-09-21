# Recorded paper computations

This records the completed original computations, not a claim that the new ancillary CUDA build has been validated. The selected manifests and actual rank-result files on ada-31 and ada-32 were inspected on 21 September 2026. [cases.json](data/cases.json) fixes the inputs and expected outcomes; [recorded-results.json](data/recorded-results.json) preserves the per-run ranks and original result paths.

Put m = g/2. All cases use the two node orbits (zeta^j, 2 zeta^j) and (4 zeta^j, 10 zeta^j), 0 ≤ j < m, with all Prym signs −1. At t = 0 the pencil sections have characters 1 and m−1, and the elimination section has character 2.

| g | p | zeta | Sector columns N | Base sector 0 rank | Other base ranks | Quadratic replacement rank |
|---|---:|---:|---:|---:|---:|---:|
| 20 | 661 | 190 | 19,448 | 19,446 | 19,448 | 19,448 |
| 22 | 661 | 9 | 75,582 | 75,582 | 75,582 | — |
| 24 | 1009 | 160 | 293,930 | 293,928 | 293,930 | 293,930 |
| 26 | 1093 | 11 | 1,144,066 | 1,144,066 | 1,144,066 | — |
| 28 | 1009 | 74 | 4,457,400 | 4,457,398 | 4,457,400 | 4,457,400 |
| 30 | 1051 | 136 | 17,383,860 | 17,383,860 | 17,383,860 | — |

For g = 20, 24, 28 the paired deformation moves the first orbit by (t zeta^(2j), t) and fixes the second. The Taylor convention is the coefficient of t^k, not the kth derivative. The recorded sector-zero kernel has dimension 2 and zero residual. First-order corrections occur in sectors 1 and m−1, each with zero residual. Their augmented matrices have N+2 columns and rank N. The quadratic replacement drops columns 0 and 1 and inserts two obstruction columns; its full rank N completes the recorded deformation test. These residual statements summarize saved reports; the large vectors are not included in this Git repository.

All rank computations used block width 4, numerical seed 1, two row-mixing rounds, row-mixing seed 10001, and `paired-sl2-target-v1`. CPU rank jobs used 32 threads. The full arrays of ordinary and eliminated sector dimensions are in `cases.json`; even-m sector row counts depend on sector parity.

## Historical timings

The campaigns used two NVIDIA RTX PRO 6000 Blackwell Server Edition GPUs. The single-run column is the median base-sector CUDA sequence-loop time, excluding setup, output compression and CPU rank recovery. Elapsed time is from the first recorded base task to the last rank completion, including waiting and overlapping CPU work, but excluding earlier fixture preparation.

| g | GPU passes | Median base sequence (s) | Campaign elapsed |
|---|---:|---:|---:|
| 20 | 16 | 0.3500 | 61 s |
| 22 | 11 | 1.9830 | 125 s |
| 24 | 18 | 22.2715 | 449 s |
| 26 | 13 | 339.8760 | 2,941 s |
| 28 | 20 | 5,416.3790 | 57,077 s (15.9 h) |
| 30 | 15 | 113,939.5950 | 919,735 s (10.6 d) |

The pass count is m for odd m and m+6 for even m. The additional six are two augmented sequences, one replacement sequence and three kernel-recovery passes; there are m+3 sequence/rank computations in the latter case. Timings describe the original hardware and software, not a performance guarantee for this repository.

## Selected evidence and provenance

Paths below are relative to `/scratch/userdata/wilthoma/prymgreene_data/` on the indicated host.

| g | Host | Selected directory |
|---|---|---|
| 20 | ada-32 | `paper_uniform_20260921_103216/g20-paired` |
| 22 | ada-32 | `paper_uniform_20260921_103216/g22` |
| 24 | ada-31 | `deformation_timing_g24_20260907_100829/g24` |
| 26 | ada-31 | `paper_uniform_20260921_104029/g26v4` |
| 28 | ada-31 | `deformation_g28_20260907_102247/g28` |
| 30 | ada-32 | `cyclic_profile_g30_20260907_113440` |

The full input files are original bytes, with SHA256 hashes in `cases.json`. In particular, g20 is the completed paired-deformation input, not the older alternating fixture; g26 is the block-width-4 cyclic campaign, not the earlier ordinary or block-width-16 runs.

Historical executable hashes and source revisions were not consistently recorded, so no current checkout is presented as the exact code used for every original run. The three deformation inputs do record generator SHA256 `240d66f41fa6b00cdd4995ac36f07cb53e7cee36c6c6fef92cac73954c826321`, which matched the original `prym_deformation_setup.py` inspected during extraction. Changes to that generator for this ancillary repository are separate from this historical hash.

## What the small checker verifies

`python3 scripts/check_geometry.py` checks original-file hashes, the frozen point/deformation recipe, section characters, regular pencils, and the small identities mu_w R = I and mu_w B = 0 (through t² for deformation inputs). It independently reconstructs section constraints and verifies the quadratic multiplication rank 3g−3 and absence of basepoints using exact finite-field arithmetic. It does not read historical rank summaries as proof of the large Koszul ranks; those are reproduced by the CUDA pipeline.

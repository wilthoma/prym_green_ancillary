# Source attribution and licenses

The Prym–Green implementation was extracted and simplified from
[wilthoma/prymgreene](https://github.com/wilthoma/prymgreene), whose Cargo
manifest declares `MIT OR Apache-2.0`. The source checkout inspected for this
extraction was `eb70fbf55dd3e9fa29b4397d2617086cff490e2b`. This identifies the
extraction source, not the exact executable revision of every historical
experiment. The original project code and ancillary additions are available
under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at your option.

The following components retain their own licenses:

| Component | Origin | License and notice |
|---|---|---|
| BCW CPU rank and generator recovery code in `third_party/bcw/` | [wilthoma/bcw_rank](https://github.com/wilthoma/bcw_rank), extraction checkout `df34d858e9704146befc1db856e8141c7fc70dd4` | [MIT; copyright 2026 skip64](LICENSE-BCW), also retained in `third_party/bcw/LICENSE` |
| CLI11 2.5.0 single header | [CLIUtils/CLI11](https://github.com/CLIUtils/CLI11), `cuda/include/CLI11.hpp` | [BSD-3-Clause; copyright 2017–2025 University of Cincinnati](LICENSE-CLI11); the original notice is also retained in the header |
| Polynomial multiplication adapted within the BCW code | [bubblemath linear_recurrence.rs](https://github.com/Bubbler-4/math-rs/blob/main/bubblemath/src/linear_recurrence.rs), by Bubbler-4 | [MIT terms and attribution](LICENSE-BUBBLEMATH); author and license are declared in the distributed `bubblemath` 0.1.2 Cargo manifest |

The `bubblemath` crate distribution inspected locally has no separate license
file; its explicit MIT declaration and source attribution are preserved here.
No upstream copyright year has been guessed. The BCW source's attribution
comment is retained alongside the adaptation.

Rust dependencies are resolved by the committed Cargo lockfiles and retain
their respective upstream licenses. NumPy, zstd and the CUDA toolkit are
external build/runtime dependencies; their binaries are not redistributed by
this source repository. The independent geometry checker uses only Python's
standard library.

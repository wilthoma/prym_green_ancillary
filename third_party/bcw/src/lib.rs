//! Exact finite-field order-basis computation for symmetric block Wiedemann sequences.
//!
//! This is the production subset of bcw_rank; the CUDA producer must finish its
//! WDM file before the `prym-rank` executable is started.

pub mod modular_linalg;
pub mod ntt;
pub mod poly_mat_mul;
pub mod rank_pipeline;
pub mod sigma_basis;
pub mod wdm_files;

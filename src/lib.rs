//! Exact CPU algebra supporting the paper's cyclic CUDA computations.
//!
//! [`cyclic`] builds/verifies section spaces and the eliminated character
//! sectors; [`deformation`] performs the three finite-order obstruction stages.
//! These modules never replace the production rank calculation with a dense
//! large-matrix calculation. The repository's `reproduce` script coordinates
//! the fixed cases, CUDA, this crate, and the vendored BCW rank solver.

pub mod artinian;
pub mod binary_form;
pub mod cyclic;
pub mod deformation;
pub mod field;
pub mod linear;
pub mod points;
pub mod sections;
pub mod sha256;
pub mod subsets;

pub type Result<T> = std::result::Result<T, String>;

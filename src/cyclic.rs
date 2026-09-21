//! Cyclic section spaces, Artinian multiplication, and third-section elimination.
//! Homogeneous bases carry character weights modulo the cyclic order. The tensor
//! mu[i,beta,a] stores multiplication with a varying fastest. The elimination
//! data include a right inverse and a three-column kernel inclusion for mu_w.
//! Large ranks are computed by CUDA/BCW; this module constructs and checks the
//! small tensors and the dimensions of the character sectors.

// Explicit indices and argument lists follow the mathematical matrix identities.
#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeSet,
    fs,
    path::{Path, PathBuf},
};

use crate::{
    Result,
    artinian::MuData,
    binary_form::{BinaryForm, has_common_projective_zero},
    field::{Field, is_prime},
    linear::{ColumnSolver, nullspace, rank_columns},
    points::{PointPair, ProjectivePoint},
    sections::constraint_matrix,
    sha256::sha256_hex,
    subsets::{binom, next_subset},
};

pub const CYCLIC_INSTANCE_FORMAT: &str = "prym-cyclic-instance-v1";
pub const CYCLIC_GENERATOR_VERSION: u32 = 1;

#[derive(Clone, Debug)]
pub struct CyclicGenerateOptions {
    pub genus: usize,
    pub order: usize,
    pub prime: u64,
    pub zeta: u64,
    pub orbit_reps: Vec<[u64; 2]>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CyclicInstanceFile {
    pub fixture_format: String,
    #[serde(default)]
    pub generator_version: Option<u32>,
    pub genus: usize,
    pub modulus: u64,
    pub status: CyclicStatus,
    pub description: String,
    pub cyclic: CyclicConfig,
    pub polynomial_convention: String,
    pub pencil: CyclicPencil,
    pub bases: CyclicBases,
    pub weights: CyclicWeights,
    pub artinian: CyclicArtinian,
    pub koszul: CyclicKoszul,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub third_section_elimination: Option<CyclicElimination>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub verification: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub checksum_algorithm: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub checksum_scope: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub checksum_sha256: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CyclicStatus {
    pub small_tensor_verified: bool,
    pub koszul_rank_verified: bool,
    pub cuda_benchmarked: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CyclicConfig {
    pub order: usize,
    pub zeta: u64,
    pub node_orbit_representatives: Vec<[u64; 2]>,
    pub points: Vec<[u64; 2]>,
    pub eta_signs: Vec<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub geometry_seed: Option<u64>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CyclicPencil {
    pub u: Vec<u64>,
    pub v: Vec<u64>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CyclicBases {
    #[serde(rename = "V")]
    pub v: Vec<Vec<u64>>,
    #[serde(rename = "A0")]
    pub a0: Vec<Vec<u64>>,
    #[serde(rename = "A1")]
    pub a1: Vec<Vec<u64>>,
    #[serde(default, rename = "W", skip_serializing_if = "Option::is_none")]
    pub w_full: Option<Vec<Vec<u64>>>,
    #[serde(default, rename = "M1", skip_serializing_if = "Option::is_none")]
    pub m1_full: Option<Vec<Vec<u64>>>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CyclicWeights {
    #[serde(rename = "V")]
    pub v: Vec<usize>,
    #[serde(rename = "A0")]
    pub a0: Vec<usize>,
    #[serde(rename = "A1")]
    pub a1: Vec<usize>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CyclicArtinian {
    pub mu: CyclicMuData,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CyclicMuData {
    pub n: usize,
    pub g: usize,
    #[serde(default = "default_mu_layout")]
    pub layout: String,
    pub values: Vec<u64>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CyclicKoszul {
    pub exterior_degree: usize,
    pub columns: u64,
    pub rows: u64,
    pub nnz: u64,
    pub sector_columns: Vec<u64>,
    pub sector_rows: Vec<u64>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub sector_nnz: Vec<u64>,
    pub sector_label: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CyclicElimination {
    #[serde(rename = "w_V_index")]
    pub w_v_index: usize,
    pub w_polynomial: Vec<u64>,
    pub w_weight: usize,
    pub mu_w_rank: usize,
    #[serde(rename = "right_inverse_A0_by_A1")]
    pub right_inverse_a0_by_a1: Vec<Vec<u64>>,
    #[serde(rename = "kernel_basis_A0_by_3")]
    pub kernel_basis_a0_by_3: Vec<Vec<u64>>,
    pub kernel_weights: Vec<usize>,
    pub source: String,
    pub operator: String,
    pub sector_convention: String,
    pub sector_columns: Vec<u64>,
    pub sector_rows: Vec<u64>,
}

#[derive(Clone, Debug)]
struct WeightedBasis {
    vectors: Vec<Vec<u64>>,
    weights: Vec<usize>,
    groups: Vec<Vec<Vec<u64>>>,
}

#[derive(Clone, Debug)]
struct QuotientData {
    denominator_columns: Vec<Vec<u64>>,
    quotient_basis: Vec<Vec<u64>>,
    quotient_weights: Vec<usize>,
    coordinate_columns: Vec<Vec<u64>>,
    coordinate_solver: ColumnSolver,
}

#[derive(Clone, Debug)]
struct WChoice {
    polynomial: Vec<u64>,
    weight: usize,
}

fn default_mu_layout() -> String {
    "i-beta-a with a varying fastest".to_string()
}

impl CyclicInstanceFile {
    pub fn canonical_hash(&self) -> Result<String> {
        let mut clone = self.clone();
        clone.checksum_algorithm = None;
        clone.checksum_scope = None;
        clone.checksum_sha256 = None;
        let bytes = serde_json::to_vec(&clone).map_err(|e| e.to_string())?;
        Ok(sha256_hex(&bytes))
    }

    pub fn attach_checksum(mut self) -> Result<Self> {
        let hash = self.canonical_hash()?;
        self.checksum_algorithm = Some("SHA-256".to_string());
        self.checksum_scope =
            Some("canonical serde_json bytes with checksum metadata omitted".to_string());
        self.checksum_sha256 = Some(hash);
        Ok(self)
    }

    pub fn verify_checksum_if_present(&self) -> Result<()> {
        if let Some(stored) = &self.checksum_sha256 {
            let expected = self.canonical_hash()?;
            if stored != &expected {
                return Err(format!(
                    "cyclic checksum mismatch: expected {expected}, file contains {stored}"
                ));
            }
        }
        Ok(())
    }

    pub fn mu_data(&self) -> MuData {
        MuData {
            n: self.artinian.mu.n,
            g: self.artinian.mu.g,
            layout: self.artinian.mu.layout.clone(),
            values: self.artinian.mu.values.clone(),
        }
    }
}

pub fn read_cyclic_instance(path: &Path) -> Result<CyclicInstanceFile> {
    let bytes = fs::read(path).map_err(|e| format!("failed to read {}: {e}", path.display()))?;
    let instance: CyclicInstanceFile = serde_json::from_slice(&bytes)
        .map_err(|e| format!("failed to parse {}: {e}", path.display()))?;
    instance.verify_checksum_if_present()?;
    Ok(instance)
}

pub fn write_cyclic_instance_dir(instance: &CyclicInstanceFile, out_dir: &Path) -> Result<PathBuf> {
    fs::create_dir_all(out_dir)
        .map_err(|e| format!("failed to create {}: {e}", out_dir.display()))?;
    let path = out_dir.join("cyclic-instance.json");
    write_cyclic_instance_file(instance, &path)?;
    Ok(path)
}

pub fn write_cyclic_instance_file(instance: &CyclicInstanceFile, path: &Path) -> Result<()> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)
            .map_err(|e| format!("failed to create {}: {e}", parent.display()))?;
    }
    let bytes = serde_json::to_vec_pretty(instance).map_err(|e| e.to_string())?;
    fs::write(path, bytes).map_err(|e| format!("failed to write {}: {e}", path.display()))
}

pub fn generate_cyclic_instance(options: &CyclicGenerateOptions) -> Result<CyclicInstanceFile> {
    validate_cyclic_parameters(options.genus, options.order, options.prime, options.zeta)?;
    let field = Field::new(options.prime)?;
    let s = options.genus / options.order;
    let reps = options.orbit_reps.clone();
    if reps.len() != s {
        return Err(format!(
            "expected {s} representative pairs, got {}",
            reps.len()
        ));
    }
    validate_representatives(&reps, options.order, field)?;
    let point_pairs = cyclic_point_pairs(&reps, options.order, options.zeta, field)?;
    let points = point_pairs
        .iter()
        .map(|pair| [pair.p.x1, pair.q.x1])
        .collect::<Vec<_>>();

    let d = 2 * options.genus - 2;
    let big_degree = 2 * d;
    let endpoints = point_pairs
        .iter()
        .flat_map(|pair| [pair.p.x1, pair.q.x1])
        .collect::<Vec<_>>();
    let node_lambdas = gluing_lambdas(&point_pairs, &endpoints, field)?;
    let canonical_multipliers = node_lambdas
        .iter()
        .map(|node| [node.canonical_b, 1])
        .collect::<Vec<_>>();
    let prym_multipliers = node_lambdas
        .iter()
        .map(|node| [node.alpha, 1])
        .collect::<Vec<_>>();
    let m1_multipliers = node_lambdas
        .iter()
        .map(|node| [field.mul(node.canonical_b, node.alpha), 1])
        .collect::<Vec<_>>();

    let a0_constraints = constraint_matrix(&point_pairs, &canonical_multipliers, d, field);
    let w_constraints = constraint_matrix(&point_pairs, &prym_multipliers, d, field);
    let m1_constraints = constraint_matrix(&point_pairs, &m1_multipliers, big_degree, field);

    let w_full = homogeneous_basis(&w_constraints, d, options.order, 1, field)?;
    let a0 = homogeneous_basis(&a0_constraints, d, options.order, 1, field)?;
    let m1 = homogeneous_basis(&m1_constraints, big_degree, options.order, 2, field)?;

    expect_total_dim("W=H0(L)", &w_full, options.genus - 1)?;
    expect_total_dim("A0=H0(omega)", &a0, options.genus)?;
    expect_total_dim("M1=H0(omega tensor L)", &m1, 3 * options.genus - 3)?;
    verify_all_minus_multiplicities("A0", &a0.weights, options.order, s, "a0")?;

    let (u, v, w_complement) = choose_regular_pencil(&w_full, options.order, field, d)?;
    let a0_forms = a0
        .vectors
        .iter()
        .map(|coeffs| BinaryForm::new(coeffs.clone(), field))
        .collect::<Result<Vec<_>>>()?;
    let quotient = build_a1_quotient(options.genus, options.order, &a0, &m1, &u, &v, field)?;
    let w_choice = choose_elimination_direction(
        options.order,
        &w_complement,
        &a0_forms,
        &quotient,
        field,
        d,
        if options.order == options.genus / 2 {
            Some(2 % options.order)
        } else {
            None
        },
    )?;
    let v_weighted = rebuild_v_with_w_first(&w_complement, &w_choice, field)?;
    verify_all_minus_multiplicities("V", &v_weighted.weights, options.order, s, "v")?;
    let mu = compute_mu_for_v_basis(
        options.genus,
        &v_weighted.vectors,
        &a0.vectors,
        &quotient,
        field,
    )?;
    verify_weight_compatibility(
        &mu,
        &v_weighted.weights,
        &a0.weights,
        &quotient.quotient_weights,
        options.order,
    )?;
    verify_all_minus_multiplicities("A1", &quotient.quotient_weights, options.order, s, "v")?;
    let elimination = build_elimination(
        options.genus,
        options.order,
        &mu,
        &v_weighted,
        &a0,
        &quotient,
        &w_choice,
        field,
    )?;
    let koszul = build_koszul_metadata(
        options.genus,
        options.order,
        &mu,
        &v_weighted.weights,
        &a0.weights,
        &quotient.quotient_weights,
    )?;

    let cyclic = CyclicConfig {
        order: options.order,
        zeta: options.zeta % options.prime,
        node_orbit_representatives: reps,
        points,
        eta_signs: vec![-1; options.genus],
        geometry_seed: None,
    };

    let instance = CyclicInstanceFile {
        fixture_format: CYCLIC_INSTANCE_FORMAT.to_string(),
        generator_version: Some(CYCLIC_GENERATOR_VERSION),
        genus: options.genus,
        modulus: options.prime,
        status: CyclicStatus {
            small_tensor_verified: true,
            koszul_rank_verified: false,
            cuda_benchmarked: false,
        },
        description: format!(
            "Generated cyclic Prym-Green instance: g={}, r={}, p={}, zeta={}, reps={:?}",
            options.genus, options.order, options.prime, options.zeta, cyclic.node_orbit_representatives
        ),
        cyclic,
        polynomial_convention:
            "ascending affine z coefficients; coefficient e is z^e with natural characters exponent+1 for W/A0 and exponent+2 for M1".to_string(),
        pencil: CyclicPencil {
            u: u.coeffs.clone(),
            v: v.coeffs.clone(),
        },
        bases: CyclicBases {
            v: v_weighted.vectors.clone(),
            a0: a0.vectors.clone(),
            a1: quotient.quotient_basis.clone(),
            w_full: Some(w_full.vectors.clone()),
            m1_full: Some(m1.vectors.clone()),
        },
        weights: CyclicWeights {
            v: v_weighted.weights.clone(),
            a0: a0.weights.clone(),
            a1: quotient.quotient_weights.clone(),
        },
        artinian: CyclicArtinian {
            mu: CyclicMuData {
                n: mu.n,
                g: mu.g,
                layout: mu.layout,
                values: mu.values,
            },
        },
        koszul,
        third_section_elimination: Some(elimination),
        verification: vec![
            format!("{} endpoints distinct", 2 * options.genus),
            "homogeneous pencil coprime".to_string(),
            "all products reconstructed in quotient".to_string(),
            "cyclic weight compatibility".to_string(),
            "mu_w R=I".to_string(),
            "mu_w B=0".to_string(),
            "sector dimensions and exact nnz counted".to_string(),
        ],
        checksum_algorithm: None,
        checksum_scope: None,
        checksum_sha256: None,
    };
    instance.attach_checksum()
}

pub fn verify_cyclic_instance(instance: &CyclicInstanceFile) -> Result<CyclicVerificationReport> {
    instance.verify_checksum_if_present()?;
    validate_cyclic_parameters(
        instance.genus,
        instance.cyclic.order,
        instance.modulus,
        instance.cyclic.zeta,
    )?;
    let field = Field::new(instance.modulus)?;
    let s = instance.genus / instance.cyclic.order;
    if instance.cyclic.node_orbit_representatives.len() != s {
        return Err(format!(
            "expected {s} representative pairs, got {}",
            instance.cyclic.node_orbit_representatives.len()
        ));
    }
    validate_representatives(
        &instance.cyclic.node_orbit_representatives,
        instance.cyclic.order,
        field,
    )?;
    let point_pairs = cyclic_point_pairs(
        &instance.cyclic.node_orbit_representatives,
        instance.cyclic.order,
        instance.cyclic.zeta,
        field,
    )?;
    let expected_points = point_pairs
        .iter()
        .map(|pair| [pair.p.x1, pair.q.x1])
        .collect::<Vec<_>>();
    if expected_points != instance.cyclic.points {
        return Err("stored cyclic points do not match representatives and zeta".to_string());
    }
    if instance.cyclic.eta_signs != vec![-1; instance.genus] {
        return Err("only all-minus torsion signs are supported by verify-cyclic".to_string());
    }

    let g = instance.genus;
    let n = g - 3;
    let r = instance.cyclic.order;
    let d = 2 * g - 2;
    let big_degree = 2 * d;
    check_basis_shape("V", &instance.bases.v, n, d + 1, instance.modulus)?;
    check_basis_shape("A0", &instance.bases.a0, g, d + 1, instance.modulus)?;
    check_basis_shape(
        "A1",
        &instance.bases.a1,
        n,
        big_degree + 1,
        instance.modulus,
    )?;
    check_weight_vector("V", &instance.weights.v, n, r)?;
    check_weight_vector("A0", &instance.weights.a0, g, r)?;
    check_weight_vector("A1", &instance.weights.a1, n, r)?;
    verify_homogeneous_basis("V", &instance.bases.v, &instance.weights.v, r, 1)?;
    verify_homogeneous_basis("A0", &instance.bases.a0, &instance.weights.a0, r, 1)?;
    verify_homogeneous_basis("A1", &instance.bases.a1, &instance.weights.a1, r, 2)?;

    let endpoints = point_pairs
        .iter()
        .flat_map(|pair| [pair.p.x1, pair.q.x1])
        .collect::<Vec<_>>();
    let node_lambdas = gluing_lambdas(&point_pairs, &endpoints, field)?;
    let canonical_multipliers = node_lambdas
        .iter()
        .map(|node| [node.canonical_b, 1])
        .collect::<Vec<_>>();
    let prym_multipliers = node_lambdas
        .iter()
        .map(|node| [node.alpha, 1])
        .collect::<Vec<_>>();
    let m1_multipliers = node_lambdas
        .iter()
        .map(|node| [field.mul(node.canonical_b, node.alpha), 1])
        .collect::<Vec<_>>();
    let a0_constraints = constraint_matrix(&point_pairs, &canonical_multipliers, d, field);
    let w_constraints = constraint_matrix(&point_pairs, &prym_multipliers, d, field);
    let m1_constraints = constraint_matrix(&point_pairs, &m1_multipliers, big_degree, field);
    ensure_all_satisfy("A0 basis", &a0_constraints, &instance.bases.a0, field)?;
    ensure_all_satisfy("V basis", &w_constraints, &instance.bases.v, field)?;
    ensure_satisfies("pencil u", &w_constraints, &instance.pencil.u, field)?;
    ensure_satisfies("pencil v", &w_constraints, &instance.pencil.v, field)?;
    ensure_all_satisfy("A1 basis", &m1_constraints, &instance.bases.a1, field)?;
    let u = BinaryForm::new(instance.pencil.u.clone(), field)?;
    let v = BinaryForm::new(instance.pencil.v.clone(), field)?;
    if has_common_projective_zero(&u, &v, field) {
        return Err("stored pencil has a common projective zero".to_string());
    }

    let quotient =
        quotient_from_stored_bases(g, &instance.bases.a0, &instance.bases.a1, &u, &v, field)?;
    let mu = compute_mu_for_v_basis(g, &instance.bases.v, &instance.bases.a0, &quotient, field)?;
    if mu.values != instance.artinian.mu.values {
        return Err(
            "stored mu tensor does not match products reconstructed from bases".to_string(),
        );
    }
    verify_weight_compatibility(
        &mu,
        &instance.weights.v,
        &instance.weights.a0,
        &instance.weights.a1,
        r,
    )?;
    let expected_koszul = build_koszul_metadata(
        g,
        r,
        &mu,
        &instance.weights.v,
        &instance.weights.a0,
        &instance.weights.a1,
    )?;
    if expected_koszul.columns != instance.koszul.columns
        || expected_koszul.rows != instance.koszul.rows
        || expected_koszul.nnz != instance.koszul.nnz
        || expected_koszul.sector_columns != instance.koszul.sector_columns
        || expected_koszul.sector_rows != instance.koszul.sector_rows
        || (!instance.koszul.sector_nnz.is_empty()
            && expected_koszul.sector_nnz != instance.koszul.sector_nnz)
    {
        return Err("stored Koszul metadata does not match recomputed cyclic metadata".to_string());
    }

    if let Some(elim) = &instance.third_section_elimination {
        verify_elimination_data(g, r, &mu, &instance.weights, elim, field)?;
        let expected_elim = eliminated_sector_counts(
            g,
            r,
            &instance.weights.v,
            &instance.weights.a0,
            &instance.weights.a1,
            elim.w_v_index,
            elim.w_weight,
            &elim.kernel_weights,
        )?;
        if expected_elim.0 != elim.sector_columns || expected_elim.1 != elim.sector_rows {
            return Err(
                "stored eliminated sector dimensions do not match recomputed metadata".to_string(),
            );
        }
    }

    Ok(CyclicVerificationReport {
        checksum_sha256: instance
            .checksum_sha256
            .clone()
            .unwrap_or_else(|| "not-present".to_string()),
        genus: g,
        order: r,
        prime: instance.modulus,
        ordinary_rows: instance.koszul.rows,
        ordinary_columns: instance.koszul.columns,
        ordinary_nnz: instance.koszul.nnz,
        eliminated_columns: instance
            .third_section_elimination
            .as_ref()
            .map(|e| e.sector_columns.iter().sum()),
        eliminated_rows: instance
            .third_section_elimination
            .as_ref()
            .map(|e| e.sector_rows.iter().sum()),
    })
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CyclicVerificationReport {
    pub checksum_sha256: String,
    pub genus: usize,
    pub order: usize,
    pub prime: u64,
    pub ordinary_rows: u64,
    pub ordinary_columns: u64,
    pub ordinary_nnz: u64,
    pub eliminated_columns: Option<u64>,
    pub eliminated_rows: Option<u64>,
}

#[derive(Clone, Copy, Debug)]
struct NodeLambda {
    canonical_b: u64,
    alpha: u64,
}

fn validate_cyclic_parameters(g: usize, r: usize, p: u64, zeta: u64) -> Result<()> {
    if g < 6 || !g.is_multiple_of(2) {
        return Err(format!("cyclic genus must be even and at least 6, got {g}"));
    }
    if r < 3 || !g.is_multiple_of(r) {
        return Err(format!(
            "cyclic order {r} must be at least 3 and divide genus {g}"
        ));
    }
    if !is_prime(p) || p == 2 {
        return Err(format!("prime {p} is not an admissible odd prime"));
    }
    if !(p - 1).is_multiple_of(r as u64) {
        return Err(format!("order {r} does not divide p-1 for p={p}"));
    }
    let s = g / r;
    if (p - 1) / (r as u64) < 2 * s as u64 {
        return Err(format!(
            "p={p}, r={r} has only {} r-th power cosets; need at least {}",
            (p - 1) / r as u64,
            2 * s
        ));
    }
    let field = Field::new(p)?;
    if zeta.is_multiple_of(p) || field.pow(zeta % p, r as u64) != 1 {
        return Err(format!(
            "zeta={zeta} does not have order dividing {r} modulo {p}"
        ));
    }
    for k in 1..r {
        if field.pow(zeta % p, k as u64) == 1 {
            return Err(format!("zeta={zeta} has order {k}, not exact order {r}"));
        }
    }
    Ok(())
}

fn validate_representatives(reps: &[[u64; 2]], r: usize, field: Field) -> Result<()> {
    let mut seen = BTreeSet::new();
    for (pair_idx, pair) in reps.iter().enumerate() {
        for (side, value) in pair.iter().copied().enumerate() {
            let reduced = value % field.modulus();
            if reduced == 0 {
                return Err(format!(
                    "representative pair {pair_idx} side {side} is zero"
                ));
            }
            let power = field.pow(reduced, r as u64);
            if !seen.insert(power) {
                return Err(format!(
                    "representative pair {pair_idx} side {side} repeats r-th power {power}"
                ));
            }
        }
    }
    Ok(())
}

fn cyclic_point_pairs(
    reps: &[[u64; 2]],
    r: usize,
    zeta: u64,
    field: Field,
) -> Result<Vec<PointPair>> {
    let mut pairs = Vec::with_capacity(reps.len() * r);
    let mut seen = BTreeSet::new();
    for pair in reps {
        let mut zeta_power = 1;
        for _ in 0..r {
            let p = field.mul(pair[0] % field.modulus(), zeta_power);
            let q = field.mul(pair[1] % field.modulus(), zeta_power);
            if !seen.insert(p) || !seen.insert(q) {
                return Err("cyclic endpoint orbits are not pairwise distinct".to_string());
            }
            pairs.push(PointPair {
                p: ProjectivePoint::affine(p, field),
                q: ProjectivePoint::affine(q, field),
            });
            zeta_power = field.mul(zeta_power, zeta % field.modulus());
        }
    }
    Ok(pairs)
}

fn gluing_lambdas(
    point_pairs: &[PointPair],
    endpoints: &[u64],
    field: Field,
) -> Result<Vec<NodeLambda>> {
    let d_poly = poly_from_roots(endpoints, field);
    let derivative = derivative(&d_poly, field);
    let mut out = Vec::with_capacity(point_pairs.len());
    for (idx, pair) in point_pairs.iter().enumerate() {
        let dp = poly_eval(&derivative, pair.p.x1, field);
        let dq = poly_eval(&derivative, pair.q.x1, field);
        if dp == 0 || dq == 0 {
            return Err(format!(
                "D' vanished at node {idx}; endpoints are not distinct"
            ));
        }
        let canonical_b = field.neg(field.div(dp, dq)?);
        out.push(NodeLambda {
            canonical_b,
            alpha: field.neg(canonical_b),
        });
    }
    Ok(out)
}

fn poly_from_roots(roots: &[u64], field: Field) -> Vec<u64> {
    let mut poly = vec![1];
    for root in roots {
        let mut next = vec![0; poly.len() + 1];
        for (i, coeff) in poly.iter().copied().enumerate() {
            next[i] = field.sub(next[i], field.mul(coeff, *root));
            next[i + 1] = field.add(next[i + 1], coeff);
        }
        poly = next;
    }
    poly
}

fn derivative(poly: &[u64], field: Field) -> Vec<u64> {
    if poly.len() <= 1 {
        return vec![0];
    }
    (1..poly.len())
        .map(|i| field.mul(poly[i], i as u64))
        .collect()
}

fn poly_eval(poly: &[u64], x: u64, field: Field) -> u64 {
    let mut value = 0;
    for coeff in poly.iter().rev().copied() {
        value = field.add(field.mul(value, x), coeff);
    }
    value
}

fn homogeneous_basis(
    constraints: &[Vec<u64>],
    degree: usize,
    r: usize,
    shift: usize,
    field: Field,
) -> Result<WeightedBasis> {
    let mut groups = vec![Vec::new(); r];
    let mut vectors = Vec::new();
    let mut weights = Vec::new();
    for ch in 0..r {
        let monomial_indices = (0..=degree)
            .filter(|&exp| (exp + shift) % r == ch)
            .collect::<Vec<_>>();
        let restricted = constraints
            .iter()
            .map(|row| {
                monomial_indices
                    .iter()
                    .map(|&idx| row[idx])
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let local_basis = nullspace(restricted, field)?;
        for local in local_basis {
            let mut full = vec![0; degree + 1];
            for (&idx, coeff) in monomial_indices.iter().zip(local) {
                full[idx] = coeff;
            }
            groups[ch].push(full.clone());
            vectors.push(full);
            weights.push(ch);
        }
    }
    Ok(WeightedBasis {
        vectors,
        weights,
        groups,
    })
}

fn expect_total_dim(label: &str, basis: &WeightedBasis, expected: usize) -> Result<()> {
    if basis.vectors.len() != expected {
        return Err(format!(
            "{label} has dimension {}, expected {expected}",
            basis.vectors.len()
        ));
    }
    Ok(())
}

fn verify_all_minus_multiplicities(
    label: &str,
    weights: &[usize],
    r: usize,
    s: usize,
    kind: &str,
) -> Result<()> {
    let counts = weight_counts(weights, r)?;
    for (ch, &count) in counts.iter().enumerate() {
        let expected = if kind == "a0" {
            s
        } else if ch == 0 || ch == 1 || ch + 1 == r {
            s - 1
        } else {
            s
        };
        if count != expected as u64 {
            return Err(format!(
                "{label} character {ch} has multiplicity {count}, expected {expected}"
            ));
        }
    }
    Ok(())
}

fn choose_regular_pencil(
    w_full: &WeightedBasis,
    r: usize,
    field: Field,
    degree: usize,
) -> Result<(BinaryForm, BinaryForm, WeightedBasis)> {
    let u_ch = 1 % r;
    let v_ch = (r - 1) % r;
    let u_candidates = candidate_vectors(&w_full.groups[u_ch], field);
    let v_candidates = candidate_vectors(&w_full.groups[v_ch], field);
    for u_coeffs in &u_candidates {
        let u = BinaryForm::new(pad_coeffs(u_coeffs, degree + 1), field)?;
        for v_coeffs in &v_candidates {
            let v = BinaryForm::new(pad_coeffs(v_coeffs, degree + 1), field)?;
            if !has_common_projective_zero(&u, &v, field) {
                let mut groups = vec![Vec::new(); r];
                for (ch, group) in w_full.groups.iter().enumerate() {
                    groups[ch] = if ch == u_ch {
                        complement_after_selected(group, std::slice::from_ref(&u.coeffs), field)?
                    } else if ch == v_ch {
                        complement_after_selected(group, std::slice::from_ref(&v.coeffs), field)?
                    } else {
                        group.clone()
                    };
                }
                return Ok((u, v, flatten_groups(groups)));
            }
        }
    }
    Err("failed to find a homogeneous regular pencil in weights 1 and r-1".to_string())
}

fn candidate_vectors(group: &[Vec<u64>], field: Field) -> Vec<Vec<u64>> {
    let mut out = group.to_vec();
    let scale_limit = (field.modulus() - 1).min(128);
    for i in 0..group.len() {
        for j in i + 1..group.len() {
            for scale in 1..=scale_limit {
                out.push(add_scaled_vec(&group[i], scale, &group[j], field));
            }
        }
    }
    out
}

fn add_scaled_vec(a: &[u64], scale: u64, b: &[u64], field: Field) -> Vec<u64> {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| field.add(x, field.mul(scale, y)))
        .collect()
}

fn complement_after_selected(
    group: &[Vec<u64>],
    selected: &[Vec<u64>],
    field: Field,
) -> Result<Vec<Vec<u64>>> {
    let target_dim = group.len();
    let mut spanning = selected.to_vec();
    let mut complement = Vec::new();
    let mut rank = rank_columns(&spanning, field)?;
    for candidate in group {
        let mut trial = spanning.clone();
        trial.push(candidate.clone());
        let trial_rank = rank_columns(&trial, field)?;
        if trial_rank > rank {
            complement.push(candidate.clone());
            spanning.push(candidate.clone());
            rank = trial_rank;
            if rank == target_dim {
                break;
            }
        }
    }
    if rank != target_dim {
        return Err(format!(
            "failed to build homogeneous complement: rank {rank}, expected {target_dim}"
        ));
    }
    Ok(complement)
}

fn flatten_groups(groups: Vec<Vec<Vec<u64>>>) -> WeightedBasis {
    let mut vectors = Vec::new();
    let mut weights = Vec::new();
    for (ch, group) in groups.iter().enumerate() {
        for vector in group {
            vectors.push(vector.clone());
            weights.push(ch);
        }
    }
    WeightedBasis {
        vectors,
        weights,
        groups,
    }
}

fn build_a1_quotient(
    g: usize,
    r: usize,
    a0: &WeightedBasis,
    m1: &WeightedBasis,
    u: &BinaryForm,
    v: &BinaryForm,
    field: Field,
) -> Result<QuotientData> {
    let mut denominator_columns = Vec::with_capacity(2 * g);
    let mut denominator_weights = Vec::with_capacity(2 * g);
    let a0_forms = forms_from_coeffs(&a0.vectors, field)?;
    for (omega, &weight) in a0_forms.iter().zip(&a0.weights) {
        denominator_columns.push(u.mul(omega, field).coeffs);
        denominator_weights.push((1 + weight) % r);
    }
    for (omega, &weight) in a0_forms.iter().zip(&a0.weights) {
        denominator_columns.push(v.mul(omega, field).coeffs);
        denominator_weights.push((r - 1 + weight) % r);
    }
    let d_rank = rank_columns(&denominator_columns, field)?;
    if d_rank != 2 * g {
        return Err(format!("denominator has rank {d_rank}, expected {}", 2 * g));
    }

    let mut quotient_basis = Vec::with_capacity(g - 3);
    let mut quotient_weights = Vec::with_capacity(g - 3);
    let mut coordinate_columns = denominator_columns.clone();
    let mut current_rank = d_rank;
    for ch in 0..r {
        for candidate in &m1.groups[ch] {
            let mut trial = coordinate_columns.clone();
            trial.push(candidate.clone());
            let trial_rank = rank_columns(&trial, field)?;
            if trial_rank > current_rank {
                coordinate_columns.push(candidate.clone());
                quotient_basis.push(candidate.clone());
                quotient_weights.push(ch);
                current_rank = trial_rank;
                if quotient_basis.len() == g - 3 {
                    break;
                }
            }
        }
        if quotient_basis.len() == g - 3 {
            break;
        }
    }
    if quotient_basis.len() != g - 3 {
        return Err(format!(
            "A1 quotient basis has dimension {}, expected {}",
            quotient_basis.len(),
            g - 3
        ));
    }
    if current_rank != 3 * g - 3 {
        return Err(format!(
            "[uA0,vA0,A1] rank {current_rank}, expected {}",
            3 * g - 3
        ));
    }
    let coordinate_solver = ColumnSolver::new(&coordinate_columns, field)?;
    Ok(QuotientData {
        denominator_columns,
        quotient_basis,
        quotient_weights,
        coordinate_columns,
        coordinate_solver,
    })
}

fn quotient_from_stored_bases(
    g: usize,
    a0_basis: &[Vec<u64>],
    a1_basis: &[Vec<u64>],
    u: &BinaryForm,
    v: &BinaryForm,
    field: Field,
) -> Result<QuotientData> {
    let a0_forms = forms_from_coeffs(a0_basis, field)?;
    let mut denominator_columns = Vec::with_capacity(2 * g);
    for omega in &a0_forms {
        denominator_columns.push(u.mul(omega, field).coeffs);
    }
    for omega in &a0_forms {
        denominator_columns.push(v.mul(omega, field).coeffs);
    }
    let d_rank = rank_columns(&denominator_columns, field)?;
    if d_rank != 2 * g {
        return Err(format!(
            "stored denominator has rank {d_rank}, expected {}",
            2 * g
        ));
    }
    let mut coordinate_columns = denominator_columns.clone();
    coordinate_columns.extend(a1_basis.iter().cloned());
    let coordinate_rank = rank_columns(&coordinate_columns, field)?;
    if coordinate_rank != 3 * g - 3 {
        return Err(format!(
            "stored [uA0,vA0,A1] rank {coordinate_rank}, expected {}",
            3 * g - 3
        ));
    }
    let coordinate_solver = ColumnSolver::new(&coordinate_columns, field)?;
    Ok(QuotientData {
        denominator_columns,
        quotient_basis: a1_basis.to_vec(),
        quotient_weights: Vec::new(),
        coordinate_columns,
        coordinate_solver,
    })
}

fn forms_from_coeffs(coeffs: &[Vec<u64>], field: Field) -> Result<Vec<BinaryForm>> {
    coeffs
        .iter()
        .map(|coeffs| BinaryForm::new(coeffs.clone(), field))
        .collect()
}

fn choose_elimination_direction(
    r: usize,
    v_basis: &WeightedBasis,
    a0_forms: &[BinaryForm],
    quotient: &QuotientData,
    field: Field,
    degree: usize,
    preferred_weight: Option<usize>,
) -> Result<WChoice> {
    let mut weights = Vec::new();
    if let Some(ch) = preferred_weight
        && ch < r
    {
        weights.push(ch);
    }
    for ch in 0..r {
        if !weights.contains(&ch) {
            weights.push(ch);
        }
    }
    for ch in weights {
        for candidate in candidate_vectors(&v_basis.groups[ch], field) {
            let candidate = pad_coeffs(&candidate, degree + 1);
            let rank = multiplication_map_rank(&candidate, a0_forms, quotient, field)?;
            if rank == quotient.quotient_basis.len() {
                return Ok(WChoice {
                    polynomial: candidate,
                    weight: ch,
                });
            }
        }
    }
    Err("no tested homogeneous V direction has surjective mu_w".to_string())
}

fn multiplication_map_rank(
    v_coeffs: &[u64],
    a0_forms: &[BinaryForm],
    quotient: &QuotientData,
    field: Field,
) -> Result<usize> {
    let v_form = BinaryForm::new(v_coeffs.to_vec(), field)?;
    let n = quotient.quotient_basis.len();
    let mut columns = Vec::with_capacity(a0_forms.len());
    for omega in a0_forms {
        let rhs = v_form.mul(omega, field).coeffs;
        let coords = quotient.coordinate_solver.solve_and_verify(
            &quotient.coordinate_columns,
            &rhs,
            field,
        )?;
        columns.push(
            coords[quotient.denominator_columns.len()..quotient.denominator_columns.len() + n]
                .to_vec(),
        );
    }
    rank_columns(&columns, field)
}

fn rebuild_v_with_w_first(
    v_basis: &WeightedBasis,
    w_choice: &WChoice,
    field: Field,
) -> Result<WeightedBasis> {
    let r = v_basis.groups.len();
    let mut groups = vec![Vec::new(); r];
    groups[w_choice.weight].push(w_choice.polynomial.clone());
    for ch in 0..r {
        let local = if ch == w_choice.weight {
            complement_after_selected(
                &v_basis.groups[ch],
                std::slice::from_ref(&w_choice.polynomial),
                field,
            )?
        } else {
            v_basis.groups[ch].clone()
        };
        groups[ch].extend(local);
    }
    let mut vectors = vec![w_choice.polynomial.clone()];
    let mut weights = vec![w_choice.weight];
    for ch in 0..r {
        for vector in groups[ch]
            .iter()
            .skip(if ch == w_choice.weight { 1 } else { 0 })
        {
            vectors.push(vector.clone());
            weights.push(ch);
        }
    }
    Ok(WeightedBasis {
        vectors,
        weights,
        groups,
    })
}

fn compute_mu_for_v_basis(
    g: usize,
    v_basis: &[Vec<u64>],
    a0_basis: &[Vec<u64>],
    quotient: &QuotientData,
    field: Field,
) -> Result<MuData> {
    let n = g - 3;
    if v_basis.len() != n || a0_basis.len() != g || quotient.quotient_basis.len() != n {
        return Err("mu basis dimensions do not match genus".to_string());
    }
    let v_forms = forms_from_coeffs(v_basis, field)?;
    let a0_forms = forms_from_coeffs(a0_basis, field)?;
    let mut mu = MuData::new(n, g);
    let offset = quotient.denominator_columns.len();
    for (i, v_form) in v_forms.iter().enumerate() {
        for (a, omega) in a0_forms.iter().enumerate() {
            let rhs = v_form.mul(omega, field).coeffs;
            let coords = quotient.coordinate_solver.solve_and_verify(
                &quotient.coordinate_columns,
                &rhs,
                field,
            )?;
            for beta in 0..n {
                mu.set(i, beta, a, coords[offset + beta]);
            }
        }
    }
    Ok(mu)
}

fn verify_weight_compatibility(
    mu: &MuData,
    v_weights: &[usize],
    a0_weights: &[usize],
    a1_weights: &[usize],
    r: usize,
) -> Result<()> {
    for (i, &v_weight) in v_weights.iter().enumerate() {
        for (beta, &a1_weight) in a1_weights.iter().enumerate() {
            for (a, &a0_weight) in a0_weights.iter().enumerate() {
                if mu.get(i, beta, a) != 0 && a1_weight != (v_weight + a0_weight) % r {
                    return Err(format!("mu weight mismatch at (i,beta,a)=({i},{beta},{a})"));
                }
            }
        }
    }
    Ok(())
}

fn build_elimination(
    g: usize,
    r: usize,
    mu: &MuData,
    v_basis: &WeightedBasis,
    a0: &WeightedBasis,
    quotient: &QuotientData,
    w_choice: &WChoice,
    field: Field,
) -> Result<CyclicElimination> {
    if v_basis.vectors.first() != Some(&w_choice.polynomial) {
        return Err("internal error: elimination direction is not first in V".to_string());
    }
    let n = g - 3;
    let columns = (0..g)
        .map(|a| (0..n).map(|beta| mu.get(0, beta, a)).collect::<Vec<_>>())
        .collect::<Vec<_>>();
    let mu_w_rank = rank_columns(&columns, field)?;
    if mu_w_rank != n {
        return Err(format!("chosen mu_w has rank {mu_w_rank}, expected {n}"));
    }
    let right_inverse = right_inverse_by_character(
        mu,
        &a0.weights,
        &quotient.quotient_weights,
        w_choice.weight,
        r,
        field,
    )?;
    let (kernel_columns, kernel_weights) = kernel_basis_by_character(
        mu,
        &a0.weights,
        &quotient.quotient_weights,
        w_choice.weight,
        r,
        field,
    )?;
    if rank_columns(&kernel_columns, field)? != 3 || kernel_columns.len() != 3 {
        return Err("mu_w kernel basis does not have rank 3".to_string());
    }
    verify_right_inverse_and_kernel(mu, &right_inverse, &kernel_columns, field)?;
    let kernel_basis_a0_by_3 = rows_from_columns_fixed(&kernel_columns, g, 3)?;
    let (sector_columns, sector_rows) = eliminated_sector_counts(
        g,
        r,
        &v_basis.weights,
        &a0.weights,
        &quotient.quotient_weights,
        0,
        w_choice.weight,
        &kernel_weights,
    )?;
    Ok(CyclicElimination {
        w_v_index: 0,
        w_polynomial: w_choice.polynomial.clone(),
        w_weight: w_choice.weight,
        mu_w_rank,
        right_inverse_a0_by_a1: rows_from_columns_fixed(&right_inverse, g, n)?,
        kernel_basis_a0_by_3,
        kernel_weights,
        source: "(Lambda^m U tensor A0) direct_sum (Lambda^(m-1) U tensor ker(mu_w))".to_string(),
        operator:
            "F(x,z)=D_(m-1)(B*z-R*D_m*x); apply as two Koszul sweeps, not as an expanded matrix"
                .to_string(),
        sector_convention:
            "x uses its natural weight; z and target use natural weight plus w_weight".to_string(),
        sector_columns,
        sector_rows,
    })
}

fn right_inverse_by_character(
    mu: &MuData,
    a0_weights: &[usize],
    a1_weights: &[usize],
    w_weight: usize,
    r: usize,
    field: Field,
) -> Result<Vec<Vec<u64>>> {
    let n = mu.n;
    let g = mu.g;
    let mut right_inverse_columns = vec![vec![0; g]; n];
    for output_ch in 0..r {
        let beta_indices = indices_with_weight(a1_weights, output_ch);
        if beta_indices.is_empty() {
            continue;
        }
        let a_ch = (output_ch + r - w_weight % r) % r;
        let a_indices = indices_with_weight(a0_weights, a_ch);
        let block_rows = beta_indices
            .iter()
            .map(|&beta| {
                a_indices
                    .iter()
                    .map(|&a| mu.get(0, beta, a))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        for (local_beta, &beta) in beta_indices.iter().enumerate() {
            let mut rhs = vec![0; beta_indices.len()];
            rhs[local_beta] = 1;
            let solution = solve_linear_system_one(block_rows.clone(), rhs, field)?;
            for (&a, coeff) in a_indices.iter().zip(solution) {
                right_inverse_columns[beta][a] = coeff;
            }
        }
    }
    Ok(right_inverse_columns)
}

fn kernel_basis_by_character(
    mu: &MuData,
    a0_weights: &[usize],
    a1_weights: &[usize],
    w_weight: usize,
    r: usize,
    field: Field,
) -> Result<(Vec<Vec<u64>>, Vec<usize>)> {
    let g = mu.g;
    let mut columns = Vec::new();
    let mut weights = Vec::new();
    for a_ch in 0..r {
        let a_indices = indices_with_weight(a0_weights, a_ch);
        if a_indices.is_empty() {
            continue;
        }
        let output_ch = (a_ch + w_weight) % r;
        let beta_indices = indices_with_weight(a1_weights, output_ch);
        let block_rows = beta_indices
            .iter()
            .map(|&beta| {
                a_indices
                    .iter()
                    .map(|&a| mu.get(0, beta, a))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        for local in nullspace(block_rows, field)? {
            let mut full = vec![0; g];
            for (&a, coeff) in a_indices.iter().zip(local) {
                full[a] = coeff;
            }
            columns.push(full);
            weights.push(a_ch);
        }
    }
    Ok((columns, weights))
}

fn verify_right_inverse_and_kernel(
    mu: &MuData,
    right_inverse_columns: &[Vec<u64>],
    kernel_columns: &[Vec<u64>],
    field: Field,
) -> Result<()> {
    let n = mu.n;
    let g = mu.g;
    if right_inverse_columns.len() != n || right_inverse_columns.iter().any(|c| c.len() != g) {
        return Err("right inverse has wrong shape".to_string());
    }
    for beta in 0..n {
        for gamma in 0..n {
            let mut acc = 0;
            for a in 0..g {
                acc = field.add(
                    acc,
                    field.mul(mu.get(0, beta, a), right_inverse_columns[gamma][a]),
                );
            }
            let expected = if beta == gamma { 1 } else { 0 };
            if acc != expected {
                return Err("mu_w*R=I check failed".to_string());
            }
        }
    }
    for beta in 0..n {
        for column in kernel_columns {
            let mut acc = 0;
            for (a, &coeff) in column.iter().enumerate() {
                acc = field.add(acc, field.mul(mu.get(0, beta, a), coeff));
            }
            if acc != 0 {
                return Err("mu_w*B=0 check failed".to_string());
            }
        }
    }
    Ok(())
}

fn build_koszul_metadata(
    g: usize,
    r: usize,
    mu: &MuData,
    v_weights: &[usize],
    a0_weights: &[usize],
    a1_weights: &[usize],
) -> Result<CyclicKoszul> {
    let n = g - 3;
    let m = g / 2;
    let column_subset_counts = subset_weight_counts(v_weights, m, r)?;
    let row_subset_counts = subset_weight_counts(v_weights, m - 1, r)?;
    let a0_counts = weight_counts(a0_weights, r)?;
    let a1_counts = weight_counts(a1_weights, r)?;
    let sector_columns = tensor_sector_counts(&column_subset_counts, &a0_counts, r)?;
    let sector_rows = tensor_sector_counts(&row_subset_counts, &a1_counts, r)?;
    let sector_nnz = ordinary_sector_nnz(g, r, mu, v_weights, a0_weights)?;
    let rows = (n as u64)
        .checked_mul(binom(n, m - 1))
        .ok_or_else(|| "row count overflow".to_string())?;
    let columns = (g as u64)
        .checked_mul(binom(n, m))
        .ok_or_else(|| "column count overflow".to_string())?;
    Ok(CyclicKoszul {
        exterior_degree: m,
        columns,
        rows,
        nnz: sector_nnz.iter().sum(),
        sector_columns,
        sector_rows,
        sector_nnz,
        sector_label: "sum(V weights in exterior subset)+A0 weight mod r".to_string(),
    })
}

fn ordinary_sector_nnz(
    g: usize,
    r: usize,
    mu: &MuData,
    v_weights: &[usize],
    a0_weights: &[usize],
) -> Result<Vec<u64>> {
    let n = g - 3;
    let m = g / 2;
    let mut nonzero_by_i = vec![Vec::<(usize, u64)>::new(); n];
    for (i, slots) in nonzero_by_i.iter_mut().enumerate() {
        for a in 0..g {
            let count = (0..n).filter(|&beta| mu.get(i, beta, a) != 0).count() as u64;
            if count != 0 {
                slots.push((a, count));
            }
        }
    }
    let mut sector_nnz = vec![0u64; r];
    if binom(n, m) == 0 {
        return Ok(sector_nnz);
    }
    let mut subset = (0..m).collect::<Vec<_>>();
    loop {
        let subset_weight = subset
            .iter()
            .fold(0usize, |acc, &i| (acc + v_weights[i]) % r);
        for &i in &subset {
            for &(a, count) in &nonzero_by_i[i] {
                let sector = (subset_weight + a0_weights[a]) % r;
                sector_nnz[sector] = sector_nnz[sector]
                    .checked_add(count)
                    .ok_or_else(|| "sector nnz overflow".to_string())?;
            }
        }
        if !next_subset(&mut subset, n) {
            break;
        }
    }
    Ok(sector_nnz)
}

fn eliminated_sector_counts(
    g: usize,
    r: usize,
    v_weights: &[usize],
    a0_weights: &[usize],
    a1_weights: &[usize],
    w_index: usize,
    w_weight: usize,
    kernel_weights: &[usize],
) -> Result<(Vec<u64>, Vec<u64>)> {
    let m = g / 2;
    if w_index >= v_weights.len() {
        return Err("elimination w index outside V weights".to_string());
    }
    let u_weights = v_weights
        .iter()
        .enumerate()
        .filter_map(|(idx, &weight)| if idx == w_index { None } else { Some(weight) })
        .collect::<Vec<_>>();
    let x_subset_counts = subset_weight_counts(&u_weights, m, r)?;
    let z_subset_counts = subset_weight_counts(&u_weights, m - 1, r)?;
    let target_subset_counts = subset_weight_counts(&u_weights, m - 2, r)?;
    let a0_counts = weight_counts(a0_weights, r)?;
    let a1_counts = weight_counts(a1_weights, r)?;
    let kernel_counts = weight_counts(kernel_weights, r)?;
    let x_counts = tensor_sector_counts(&x_subset_counts, &a0_counts, r)?;
    let z_natural = tensor_sector_counts(&z_subset_counts, &kernel_counts, r)?;
    let target_natural = tensor_sector_counts(&target_subset_counts, &a1_counts, r)?;
    let mut columns = vec![0; r];
    let mut rows = vec![0; r];
    for sector in 0..r {
        let shifted = (sector + r - w_weight % r) % r;
        columns[sector] = x_counts[sector]
            .checked_add(z_natural[shifted])
            .ok_or_else(|| "eliminated column count overflow".to_string())?;
        rows[sector] = target_natural[shifted];
    }
    Ok((columns, rows))
}

fn subset_weight_counts(weights: &[usize], k: usize, r: usize) -> Result<Vec<u64>> {
    let n = weights.len();
    let mut counts = vec![0u64; r];
    if k > n {
        return Ok(counts);
    }
    if k == 0 {
        counts[0] = 1;
        return Ok(counts);
    }
    let mut subset = (0..k).collect::<Vec<_>>();
    loop {
        let weight = subset.iter().fold(0usize, |acc, &i| (acc + weights[i]) % r);
        counts[weight] = counts[weight]
            .checked_add(1)
            .ok_or_else(|| "subset count overflow".to_string())?;
        if !next_subset(&mut subset, n) {
            break;
        }
    }
    Ok(counts)
}

fn tensor_sector_counts(subset_counts: &[u64], coeff_counts: &[u64], r: usize) -> Result<Vec<u64>> {
    let mut out = vec![0u64; r];
    for sector in 0..r {
        let mut total = 0u64;
        for subset_weight in 0..r {
            let coeff_weight = (sector + r - subset_weight) % r;
            total = total
                .checked_add(
                    subset_counts[subset_weight]
                        .checked_mul(coeff_counts[coeff_weight])
                        .ok_or_else(|| "sector dimension multiplication overflow".to_string())?,
                )
                .ok_or_else(|| "sector dimension addition overflow".to_string())?;
        }
        out[sector] = total;
    }
    Ok(out)
}

fn weight_counts(weights: &[usize], r: usize) -> Result<Vec<u64>> {
    let mut counts = vec![0u64; r];
    for &weight in weights {
        if weight >= r {
            return Err(format!("weight {weight} outside 0..{}", r - 1));
        }
        counts[weight] += 1;
    }
    Ok(counts)
}

fn solve_linear_system_one(rows: Vec<Vec<u64>>, rhs: Vec<u64>, field: Field) -> Result<Vec<u64>> {
    if rows.len() != rhs.len() {
        return Err("linear solve row/rhs length mismatch".to_string());
    }
    let vars = rows.first().map_or(0, |row| row.len());
    if rows.iter().any(|row| row.len() != vars) {
        return Err("linear solve rows have inconsistent lengths".to_string());
    }
    let mut aug = rows
        .into_iter()
        .zip(rhs)
        .map(|(mut row, value)| {
            row.push(value);
            row
        })
        .collect::<Vec<_>>();
    let mut pivot_cols = Vec::new();
    let mut pivot_row = 0;
    for col in 0..vars {
        let Some(found) = (pivot_row..aug.len()).find(|&row| aug[row][col] != 0) else {
            continue;
        };
        aug.swap(pivot_row, found);
        let inv = field.inv(aug[pivot_row][col])?;
        for c in col..=vars {
            aug[pivot_row][c] = field.mul(aug[pivot_row][c], inv);
        }
        for row in 0..aug.len() {
            if row == pivot_row {
                continue;
            }
            let factor = aug[row][col];
            if factor == 0 {
                continue;
            }
            for c in col..=vars {
                aug[row][c] = field.sub(aug[row][c], field.mul(factor, aug[pivot_row][c]));
            }
        }
        pivot_cols.push(col);
        pivot_row += 1;
        if pivot_row == aug.len() {
            break;
        }
    }
    for row in &aug {
        if row[..vars].iter().all(|&x| x == 0) && row[vars] != 0 {
            return Err("linear system is inconsistent".to_string());
        }
    }
    let mut solution = vec![0; vars];
    for (row, &col) in pivot_cols.iter().enumerate() {
        solution[col] = aug[row][vars];
    }
    Ok(solution)
}

fn rows_from_columns_fixed(
    columns: &[Vec<u64>],
    rows: usize,
    cols: usize,
) -> Result<Vec<Vec<u64>>> {
    if columns.len() != cols || columns.iter().any(|col| col.len() != rows) {
        return Err(format!(
            "column matrix shape mismatch: expected {rows}x{cols}, got {} columns",
            columns.len()
        ));
    }
    let mut out = vec![vec![0; cols]; rows];
    for (j, column) in columns.iter().enumerate() {
        for (i, &value) in column.iter().enumerate() {
            out[i][j] = value;
        }
    }
    Ok(out)
}

fn indices_with_weight(weights: &[usize], ch: usize) -> Vec<usize> {
    weights
        .iter()
        .enumerate()
        .filter_map(|(idx, &weight)| if weight == ch { Some(idx) } else { None })
        .collect()
}

fn pad_coeffs(coeffs: &[u64], len: usize) -> Vec<u64> {
    let mut out = coeffs.to_vec();
    out.resize(len, 0);
    out
}

fn check_basis_shape(
    label: &str,
    basis: &[Vec<u64>],
    rows: usize,
    cols: usize,
    modulus: u64,
) -> Result<()> {
    if basis.len() != rows {
        return Err(format!(
            "{label} basis has {} vectors, expected {rows}",
            basis.len()
        ));
    }
    for (idx, vector) in basis.iter().enumerate() {
        if vector.len() != cols {
            return Err(format!(
                "{label} basis vector {idx} has length {}, expected {cols}",
                vector.len()
            ));
        }
        if vector.iter().any(|&x| x >= modulus) {
            return Err(format!(
                "{label} basis vector {idx} has coefficient outside field"
            ));
        }
    }
    Ok(())
}

fn check_weight_vector(label: &str, weights: &[usize], expected: usize, r: usize) -> Result<()> {
    if weights.len() != expected {
        return Err(format!(
            "{label} weights have length {}, expected {expected}",
            weights.len()
        ));
    }
    if let Some(weight) = weights.iter().copied().find(|&w| w >= r) {
        return Err(format!(
            "{label} contains weight {weight} outside 0..{}",
            r - 1
        ));
    }
    Ok(())
}

fn verify_homogeneous_basis(
    label: &str,
    basis: &[Vec<u64>],
    weights: &[usize],
    r: usize,
    shift: usize,
) -> Result<()> {
    for (idx, (vector, &weight)) in basis.iter().zip(weights).enumerate() {
        for (exp, &coeff) in vector.iter().enumerate() {
            if coeff != 0 && (exp + shift) % r != weight {
                return Err(format!(
                    "{label} vector {idx} has nonzero exponent {exp} with character {}, expected {weight}",
                    (exp + shift) % r
                ));
            }
        }
    }
    Ok(())
}

fn ensure_all_satisfy(
    label: &str,
    matrix: &[Vec<u64>],
    vectors: &[Vec<u64>],
    field: Field,
) -> Result<()> {
    for (idx, vector) in vectors.iter().enumerate() {
        ensure_satisfies(&format!("{label} {idx}"), matrix, vector, field)?;
    }
    Ok(())
}

fn ensure_satisfies(label: &str, matrix: &[Vec<u64>], vector: &[u64], field: Field) -> Result<()> {
    for (row_idx, row) in matrix.iter().enumerate() {
        if row.len() != vector.len() {
            return Err(format!(
                "{label}: constraint row {row_idx} has length {}, vector has length {}",
                row.len(),
                vector.len()
            ));
        }
        let value = row
            .iter()
            .zip(vector)
            .fold(0, |acc, (&a, &b)| field.add(acc, field.mul(a, b)));
        if value != 0 {
            return Err(format!(
                "{label}: constraint row {row_idx} evaluates to {value}"
            ));
        }
    }
    Ok(())
}

fn verify_elimination_data(
    g: usize,
    r: usize,
    mu: &MuData,
    weights: &CyclicWeights,
    elim: &CyclicElimination,
    field: Field,
) -> Result<()> {
    let n = g - 3;
    if elim.w_v_index >= weights.v.len() {
        return Err("elimination w_V_index outside V".to_string());
    }
    if weights.v[elim.w_v_index] != elim.w_weight {
        return Err("elimination w_weight does not match V weight".to_string());
    }
    if elim.right_inverse_a0_by_a1.len() != g
        || elim.right_inverse_a0_by_a1.iter().any(|row| row.len() != n)
    {
        return Err("right_inverse_A0_by_A1 has wrong shape".to_string());
    }
    if elim.kernel_basis_a0_by_3.len() != g
        || elim.kernel_basis_a0_by_3.iter().any(|row| row.len() != 3)
    {
        return Err("kernel_basis_A0_by_3 has wrong shape".to_string());
    }
    check_weight_vector("kernel_weights", &elim.kernel_weights, 3, r)?;
    let right_inverse_columns = (0..n)
        .map(|beta| {
            (0..g)
                .map(|a| elim.right_inverse_a0_by_a1[a][beta])
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let kernel_columns = (0..3)
        .map(|j| {
            (0..g)
                .map(|a| elim.kernel_basis_a0_by_3[a][j])
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let mu_w_columns = (0..g)
        .map(|a| (0..n).map(|beta| mu.get(elim.w_v_index, beta, a)).collect())
        .collect::<Vec<Vec<_>>>();
    let mu_w_rank = rank_columns(&mu_w_columns, field)?;
    if mu_w_rank != elim.mu_w_rank {
        return Err("stored mu_w_rank does not match recomputed rank".to_string());
    }
    if mu_w_rank != n {
        return Err("stored elimination direction is not surjective".to_string());
    }
    verify_right_inverse_and_kernel_for_index(
        mu,
        elim.w_v_index,
        &right_inverse_columns,
        &kernel_columns,
        field,
    )
}

fn verify_right_inverse_and_kernel_for_index(
    mu: &MuData,
    i: usize,
    right_inverse_columns: &[Vec<u64>],
    kernel_columns: &[Vec<u64>],
    field: Field,
) -> Result<()> {
    let n = mu.n;
    let g = mu.g;
    for beta in 0..n {
        for gamma in 0..n {
            let mut acc = 0;
            for a in 0..g {
                acc = field.add(
                    acc,
                    field.mul(mu.get(i, beta, a), right_inverse_columns[gamma][a]),
                );
            }
            let expected = if beta == gamma { 1 } else { 0 };
            if acc != expected {
                return Err("stored mu_w*R=I check failed".to_string());
            }
        }
    }
    for beta in 0..n {
        for column in kernel_columns {
            let mut acc = 0;
            for (a, &coeff) in column.iter().enumerate() {
                acc = field.add(acc, field.mul(mu.get(i, beta, a), coeff));
            }
            if acc != 0 {
                return Err("stored mu_w*B=0 check failed".to_string());
            }
        }
    }
    Ok(())
}

pub fn parse_orbit_reps(input: &str) -> Result<Vec<[u64; 2]>> {
    let values = input
        .split(',')
        .enumerate()
        .map(|(idx, token)| {
            token.trim().parse::<u64>().map_err(|e| {
                format!("orbit representative entry {idx} ({token:?}) is not a u64: {e}")
            })
        })
        .collect::<Result<Vec<_>>>()?;
    if values.len() % 2 != 0 {
        return Err("orbit representatives must contain an even number of entries".to_string());
    }
    Ok(values
        .chunks_exact(2)
        .map(|pair| [pair[0], pair[1]])
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_orbit_reps_pairs_coordinates() {
        assert_eq!(
            parse_orbit_reps("1, 2,4,10").unwrap(),
            vec![[1, 2], [4, 10]]
        );
        assert!(parse_orbit_reps("1,2,3").is_err());
    }

    #[test]
    fn generate_and_verify_small_cyclic_instance() {
        let instance = generate_cyclic_instance(&CyclicGenerateOptions {
            genus: 6,
            order: 3,
            prime: 31,
            zeta: 5,
            orbit_reps: vec![[1, 2], [3, 4]],
        })
        .unwrap();
        let report = verify_cyclic_instance(&instance).unwrap();
        assert_eq!(report.ordinary_rows, 9);
        assert_eq!(report.ordinary_columns, 6);
        assert_eq!(report.eliminated_rows, Some(6));
        assert_eq!(report.eliminated_columns, Some(3));
    }
}

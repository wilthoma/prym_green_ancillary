//! Matrix-free applications of the paired deformation through order two.
//! All coefficients are coefficients of t^k, not kth derivatives. This module
//! extracts two independent base-kernel columns, forms F1*K, normalizes the
//! augmented kernels to solutions F0*S=F1*K, and forms Z0=F2*K-F1*S.
//! Dense blocks are row-major; original, unpreconditioned kernel coordinates
//! are read from CUDA recovery outputs. The reader-facing workflow is the
//! repository reproduce script; the commands here are its internal stages.

use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    fs::{self, File},
    io::{BufRead, BufReader, BufWriter, Write},
    path::{Path, PathBuf},
    time::Instant,
};

use crate::Result;

const DENSE_BLOCK_FORMAT: &str = "prym_deformation_dense_block_v1";
const DROP_COLUMNS_FORMAT: &str = "prym_deformation_drop_columns_v1";

#[derive(Clone, Debug, Deserialize)]
struct DeformationInstance {
    fixture_format: String,
    genus: usize,
    modulus: u64,
    cyclic: DeformationCyclic,
    weights: DeformationWeights,
    third_section_elimination: DeformationElimination,
    deformation: DeformationData,
}

#[derive(Clone, Debug, Deserialize)]
struct DeformationCyclic {
    order: usize,
}

#[derive(Clone, Debug, Deserialize)]
struct DeformationWeights {
    #[serde(rename = "V")]
    v: Vec<usize>,
    #[serde(rename = "A0")]
    a0: Vec<usize>,
    #[serde(rename = "A1")]
    a1: Vec<usize>,
}

#[derive(Clone, Debug, Deserialize)]
struct DeformationElimination {
    #[serde(rename = "w_V_index")]
    w_v_index: usize,
    w_weight: usize,
    kernel_weights: Vec<usize>,
    sector_columns: Vec<u64>,
    sector_rows: Vec<u64>,
}

#[derive(Clone, Debug, Deserialize)]
struct DeformationData {
    order: usize,
    mode: String,
    coefficient_convention: String,
    #[serde(default)]
    allowed_charges: Vec<Vec<usize>>,
    mu_shape: Vec<usize>,
    mu_layout: String,
    mu_coefficients: Vec<Vec<u64>>,
    right_inverse_shape: Vec<usize>,
    right_inverse_coefficients: Vec<Vec<u64>>,
    kernel_basis_shape: Vec<usize>,
    kernel_basis_coefficients: Vec<Vec<u64>>,
}

#[derive(Clone, Debug)]
struct DenseBlock {
    rows: usize,
    columns: usize,
    values: Vec<u32>,
}

#[derive(Clone, Copy, Debug)]
struct MuTerm {
    beta: usize,
    a: usize,
    coeff: u32,
}

#[derive(Clone, Copy, Debug)]
struct LinTerm {
    index: usize,
    coeff: u32,
}

#[derive(Clone, Copy, Debug)]
struct DiffTerm {
    local_i: usize,
    source_rank: usize,
    neg: bool,
}

#[derive(Clone, Debug)]
struct DiffIncidence {
    source_count: usize,
    target_count: usize,
    offsets: Vec<usize>,
    terms: Vec<DiffTerm>,
}

#[derive(Clone, Debug)]
struct DeformationAction {
    g: usize,
    p: u32,
    r: usize,
    n: usize,
    max_order: usize,
    local_indices: Vec<usize>,
    mu_terms: Vec<Vec<Vec<MuTerm>>>,
    right_terms: Vec<Vec<Vec<LinTerm>>>,
    basis_terms: Vec<Vec<Vec<LinTerm>>>,
    diff_dm: DiffIncidence,
    diff_d2: DiffIncidence,
    sub_m1_count: usize,
    x_total: usize,
    col_total: usize,
    row_total: usize,
    col_indices: Vec<Vec<usize>>,
    row_indices: Vec<Vec<usize>>,
}

#[derive(Debug, Deserialize)]
struct DenseBlockPayload {
    #[serde(rename = "type")]
    kind: Option<String>,
    prime: u64,
    rows: usize,
    columns: usize,
    layout: String,
    values: Vec<u32>,
}

#[derive(Serialize)]
struct DenseBlockOutput<'a> {
    #[serde(rename = "type")]
    kind: &'static str,
    prime: u64,
    rows: usize,
    columns: usize,
    layout: &'static str,
    values: &'a [u32],
    #[serde(flatten)]
    metadata: BTreeMap<String, Value>,
}

#[derive(Serialize)]
struct DropColumnsOutput {
    #[serde(rename = "type")]
    kind: &'static str,
    prime: u64,
    indices: Vec<usize>,
    #[serde(flatten)]
    metadata: BTreeMap<String, Value>,
}

#[derive(Debug, Deserialize)]
struct KernelReport {
    operator: String,
    sector: usize,
    rows: usize,
    columns: usize,
    prime: u64,
    #[serde(default)]
    candidate_rank: Option<usize>,
    #[serde(default)]
    pivot_columns: Option<Vec<usize>>,
    #[serde(default)]
    product_nonzeros: Option<Vec<usize>>,
}

#[derive(Debug, Deserialize)]
struct RunManifest {
    runs: Vec<ManifestRun>,
}

#[derive(Debug, Deserialize)]
struct ManifestRun {
    run_id: String,
    sector: usize,
    rows: usize,
    columns: usize,
    wdm_path: String,
    args: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct SolutionsManifest {
    solutions: Vec<SolutionRecordInput>,
}

#[derive(Debug, Deserialize)]
struct SolutionRecordInput {
    sector: usize,
    path: PathBuf,
}

pub fn extract_base_kernel(
    instance_path: &Path,
    kernel_vectors_path: &Path,
    kernel_report_path: Option<&Path>,
    out_dir: &Path,
) -> Result<()> {
    eprintln!(
        "deformation-extract-base-kernel: loading {}",
        instance_path.display()
    );
    let instance = read_deformation_instance(instance_path)?;
    let action = DeformationAction::new(&instance)?;
    fs::create_dir_all(out_dir)
        .map_err(|e| format!("failed to create {}: {e}", out_dir.display()))?;

    eprintln!(
        "deformation-extract-base-kernel: reading candidate vectors {}",
        kernel_vectors_path.display()
    );
    let candidates = read_vector_lines(kernel_vectors_path, action.p)?;
    let expected = action.sector_columns(0)?;
    if candidates.rows != expected {
        return Err(format!(
            "kernel vector length {} does not match sector-0 columns {expected}",
            candidates.rows
        ));
    }
    let (selected, pivots) = select_kernel_columns(&candidates, kernel_report_path, action.p)?;

    let (certified_by_cuda, cuda_report) = cuda_kernel_report_certifies(
        kernel_report_path,
        "eliminated",
        0,
        action.sector_rows(0)?,
        expected,
        action.p,
    )?;
    let (residual_nonzeros, certificate_source) = if certified_by_cuda {
        (0usize, "cuda_kernel_vector_report")
    } else {
        eprintln!("deformation-extract-base-kernel: certifying F0*K in Rust");
        let full = action.sector_to_global_source(0, &selected)?;
        let residual = action.apply_f_order(0, &full, selected.columns)?;
        let residual0 = action.global_to_target_sector(0, &residual, selected.columns)?;
        let nonzeros = count_nonzeros(&residual0.values);
        if nonzeros != 0 {
            return Err("base kernel vectors do not satisfy F0,0 K=0".to_string());
        }
        (nonzeros, "rust_matrix_free_F0")
    };

    let kernel_path = out_dir.join("K_sector0.json");
    let mut metadata = BTreeMap::new();
    metadata.insert("sector".to_string(), json!(0));
    metadata.insert("selected_candidate_columns".to_string(), json!(pivots));
    write_dense_block(&kernel_path, &selected, action.p, metadata)?;

    let report = json!({
        "type": "prym_deformation_base_kernel_report_v1",
        "kernel_vectors": path_string(kernel_vectors_path),
        "kernel_report": kernel_report_path.map(path_string),
        "selected_candidate_columns": pivots,
        "sector": 0,
        "rows": selected.rows,
        "columns": selected.columns,
        "rank": 2,
        "F0_residual_nonzeros": residual_nonzeros,
        "certificate_source": certificate_source,
        "cuda_report_product_nonzeros": cuda_report.and_then(|report| report.product_nonzeros),
        "output": path_string(&kernel_path),
    });
    write_pretty_json(&out_dir.join("base_kernel_report.json"), &report)?;
    println!("Wrote base kernel block: {}", kernel_path.display());
    print_pretty_json(&report)?;
    Ok(())
}

pub fn derive_rhs(
    instance_path: &Path,
    kernel_path: &Path,
    out_dir: &Path,
    expected_corrections: Option<&str>,
) -> Result<()> {
    eprintln!(
        "deformation-derive-rhs: loading {}",
        instance_path.display()
    );
    let instance = read_deformation_instance(instance_path)?;
    let action = DeformationAction::new(&instance)?;
    fs::create_dir_all(out_dir)
        .map_err(|e| format!("failed to create {}: {e}", out_dir.display()))?;

    eprintln!("deformation-derive-rhs: reading {}", kernel_path.display());
    let kernel = read_dense_block(kernel_path, action.p)?;
    let started = Instant::now();
    let full_k = action.sector_to_global_source(0, &kernel)?;
    eprintln!(
        "deformation-derive-rhs: applying F1 to {} vectors",
        kernel.columns
    );
    let rhs_global = action.apply_f_order(1, &full_k, kernel.columns)?;

    let mut rhs_records = Vec::new();
    for sector in 0..action.r {
        let block = action.global_to_target_sector(sector, &rhs_global, kernel.columns)?;
        let nonzeros = count_nonzeros(&block.values);
        if nonzeros == 0 {
            continue;
        }
        let out = out_dir.join(format!("rhs_F1K_sector{sector:02}.json"));
        let mut metadata = BTreeMap::new();
        metadata.insert("sector".to_string(), json!(sector));
        metadata.insert("source".to_string(), json!("F1*K"));
        metadata.insert("nonzeros".to_string(), json!(nonzeros));
        write_dense_block(&out, &block, action.p, metadata)?;
        rhs_records.push(json!({
            "sector": sector,
            "rows": block.rows,
            "columns": block.columns,
            "nonzeros": nonzeros,
            "path": path_string(&out),
        }));
    }

    let expected = parse_sector_list(expected_corrections)?;
    let found: BTreeSet<usize> = rhs_records
        .iter()
        .filter_map(|record| {
            record
                .get("sector")
                .and_then(Value::as_u64)
                .map(|x| x as usize)
        })
        .collect();
    let expected_set: BTreeSet<usize> = expected.iter().copied().collect();
    let matches_expected = expected_set.is_empty() || found == expected_set;
    if !matches_expected {
        return Err(format!(
            "correction sectors {:?} do not match expected {:?}",
            found, expected_set
        ));
    }

    let report = json!({
        "type": "prym_deformation_rhs_manifest_v1",
        "elapsed_seconds": started.elapsed().as_secs_f64(),
        "expected_correction_sectors": expected,
        "correction_sectors": found.into_iter().collect::<Vec<_>>(),
        "matches_expected": matches_expected,
        "rhs": rhs_records,
    });
    let manifest_path = out_dir.join("rhs_manifest.json");
    write_pretty_json(&manifest_path, &report)?;
    println!("Wrote RHS manifest: {}", manifest_path.display());
    print_pretty_json(&report)?;
    Ok(())
}

pub fn normalize_augmented(
    instance_path: &Path,
    manifest_path: &Path,
    out_dir: &Path,
) -> Result<()> {
    eprintln!(
        "deformation-normalize-augmented: loading {}",
        instance_path.display()
    );
    let instance = read_deformation_instance(instance_path)?;
    let action = DeformationAction::new(&instance)?;
    let manifest: RunManifest = read_json_file(manifest_path)?;
    let expected_sectors = BTreeSet::from([1, action.r - 1]);
    let found_sectors: BTreeSet<_> = manifest.runs.iter().map(|run| run.sector).collect();
    if manifest.runs.len() != 2 || found_sectors != expected_sectors {
        return Err(format!(
            "paired deformation requires exactly the two augmented sectors {expected_sectors:?}"
        ));
    }
    fs::create_dir_all(out_dir)
        .map_err(|e| format!("failed to create {}: {e}", out_dir.display()))?;

    let mut solution_records = Vec::new();
    for run in &manifest.runs {
        let sector = run.sector;
        let vectors_path = default_nullvectors_path(&run.wdm_path);
        let report_path = default_nullvectors_report_path(&run.wdm_path);
        eprintln!(
            "deformation-normalize-augmented: reading augmented candidates for {}",
            run.run_id
        );
        let candidates = read_vector_lines(&vectors_path, action.p)?;
        let base_cols = action.sector_columns(sector)?;
        if candidates.rows != base_cols + 2 {
            return Err(format!(
                "{} length {} does not match augmented columns {}",
                vectors_path.display(),
                candidates.rows,
                base_cols + 2
            ));
        }
        if run.columns != base_cols + 2 {
            return Err(format!(
                "{} manifest columns {} do not match expected {}",
                run.run_id,
                run.columns,
                base_cols + 2
            ));
        }
        let (solution, selected_candidates) =
            normalize_augmented_candidates(&candidates, base_cols, action.p)?;
        let rhs_path = dense_file_arg(run)?;
        let rhs = read_dense_block(&rhs_path, action.p)?;
        if rhs.rows != run.rows || rhs.columns != 2 {
            return Err(format!(
                "{} dimensions {}x{} do not match augmented run {}x2",
                rhs_path.display(),
                rhs.rows,
                rhs.columns,
                run.rows
            ));
        }

        let (certified_by_cuda, cuda_report) = cuda_kernel_report_certifies(
            Some(&report_path),
            "augmented",
            sector,
            run.rows,
            run.columns,
            action.p,
        )?;
        let (residual_nonzeros, certificate_source) = if certified_by_cuda {
            (0usize, "cuda_augmented_kernel_report")
        } else {
            eprintln!(
                "deformation-normalize-augmented: certifying F0*S=b for sector {sector} in Rust"
            );
            let full_solution = action.sector_to_global_source(sector, &solution)?;
            let lhs_global = action.apply_f_order(0, &full_solution, solution.columns)?;
            let lhs = action.global_to_target_sector(sector, &lhs_global, solution.columns)?;
            let nonzeros = count_difference_nonzeros(&lhs, &rhs, action.p)?;
            if nonzeros != 0 {
                return Err(format!(
                    "first-order residual for sector {sector} has {nonzeros} nonzeros"
                ));
            }
            (nonzeros, "rust_matrix_free_F0")
        };

        let out = out_dir.join(format!("S_sector{sector:02}.json"));
        let mut metadata = BTreeMap::new();
        metadata.insert("sector".to_string(), json!(sector));
        metadata.insert("source".to_string(), json!("normalized augmented kernel"));
        metadata.insert(
            "selected_candidate_columns".to_string(),
            json!(selected_candidates),
        );
        metadata.insert(
            "augmented_vectors".to_string(),
            json!(path_string(&vectors_path)),
        );
        write_dense_block(&out, &solution, action.p, metadata)?;
        solution_records.push(json!({
            "sector": sector,
            "path": path_string(&out),
            "rows": solution.rows,
            "columns": solution.columns,
            "selected_candidate_columns": selected_candidates,
            "residual_nonzeros": residual_nonzeros,
            "certificate_source": certificate_source,
            "cuda_report_product_nonzeros": cuda_report.and_then(|report| report.product_nonzeros),
        }));
    }

    let report = json!({
        "type": "prym_deformation_solutions_manifest_v1",
        "solutions": solution_records,
        "first_order_equations_verified": solution_records
            .iter()
            .all(|row| row.get("residual_nonzeros").and_then(Value::as_u64) == Some(0)),
    });
    let solutions_path = out_dir.join("solutions_manifest.json");
    write_pretty_json(&solutions_path, &report)?;
    println!("Wrote solutions manifest: {}", solutions_path.display());
    print_pretty_json(&report)?;
    Ok(())
}

pub fn derive_quadratic(
    instance_path: &Path,
    kernel_path: &Path,
    solutions_manifest_path: &Path,
    out_dir: &Path,
) -> Result<()> {
    eprintln!(
        "deformation-derive-quadratic: loading {}",
        instance_path.display()
    );
    let instance = read_deformation_instance(instance_path)?;
    let action = DeformationAction::new(&instance)?;
    let kernel = read_dense_block(kernel_path, action.p)?;
    let solutions_manifest: SolutionsManifest = read_json_file(solutions_manifest_path)?;
    let expected_sectors = BTreeSet::from([1, action.r - 1]);
    let found_sectors: BTreeSet<_> = solutions_manifest
        .solutions
        .iter()
        .map(|record| record.sector)
        .collect();
    if solutions_manifest.solutions.len() != 2 || found_sectors != expected_sectors {
        return Err(format!(
            "paired deformation requires exactly the two correction sectors {expected_sectors:?}"
        ));
    }
    fs::create_dir_all(out_dir)
        .map_err(|e| format!("failed to create {}: {e}", out_dir.display()))?;

    let mut solutions = BTreeMap::new();
    for record in &solutions_manifest.solutions {
        let block = read_dense_block(&record.path, action.p)?;
        solutions.insert(record.sector, block);
    }

    let started = Instant::now();
    let full_k = action.sector_to_global_source(0, &kernel)?;
    let full_s = action.sectors_to_global_source(&solutions, kernel.columns)?;
    eprintln!(
        "deformation-derive-quadratic: applying F2*K and F1*S to {} vectors",
        kernel.columns
    );
    let mut obstruction = action.apply_f_order(2, &full_k, kernel.columns)?;
    let first_order = action.apply_f_order(1, &full_s, kernel.columns)?;
    sub_vec_in_place(&mut obstruction, &first_order, action.p);
    let z0 = action.global_to_target_sector(0, &obstruction, kernel.columns)?;
    let drop_columns = choose_independent_rows(&kernel, action.p)?;

    let z_path = out_dir.join("Z0_quadratic_obstruction.json");
    let mut z_meta = BTreeMap::new();
    z_meta.insert("sector".to_string(), json!(0));
    z_meta.insert("source".to_string(), json!("F2*K-F1*S"));
    write_dense_block(&z_path, &z0, action.p, z_meta)?;

    let drop_path = out_dir.join("replacement_drop_columns.json");
    let mut drop_meta = BTreeMap::new();
    drop_meta.insert("sector".to_string(), json!(0));
    drop_meta.insert("source".to_string(), json!("pivot rows of K_sector0"));
    write_drop_columns(&drop_path, &drop_columns, action.p, drop_meta)?;

    let report = json!({
        "type": "prym_deformation_quadratic_manifest_v1",
        "elapsed_seconds": started.elapsed().as_secs_f64(),
        "Z0": {
            "path": path_string(&z_path),
            "rows": z0.rows,
            "columns": z0.columns,
            "nonzeros": count_nonzeros(&z0.values),
        },
        "drop_columns": {
            "path": path_string(&drop_path),
            "indices": drop_columns,
        },
    });
    let manifest_path = out_dir.join("quadratic_manifest.json");
    write_pretty_json(&manifest_path, &report)?;
    println!("Wrote quadratic manifest: {}", manifest_path.display());
    print_pretty_json(&report)?;
    Ok(())
}

impl DeformationAction {
    fn new(instance: &DeformationInstance) -> Result<Self> {
        if instance.fixture_format != "prym-cyclic-deformation-v1" {
            return Err(format!(
                "unsupported deformation fixture format {}",
                instance.fixture_format
            ));
        }
        let g = instance.genus;
        let n = g
            .checked_sub(3)
            .ok_or_else(|| format!("genus {g} is too small"))?;
        // Addition uses u32 and products use u64. The paper's primes and
        // the independent reference are all within this conservative bound.
        if instance.modulus > 65521
            || instance.modulus < 3
            || !crate::field::is_prime(instance.modulus)
        {
            return Err(format!(
                "deformation helper expects an odd prime at most 65521, got {}",
                instance.modulus
            ));
        }
        let p = instance.modulus as u32;
        let r = instance.cyclic.order;
        let m = g / 2;
        if g < 6 || !g.is_multiple_of(2) || r != m {
            return Err(
                "paired deformation requires even genus g>=6 and cyclic order g/2".to_string(),
            );
        }
        let max_order = instance.deformation.order;
        if max_order != 2 || instance.deformation.mode != "paired" {
            return Err(
                "deformation fixture must use paired motion through order exactly 2".to_string(),
            );
        }
        if instance.deformation.coefficient_convention != "coefficient of t^k, not kth derivative" {
            return Err(format!(
                "unsupported coefficient convention {}",
                instance.deformation.coefficient_convention
            ));
        }
        if instance.deformation.mu_layout != "((i*n)+beta)*g+a" {
            return Err(format!(
                "unsupported mu layout {}",
                instance.deformation.mu_layout
            ));
        }
        if instance.weights.v.len() != n {
            return Err(format!(
                "V weight count {} does not match n={n}",
                instance.weights.v.len()
            ));
        }
        if instance.weights.a0.len() != g {
            return Err(format!(
                "A0 weight count {} does not match g={g}",
                instance.weights.a0.len()
            ));
        }
        if instance.weights.a1.len() != n {
            return Err(format!(
                "A1 weight count {} does not match n={n}",
                instance.weights.a1.len()
            ));
        }
        if instance.third_section_elimination.kernel_weights.len() != 3 {
            return Err("elimination kernel basis must have three weights".to_string());
        }
        let w_index = instance.third_section_elimination.w_v_index;
        if w_index >= n {
            return Err(format!("w_V_index {w_index} is outside 0..{n}"));
        }
        let local_indices: Vec<_> = (0..n).filter(|&i| i != w_index).collect();
        let local_weights: Vec<_> = local_indices
            .iter()
            .map(|&i| instance.weights.v[i])
            .collect();

        validate_shape(&instance.deformation.mu_shape, &[n, n, g], "mu_shape")?;
        validate_shape(
            &instance.deformation.right_inverse_shape,
            &[g, n],
            "right_inverse_shape",
        )?;
        validate_shape(
            &instance.deformation.kernel_basis_shape,
            &[g, 3],
            "kernel_basis_shape",
        )?;

        let mu_terms = build_mu_terms(&instance.deformation.mu_coefficients, n, g, p)?;
        let right_terms = build_linear_terms(
            &instance.deformation.right_inverse_coefficients,
            g,
            n,
            p,
            "right_inverse_coefficients",
        )?;
        let basis_terms = build_linear_terms(
            &instance.deformation.kernel_basis_coefficients,
            g,
            3,
            p,
            "kernel_basis_coefficients",
        )?;

        let (sub_m, sub_m_weights) = enumerate_subsets(&local_weights, m)?;
        let (sub_m1, sub_m1_weights) = enumerate_subsets(&local_weights, m - 1)?;
        let (sub_m2, sub_m2_weights) = enumerate_subsets(&local_weights, m - 2)?;
        let diff_dm = build_incidence(&sub_m, &sub_m1, local_indices.len())?;
        let diff_d2 = build_incidence(&sub_m1, &sub_m2, local_indices.len())?;

        let x_total = sub_m.len() * g;
        let col_total = x_total + sub_m1.len() * 3;
        let row_total = sub_m2.len() * n;
        let column_weights = {
            let mut weights = tensor_weights(&sub_m_weights, &instance.weights.a0, 0, r);
            weights.extend(tensor_weights(
                &sub_m1_weights,
                &instance.third_section_elimination.kernel_weights,
                instance.third_section_elimination.w_weight,
                r,
            ));
            weights
        };
        let row_weights = tensor_weights(
            &sub_m2_weights,
            &instance.weights.a1,
            instance.third_section_elimination.w_weight,
            r,
        );
        let col_indices = indices_by_residue(&column_weights, r);
        let row_indices = indices_by_residue(&row_weights, r);

        let computed_cols: Vec<u64> = col_indices
            .iter()
            .map(|indices| indices.len() as u64)
            .collect();
        let computed_rows: Vec<u64> = row_indices
            .iter()
            .map(|indices| indices.len() as u64)
            .collect();
        if computed_cols != instance.third_section_elimination.sector_columns {
            return Err(format!(
                "sector column mismatch: computed {:?}, fixture {:?}",
                computed_cols, instance.third_section_elimination.sector_columns
            ));
        }
        if computed_rows != instance.third_section_elimination.sector_rows {
            return Err(format!(
                "sector row mismatch: computed {:?}, fixture {:?}",
                computed_rows, instance.third_section_elimination.sector_rows
            ));
        }
        if !instance.deformation.allowed_charges.is_empty() {
            eprintln!(
                "deformation helper: mode={} allowed_charges={:?}",
                instance.deformation.mode, instance.deformation.allowed_charges
            );
        }

        Ok(Self {
            g,
            p,
            r,
            n,
            max_order,
            local_indices,
            mu_terms,
            right_terms,
            basis_terms,
            diff_dm,
            diff_d2,
            sub_m1_count: sub_m1.len(),
            x_total,
            col_total,
            row_total,
            col_indices,
            row_indices,
        })
    }

    fn sector_columns(&self, sector: usize) -> Result<usize> {
        self.col_indices
            .get(sector)
            .map(Vec::len)
            .ok_or_else(|| format!("sector {sector} is outside 0..{}", self.r))
    }

    fn sector_rows(&self, sector: usize) -> Result<usize> {
        self.row_indices
            .get(sector)
            .map(Vec::len)
            .ok_or_else(|| format!("sector {sector} is outside 0..{}", self.r))
    }

    fn sector_to_global_source(&self, sector: usize, block: &DenseBlock) -> Result<Vec<u32>> {
        let indices = self
            .col_indices
            .get(sector)
            .ok_or_else(|| format!("sector {sector} is outside 0..{}", self.r))?;
        if block.rows != indices.len() {
            return Err(format!(
                "sector {sector} source length {} does not match {}",
                block.rows,
                indices.len()
            ));
        }
        let mut out = vec![0u32; self.col_total * block.columns];
        for (local_row, &global_row) in indices.iter().enumerate() {
            let source = local_row * block.columns;
            let target = global_row * block.columns;
            out[target..target + block.columns]
                .copy_from_slice(&block.values[source..source + block.columns]);
        }
        Ok(out)
    }

    fn sectors_to_global_source(
        &self,
        blocks: &BTreeMap<usize, DenseBlock>,
        vecs: usize,
    ) -> Result<Vec<u32>> {
        let mut out = vec![0u32; self.col_total * vecs];
        for (&sector, block) in blocks {
            if block.columns != vecs {
                return Err(format!(
                    "sector {sector} block has {} vectors, expected {vecs}",
                    block.columns
                ));
            }
            let indices = self
                .col_indices
                .get(sector)
                .ok_or_else(|| format!("sector {sector} is outside 0..{}", self.r))?;
            if block.rows != indices.len() {
                return Err(format!(
                    "sector {sector} source length {} does not match {}",
                    block.rows,
                    indices.len()
                ));
            }
            for (local_row, &global_row) in indices.iter().enumerate() {
                let source = local_row * vecs;
                let target = global_row * vecs;
                out[target..target + vecs].copy_from_slice(&block.values[source..source + vecs]);
            }
        }
        Ok(out)
    }

    fn global_to_target_sector(
        &self,
        sector: usize,
        global: &[u32],
        vecs: usize,
    ) -> Result<DenseBlock> {
        if global.len() != self.row_total * vecs {
            return Err(format!(
                "target block length {} does not match {}",
                global.len(),
                self.row_total * vecs
            ));
        }
        let indices = self
            .row_indices
            .get(sector)
            .ok_or_else(|| format!("sector {sector} is outside 0..{}", self.r))?;
        let mut values = Vec::with_capacity(indices.len() * vecs);
        for &global_row in indices {
            let source = global_row * vecs;
            values.extend_from_slice(&global[source..source + vecs]);
        }
        Ok(DenseBlock {
            rows: indices.len(),
            columns: vecs,
            values,
        })
    }

    fn apply_f_order(&self, order: usize, source_vectors: &[u32], vecs: usize) -> Result<Vec<u32>> {
        if order > self.max_order {
            return Err(format!(
                "requested deformation order {order}, fixture contains only {}",
                self.max_order
            ));
        }
        if source_vectors.len() != self.col_total * vecs {
            return Err(format!(
                "source vector length {} does not match {}",
                source_vectors.len(),
                self.col_total * vecs
            ));
        }
        let x_end = self.x_total * vecs;
        let x = &source_vectors[..x_end];
        let z = &source_vectors[x_end..];

        eprintln!("  F{order}: computing D_m terms");
        let mut dm_terms = Vec::with_capacity(order + 1);
        for k in 0..=order {
            dm_terms.push(self.apply_differential(k, x, &self.diff_dm, self.g, self.n, vecs)?);
        }

        eprintln!("  F{order}: computing B terms");
        let mut basis_terms = Vec::with_capacity(order + 1);
        for k in 0..=order {
            basis_terms.push(self.apply_basis(k, z, self.sub_m1_count, vecs)?);
        }

        let mut out = vec![0u32; self.row_total * vecs];
        for a in 0..=order {
            eprintln!("  F{order}: accumulating D2_{a} B_{}", order - a);
            let dz = self.apply_differential(
                a,
                &basis_terms[order - a],
                &self.diff_d2,
                self.g,
                self.n,
                vecs,
            )?;
            add_vec_in_place(&mut out, &dz, self.p);

            for b in 0..=order - a {
                let c = order - a - b;
                eprintln!("  F{order}: accumulating -D2_{a} R_{b} Dm_{c}");
                let y = self.apply_right(b, &dm_terms[c], self.sub_m1_count, vecs)?;
                let dy = self.apply_differential(a, &y, &self.diff_d2, self.g, self.n, vecs)?;
                sub_vec_in_place(&mut out, &dy, self.p);
            }
        }
        Ok(out)
    }

    fn apply_differential(
        &self,
        order: usize,
        source: &[u32],
        incidence: &DiffIncidence,
        source_coeffs: usize,
        target_coeffs: usize,
        vecs: usize,
    ) -> Result<Vec<u32>> {
        if source.len() != incidence.source_count * source_coeffs * vecs {
            return Err(format!(
                "differential source length {} does not match {}",
                source.len(),
                incidence.source_count * source_coeffs * vecs
            ));
        }
        let mut out = vec![0u32; incidence.target_count * target_coeffs * vecs];
        let p = self.p;
        let mu_terms = &self.mu_terms[order];
        let local_indices = &self.local_indices;
        out.par_chunks_mut(target_coeffs * vecs)
            .enumerate()
            .for_each(|(target_rank, out_chunk)| {
                for term_index in incidence.offsets[target_rank]..incidence.offsets[target_rank + 1]
                {
                    let term = incidence.terms[term_index];
                    let original_i = local_indices[term.local_i];
                    let source_base = term.source_rank * source_coeffs * vecs;
                    for mu in &mu_terms[original_i] {
                        let coeff = if term.neg {
                            neg_mod(mu.coeff, p)
                        } else {
                            mu.coeff
                        };
                        if coeff == 0 {
                            continue;
                        }
                        let source_offset = source_base + mu.a * vecs;
                        let target_offset = mu.beta * vecs;
                        for v in 0..vecs {
                            add_mul_mod(
                                &mut out_chunk[target_offset + v],
                                coeff,
                                source[source_offset + v],
                                p,
                            );
                        }
                    }
                }
            });
        Ok(out)
    }

    fn apply_right(
        &self,
        order: usize,
        source: &[u32],
        subsets: usize,
        vecs: usize,
    ) -> Result<Vec<u32>> {
        self.apply_linear(order, source, subsets, self.n, &self.right_terms, vecs)
    }

    fn apply_basis(
        &self,
        order: usize,
        source: &[u32],
        subsets: usize,
        vecs: usize,
    ) -> Result<Vec<u32>> {
        self.apply_linear(order, source, subsets, 3, &self.basis_terms, vecs)
    }

    fn apply_linear(
        &self,
        order: usize,
        source: &[u32],
        subsets: usize,
        source_dim: usize,
        terms: &[Vec<Vec<LinTerm>>],
        vecs: usize,
    ) -> Result<Vec<u32>> {
        if source.len() != subsets * source_dim * vecs {
            return Err(format!(
                "linear source length {} does not match {}",
                source.len(),
                subsets * source_dim * vecs
            ));
        }
        let mut out = vec![0u32; subsets * self.g * vecs];
        let p = self.p;
        out.par_chunks_mut(self.g * vecs)
            .enumerate()
            .for_each(|(subset_rank, out_chunk)| {
                let source_base = subset_rank * source_dim * vecs;
                for (a, output_terms) in terms[order].iter().enumerate() {
                    let target_offset = a * vecs;
                    for term in output_terms {
                        let source_offset = source_base + term.index * vecs;
                        for v in 0..vecs {
                            add_mul_mod(
                                &mut out_chunk[target_offset + v],
                                term.coeff,
                                source[source_offset + v],
                                p,
                            );
                        }
                    }
                }
            });
        Ok(out)
    }
}

fn read_deformation_instance(path: &Path) -> Result<DeformationInstance> {
    read_json_file(path)
}

fn read_json_file<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T> {
    let file = File::open(path).map_err(|e| format!("failed to open {}: {e}", path.display()))?;
    serde_json::from_reader(BufReader::new(file))
        .map_err(|e| format!("failed to parse {}: {e}", path.display()))
}

fn validate_shape(actual: &[usize], expected: &[usize], label: &str) -> Result<()> {
    if actual != expected {
        return Err(format!("{label} {actual:?} does not match {expected:?}"));
    }
    Ok(())
}

fn build_mu_terms(
    coefficients: &[Vec<u64>],
    n: usize,
    g: usize,
    p: u32,
) -> Result<Vec<Vec<Vec<MuTerm>>>> {
    if coefficients.len() < 3 {
        return Err(format!(
            "mu_coefficients contains {} orders, expected at least 3",
            coefficients.len()
        ));
    }
    let expected_len = n * n * g;
    let mut all = Vec::with_capacity(coefficients.len());
    for (order, flat) in coefficients.iter().enumerate() {
        if flat.len() != expected_len {
            return Err(format!(
                "mu_coefficients[{order}] has length {}, expected {expected_len}",
                flat.len()
            ));
        }
        let mut by_i = vec![Vec::new(); n];
        for i in 0..n {
            for beta in 0..n {
                for a in 0..g {
                    let value = (flat[((i * n) + beta) * g + a] % p as u64) as u32;
                    if value != 0 {
                        by_i[i].push(MuTerm {
                            beta,
                            a,
                            coeff: value,
                        });
                    }
                }
            }
        }
        all.push(by_i);
    }
    Ok(all)
}

fn build_linear_terms(
    coefficients: &[Vec<u64>],
    output_dim: usize,
    input_dim: usize,
    p: u32,
    label: &str,
) -> Result<Vec<Vec<Vec<LinTerm>>>> {
    if coefficients.len() < 3 {
        return Err(format!(
            "{label} contains {} orders, expected at least 3",
            coefficients.len()
        ));
    }
    let expected_len = output_dim * input_dim;
    let mut all = Vec::with_capacity(coefficients.len());
    for (order, flat) in coefficients.iter().enumerate() {
        if flat.len() != expected_len {
            return Err(format!(
                "{label}[{order}] has length {}, expected {expected_len}",
                flat.len()
            ));
        }
        let mut by_output = vec![Vec::new(); output_dim];
        for out in 0..output_dim {
            for input in 0..input_dim {
                let value = (flat[out * input_dim + input] % p as u64) as u32;
                if value != 0 {
                    by_output[out].push(LinTerm {
                        index: input,
                        coeff: value,
                    });
                }
            }
        }
        all.push(by_output);
    }
    Ok(all)
}

fn enumerate_subsets(weights: &[usize], degree: usize) -> Result<(Vec<u64>, Vec<usize>)> {
    if degree > weights.len() {
        return Err(format!(
            "cannot enumerate {} choose {degree}",
            weights.len()
        ));
    }
    if weights.len() > 63 {
        return Err("subset masks currently require at most 63 local basis elements".to_string());
    }
    if degree == 0 {
        return Ok((vec![0], vec![0]));
    }
    let mut subset: Vec<usize> = (0..degree).collect();
    let mut masks = Vec::new();
    let mut subset_weights = Vec::new();
    loop {
        let mut mask = 0u64;
        let mut total_weight = 0usize;
        for &item in &subset {
            mask |= 1u64 << item;
            total_weight += weights[item];
        }
        masks.push(mask);
        subset_weights.push(total_weight);
        if !next_subset(&mut subset, weights.len()) {
            break;
        }
    }
    Ok((masks, subset_weights))
}

fn next_subset(subset: &mut [usize], n: usize) -> bool {
    let k = subset.len();
    for i in (0..k).rev() {
        if subset[i] < n - k + i {
            subset[i] += 1;
            for j in i + 1..k {
                subset[j] = subset[j - 1] + 1;
            }
            return true;
        }
    }
    false
}

fn build_incidence(
    source_masks: &[u64],
    target_masks: &[u64],
    local_n: usize,
) -> Result<DiffIncidence> {
    let target_rank: HashMap<u64, usize> = target_masks
        .iter()
        .copied()
        .enumerate()
        .map(|(rank, mask)| (mask, rank))
        .collect();
    let mut buckets: Vec<Vec<DiffTerm>> = vec![Vec::new(); target_masks.len()];
    for (source_rank, &source_mask) in source_masks.iter().enumerate() {
        let mut position = 0usize;
        for local_i in 0..local_n {
            if source_mask & (1u64 << local_i) == 0 {
                continue;
            }
            let target_mask = source_mask & !(1u64 << local_i);
            let target = *target_rank
                .get(&target_mask)
                .ok_or_else(|| "internal subset incidence lookup failed".to_string())?;
            buckets[target].push(DiffTerm {
                local_i,
                source_rank,
                neg: position % 2 == 1,
            });
            position += 1;
        }
    }
    let mut offsets = Vec::with_capacity(buckets.len() + 1);
    let mut terms = Vec::new();
    offsets.push(0);
    for bucket in buckets {
        terms.extend(bucket);
        offsets.push(terms.len());
    }
    Ok(DiffIncidence {
        source_count: source_masks.len(),
        target_count: target_masks.len(),
        offsets,
        terms,
    })
}

fn tensor_weights(
    subset_weights: &[usize],
    coeff_weights: &[usize],
    shift: usize,
    r: usize,
) -> Vec<usize> {
    let mut out = Vec::with_capacity(subset_weights.len() * coeff_weights.len());
    for &subset_weight in subset_weights {
        for &coeff_weight in coeff_weights {
            out.push((subset_weight + coeff_weight + shift) % r);
        }
    }
    out
}

fn indices_by_residue(weights: &[usize], r: usize) -> Vec<Vec<usize>> {
    let mut out = vec![Vec::new(); r];
    for (index, &weight) in weights.iter().enumerate() {
        out[weight % r].push(index);
    }
    out
}

fn read_vector_lines(path: &Path, p: u32) -> Result<DenseBlock> {
    let file = File::open(path).map_err(|e| format!("failed to open {}: {e}", path.display()))?;
    let mut reader = BufReader::new(file);
    let mut line = String::new();
    let mut vectors: Vec<Vec<u32>> = Vec::new();
    loop {
        line.clear();
        let bytes = reader
            .read_line(&mut line)
            .map_err(|e| format!("failed to read {}: {e}", path.display()))?;
        if bytes == 0 {
            break;
        }
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let mut vector = Vec::new();
        for token in trimmed.split_whitespace() {
            let value = token
                .parse::<u64>()
                .map_err(|e| format!("failed to parse integer in {}: {e}", path.display()))?;
            vector.push((value % p as u64) as u32);
        }
        vectors.push(vector);
    }
    if vectors.is_empty() {
        return Err(format!("{} contains no vectors", path.display()));
    }
    let rows = vectors[0].len();
    if vectors.iter().any(|vector| vector.len() != rows) {
        return Err(format!(
            "{} has inconsistent vector lengths",
            path.display()
        ));
    }
    let columns = vectors.len();
    let mut values = vec![0u32; rows * columns];
    for (column, vector) in vectors.iter().enumerate() {
        for (row, &value) in vector.iter().enumerate() {
            values[row * columns + column] = value;
        }
    }
    Ok(DenseBlock {
        rows,
        columns,
        values,
    })
}

fn read_dense_block(path: &Path, p: u32) -> Result<DenseBlock> {
    let payload: DenseBlockPayload = read_json_file(path)?;
    if let Some(kind) = &payload.kind
        && kind != DENSE_BLOCK_FORMAT
    {
        return Err(format!("{} has unsupported type {kind}", path.display()));
    }
    if payload.prime != p as u64 {
        return Err(format!(
            "{} prime {} does not match fixture prime {p}",
            path.display(),
            payload.prime
        ));
    }
    if payload.layout != "row-major" {
        return Err(format!(
            "{} has unsupported layout {}",
            path.display(),
            payload.layout
        ));
    }
    if payload.values.len() != payload.rows * payload.columns {
        return Err(format!(
            "{} value count {} does not match {}x{}",
            path.display(),
            payload.values.len(),
            payload.rows,
            payload.columns
        ));
    }
    Ok(DenseBlock {
        rows: payload.rows,
        columns: payload.columns,
        values: payload.values.into_iter().map(|value| value % p).collect(),
    })
}

fn write_dense_block(
    path: &Path,
    block: &DenseBlock,
    p: u32,
    metadata: BTreeMap<String, Value>,
) -> Result<()> {
    if block.values.len() != block.rows * block.columns {
        return Err(format!(
            "dense block value count {} does not match {}x{}",
            block.values.len(),
            block.rows,
            block.columns
        ));
    }
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)
            .map_err(|e| format!("failed to create {}: {e}", parent.display()))?;
    }
    let tmp = path.with_extension(format!(
        "{}.tmp",
        path.extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or("json")
    ));
    let file =
        File::create(&tmp).map_err(|e| format!("failed to create {}: {e}", tmp.display()))?;
    let mut writer = BufWriter::new(file);
    let payload = DenseBlockOutput {
        kind: DENSE_BLOCK_FORMAT,
        prime: p as u64,
        rows: block.rows,
        columns: block.columns,
        layout: "row-major",
        values: &block.values,
        metadata,
    };
    serde_json::to_writer(&mut writer, &payload)
        .map_err(|e| format!("failed to serialize {}: {e}", path.display()))?;
    writer
        .write_all(b"\n")
        .map_err(|e| format!("failed to write {}: {e}", tmp.display()))?;
    writer
        .flush()
        .map_err(|e| format!("failed to flush {}: {e}", tmp.display()))?;
    fs::rename(&tmp, path).map_err(|e| {
        format!(
            "failed to move {} to {}: {e}",
            tmp.display(),
            path.display()
        )
    })
}

fn write_drop_columns(
    path: &Path,
    indices: &[usize],
    p: u32,
    metadata: BTreeMap<String, Value>,
) -> Result<()> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)
            .map_err(|e| format!("failed to create {}: {e}", parent.display()))?;
    }
    let mut sorted = indices.to_vec();
    sorted.sort_unstable();
    let payload = DropColumnsOutput {
        kind: DROP_COLUMNS_FORMAT,
        prime: p as u64,
        indices: sorted,
        metadata,
    };
    write_pretty_json(path, &payload)
}

fn write_pretty_json<T: Serialize>(path: &Path, payload: &T) -> Result<()> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)
            .map_err(|e| format!("failed to create {}: {e}", parent.display()))?;
    }
    let tmp = path.with_extension(format!(
        "{}.tmp",
        path.extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or("json")
    ));
    let file =
        File::create(&tmp).map_err(|e| format!("failed to create {}: {e}", tmp.display()))?;
    let mut writer = BufWriter::new(file);
    serde_json::to_writer_pretty(&mut writer, payload)
        .map_err(|e| format!("failed to serialize {}: {e}", path.display()))?;
    writer
        .write_all(b"\n")
        .map_err(|e| format!("failed to write {}: {e}", tmp.display()))?;
    writer
        .flush()
        .map_err(|e| format!("failed to flush {}: {e}", tmp.display()))?;
    fs::rename(&tmp, path).map_err(|e| {
        format!(
            "failed to move {} to {}: {e}",
            tmp.display(),
            path.display()
        )
    })
}

fn print_pretty_json(value: &Value) -> Result<()> {
    let mut stdout = std::io::stdout().lock();
    serde_json::to_writer_pretty(&mut stdout, value).map_err(|e| e.to_string())?;
    stdout.write_all(b"\n").map_err(|e| e.to_string())
}

fn select_kernel_columns(
    candidates: &DenseBlock,
    report_path: Option<&Path>,
    p: u32,
) -> Result<(DenseBlock, Vec<usize>)> {
    let mut pivots = None;
    if let Some(path) = report_path
        && path.exists()
    {
        let report: KernelReport = read_json_file(path)?;
        if let Some(raw) = report.pivot_columns
            && raw.len() >= 2
        {
            pivots = Some(raw[..2].to_vec());
        }
    }
    let pivots = match pivots {
        Some(pivots) => pivots,
        None => {
            let computed = column_pivots(candidates, p)?;
            if computed.len() < 2 {
                return Err("candidate kernel vector file has rank < 2".to_string());
            }
            computed[..2].to_vec()
        }
    };
    if pivots.iter().any(|&pivot| pivot >= candidates.columns) {
        return Err(format!(
            "candidate pivot columns {:?} are outside 0..{}",
            pivots, candidates.columns
        ));
    }
    let block = select_columns(candidates, &pivots);
    if column_pivots(&block, p)?.len() != 2 {
        return Err("selected kernel columns are not independent".to_string());
    }
    Ok((block, pivots))
}

fn select_columns(block: &DenseBlock, columns: &[usize]) -> DenseBlock {
    let mut values = Vec::with_capacity(block.rows * columns.len());
    for row in 0..block.rows {
        let base = row * block.columns;
        for &column in columns {
            values.push(block.values[base + column]);
        }
    }
    DenseBlock {
        rows: block.rows,
        columns: columns.len(),
        values,
    }
}

fn column_pivots(block: &DenseBlock, p: u32) -> Result<Vec<usize>> {
    let rows = block.rows;
    let cols = block.columns;
    let mut values = block.values.clone();
    let mut pivot_row = 0usize;
    let mut pivots = Vec::new();
    for col in 0..cols {
        let pivot = (pivot_row..rows).find(|&row| values[row * cols + col] != 0);
        let Some(pivot) = pivot else {
            continue;
        };
        if pivot != pivot_row {
            for c in 0..cols {
                values.swap(pivot * cols + c, pivot_row * cols + c);
            }
        }
        let inv = mod_inv(values[pivot_row * cols + col], p)?;
        for c in col..cols {
            let offset = pivot_row * cols + c;
            values[offset] = mul_mod(values[offset], inv, p);
        }
        for row in 0..rows {
            if row == pivot_row {
                continue;
            }
            let factor = values[row * cols + col];
            if factor == 0 {
                continue;
            }
            for c in col..cols {
                let target = row * cols + c;
                let pivot_value = values[pivot_row * cols + c];
                values[target] = sub_mod(values[target], mul_mod(factor, pivot_value, p), p);
            }
        }
        pivots.push(col);
        pivot_row += 1;
        if pivot_row == rows {
            break;
        }
    }
    Ok(pivots)
}

fn choose_independent_rows(two_column_block: &DenseBlock, p: u32) -> Result<Vec<usize>> {
    if two_column_block.columns != 2 {
        return Err(format!(
            "expected a two-column kernel block, got {} columns",
            two_column_block.columns
        ));
    }
    let mut first = None;
    for row in 0..two_column_block.rows {
        let a = two_column_block.values[row * 2];
        let b = two_column_block.values[row * 2 + 1];
        if a != 0 || b != 0 {
            first = Some(row);
            break;
        }
    }
    let first = first.ok_or_else(|| "kernel block is zero".to_string())?;
    let a0 = two_column_block.values[first * 2];
    let a1 = two_column_block.values[first * 2 + 1];
    for row in first + 1..two_column_block.rows {
        let b0 = two_column_block.values[row * 2];
        let b1 = two_column_block.values[row * 2 + 1];
        let det = sub_mod(mul_mod(a0, b1, p), mul_mod(a1, b0, p), p);
        if det != 0 {
            return Ok(vec![first, row]);
        }
    }
    Err("kernel block has rank less than two".to_string())
}

fn normalize_augmented_candidates(
    candidates: &DenseBlock,
    base_cols: usize,
    p: u32,
) -> Result<(DenseBlock, Vec<usize>)> {
    let mut selected = None;
    for i in 0..candidates.columns {
        for j in i + 1..candidates.columns {
            let a = candidates.values[base_cols * candidates.columns + i];
            let b = candidates.values[base_cols * candidates.columns + j];
            let c = candidates.values[(base_cols + 1) * candidates.columns + i];
            let d = candidates.values[(base_cols + 1) * candidates.columns + j];
            let det = sub_mod(mul_mod(a, d, p), mul_mod(b, c, p), p);
            if det != 0 {
                selected = Some((i, j, [a, b, c, d]));
                break;
            }
        }
        if selected.is_some() {
            break;
        }
    }
    let Some((i, j, bottom)) = selected else {
        return Err("augmented candidates have no invertible bottom 2x2 block".to_string());
    };
    let inverse = inverse_2x2(bottom, p)?;
    let mut values = Vec::with_capacity(base_cols * 2);
    for row in 0..base_cols {
        let x = candidates.values[row * candidates.columns + i];
        let y = candidates.values[row * candidates.columns + j];
        values.push(add_mod(
            mul_mod(x, inverse[0], p),
            mul_mod(y, inverse[2], p),
            p,
        ));
        values.push(add_mod(
            mul_mod(x, inverse[1], p),
            mul_mod(y, inverse[3], p),
            p,
        ));
    }
    let bottom00 = add_mod(
        mul_mod(bottom[0], inverse[0], p),
        mul_mod(bottom[1], inverse[2], p),
        p,
    );
    let bottom01 = add_mod(
        mul_mod(bottom[0], inverse[1], p),
        mul_mod(bottom[1], inverse[3], p),
        p,
    );
    let bottom10 = add_mod(
        mul_mod(bottom[2], inverse[0], p),
        mul_mod(bottom[3], inverse[2], p),
        p,
    );
    let bottom11 = add_mod(
        mul_mod(bottom[2], inverse[1], p),
        mul_mod(bottom[3], inverse[3], p),
        p,
    );
    if bottom00 != 1 || bottom01 != 0 || bottom10 != 0 || bottom11 != 1 {
        return Err("normalization failed".to_string());
    }
    Ok((
        DenseBlock {
            rows: base_cols,
            columns: 2,
            values,
        },
        vec![i, j],
    ))
}

fn inverse_2x2(matrix: [u32; 4], p: u32) -> Result<[u32; 4]> {
    let det = sub_mod(
        mul_mod(matrix[0], matrix[3], p),
        mul_mod(matrix[1], matrix[2], p),
        p,
    );
    let det_inv = mod_inv(det, p)?;
    Ok([
        mul_mod(matrix[3], det_inv, p),
        mul_mod(neg_mod(matrix[1], p), det_inv, p),
        mul_mod(neg_mod(matrix[2], p), det_inv, p),
        mul_mod(matrix[0], det_inv, p),
    ])
}

fn cuda_kernel_report_certifies(
    path: Option<&Path>,
    operator: &str,
    sector: usize,
    rows: usize,
    columns: usize,
    p: u32,
) -> Result<(bool, Option<KernelReport>)> {
    let Some(path) = path else {
        return Ok((false, None));
    };
    if !path.exists() {
        return Ok((false, None));
    }
    let report: KernelReport = read_json_file(path)?;
    let ok = report.operator == operator
        && report.sector == sector
        && report.rows == rows
        && report.columns == columns
        && report.prime == p as u64
        && report.candidate_rank.unwrap_or(0) >= 2
        && report
            .product_nonzeros
            .as_ref()
            .is_some_and(|values| !values.is_empty() && values.iter().all(|&value| value == 0));
    Ok((ok, Some(report)))
}

fn dense_file_arg(run: &ManifestRun) -> Result<PathBuf> {
    for pair in run.args.windows(2) {
        if pair[0] == "--dense-file" {
            return Ok(PathBuf::from(&pair[1]));
        }
    }
    Err(format!("{} does not record --dense-file", run.run_id))
}

fn default_nullvectors_path(wdm_path: &str) -> PathBuf {
    PathBuf::from(format!("{wdm_path}_nullvectors_2.txt"))
}

fn default_nullvectors_report_path(wdm_path: &str) -> PathBuf {
    PathBuf::from(format!("{wdm_path}_nullvectors_report.json"))
}

fn parse_sector_list(raw: Option<&str>) -> Result<Vec<usize>> {
    let Some(raw) = raw else {
        return Ok(Vec::new());
    };
    if raw.trim().is_empty() {
        return Ok(Vec::new());
    }
    raw.split(',')
        .map(|part| {
            part.trim()
                .parse::<usize>()
                .map_err(|e| format!("failed to parse sector list {raw:?}: {e}"))
        })
        .collect()
}

fn count_nonzeros(values: &[u32]) -> usize {
    values.par_iter().filter(|&&value| value != 0).count()
}

fn count_difference_nonzeros(left: &DenseBlock, right: &DenseBlock, p: u32) -> Result<usize> {
    if left.rows != right.rows || left.columns != right.columns {
        return Err(format!(
            "block dimensions {}x{} and {}x{} do not match",
            left.rows, left.columns, right.rows, right.columns
        ));
    }
    Ok(left
        .values
        .par_iter()
        .zip(right.values.par_iter())
        .filter(|&(&lhs, &rhs)| sub_mod(lhs, rhs, p) != 0)
        .count())
}

fn path_string(path: &Path) -> String {
    fs::canonicalize(path)
        .unwrap_or_else(|_| path.to_path_buf())
        .display()
        .to_string()
}

#[inline(always)]
fn add_mod(a: u32, b: u32, p: u32) -> u32 {
    let sum = a + b;
    if sum >= p { sum - p } else { sum }
}

#[inline(always)]
fn sub_mod(a: u32, b: u32, p: u32) -> u32 {
    if a >= b { a - b } else { a + p - b }
}

#[inline(always)]
fn neg_mod(value: u32, p: u32) -> u32 {
    if value == 0 { 0 } else { p - value }
}

#[inline(always)]
fn mul_mod(a: u32, b: u32, p: u32) -> u32 {
    ((a as u64 * b as u64) % p as u64) as u32
}

#[inline(always)]
fn add_mul_mod(target: &mut u32, coeff: u32, source: u32, p: u32) {
    if coeff == 0 || source == 0 {
        return;
    }
    *target = add_mod(*target, mul_mod(coeff, source, p), p);
}

fn mod_inv(value: u32, p: u32) -> Result<u32> {
    if value == 0 {
        return Err("attempted to invert zero".to_string());
    }
    Ok(mod_pow(value, p as u64 - 2, p))
}

fn mod_pow(mut base: u32, mut exponent: u64, p: u32) -> u32 {
    let mut acc = 1u32;
    while exponent > 0 {
        if exponent & 1 == 1 {
            acc = mul_mod(acc, base, p);
        }
        base = mul_mod(base, base, p);
        exponent >>= 1;
    }
    acc
}

fn add_vec_in_place(target: &mut [u32], source: &[u32], p: u32) {
    target
        .par_iter_mut()
        .zip(source.par_iter())
        .for_each(|(lhs, &rhs)| *lhs = add_mod(*lhs, rhs, p));
}

fn sub_vec_in_place(target: &mut [u32], source: &[u32], p: u32) {
    target
        .par_iter_mut()
        .zip(source.par_iter())
        .for_each(|(lhs, &rhs)| *lhs = sub_mod(*lhs, rhs, p));
}

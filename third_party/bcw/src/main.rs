//! Consume one completed CUDA WDM file. This command never polls a producer.
use clap::Parser;
use prym_bcw::rank_pipeline::{
    RankComputation, RankComputationOptions, RankComputationStatus, WdmMetadata,
    prepare_wdm_sequence_for_rank, prepared_sequence_len,
};
use prym_bcw::sigma_basis::{ProgressData, save_generator_list};
use prym_bcw::wdm_files::load_wdm_file_sym;
use std::error::Error;
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};

#[derive(Parser)]
#[command(
    name = "prym-rank",
    version,
    about = "Exact rank lower bound from a completed symmetric WDM file"
)]
struct Args {
    /// WDM file written by the CUDA sequence generator.
    filename: PathBuf,
    /// Save the polynomial generator for subsequent CUDA kernel recovery.
    #[arg(short = 'g', long = "generator")]
    generator: bool,
    /// Total number of Rayon worker threads.
    #[arg(short = 't', long = "threads", default_value_t = 4)]
    threads: usize,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    if args.threads == 0 {
        return Err("--threads must be positive".into());
    }
    rayon::ThreadPoolBuilder::new()
        .num_threads(args.threads)
        .build_global()?;

    println!("Loading completed WDM file {}...", args.filename.display());
    let (metadata, seq) = load_rank_input(&args.filename)?;
    println!(
        "File data: prime {}, matrix size {}x{}, num_v {}, usable sequence entries {}.",
        metadata.prime,
        metadata.n_rows,
        metadata.n_cols,
        metadata.num_vectors,
        prepared_sequence_len(&seq)
    );
    let mut computation = RankComputation::new(metadata, RankComputationOptions::default())?;
    if computation.needs_more_sequence(&seq) {
        return Err("completed WDM file contains no usable sequence terms after removing the initial Gram term".into());
    }
    let mut progress = ProgressData::new(computation.max_nlen());
    let step = computation.step(&seq, &mut progress)?;
    println!(
        "\nProcessed {} sequence entries; delta_spread={}, rank_lower_bound={}.",
        step.processed_sequence_len, step.delta_spread, step.rank_lower_bound
    );

    // The producer is already finished. Insufficient input is a failed run, not
    // a reason to wait indefinitely for a WDM update or write a .stop sentinel.
    if computation.rank_lower_bound() > metadata.n_rows.min(metadata.n_cols) as u64 {
        return Err("computed rank lower bound exceeds the matrix dimensions".into());
    }
    match computation.status() {
        RankComputationStatus::Success => {}
        RankComputationStatus::Running | RankComputationStatus::SequenceLimitReached => {
            return Err(format!(
                "completed WDM sequence is insufficient: processed {} terms, delta spread {} (required {}); no rank result was written",
                computation.processed_sequence_len(), computation.delta_spread(),
                RankComputationOptions::default().stopping_threshold
            ).into());
        }
        RankComputationStatus::RankExceedsMatrixDimension => {
            return Err("computed rank lower bound exceeds the matrix dimensions".into());
        }
    }

    println!("Final rank lower bound: {}", computation.rank_lower_bound());
    if args.generator {
        let path = output_path_with_suffix(&args.filename, "_generators.txt");
        println!("Writing generator matrices to {}", path.display());
        save_generator_list(
            computation.basis_matrix(),
            computation.delta(),
            path_to_str(&path)?,
        );
    }
    let result_path = output_path_with_suffix(&args.filename, "_result.txt");
    let mut result = File::create(&result_path)?;
    writeln!(
        result,
        "Matrix size: {} x {}",
        metadata.n_rows, metadata.n_cols
    )?;
    writeln!(result, "Rank: {}", computation.rank_lower_bound())?;
    println!("Wrote {}", result_path.display());
    Ok(())
}

type RankInput = (WdmMetadata, Vec<Vec<u32>>);

fn load_rank_input(path: &Path) -> Result<RankInput, Box<dyn Error>> {
    let mut row_precond = Vec::new();
    let mut col_precond = Vec::new();
    let mut v = Vec::new();
    let mut curv = Vec::new();
    let mut seq = Vec::new();
    let (prime, n_rows, n_cols, num_vectors) = load_wdm_file_sym(
        path_to_str(path)?,
        &mut row_precond,
        &mut col_precond,
        &mut v,
        &mut curv,
        &mut seq,
    )?;
    prepare_wdm_sequence_for_rank(&mut seq)?;
    Ok((
        WdmMetadata {
            prime,
            n_rows,
            n_cols,
            num_vectors,
        },
        seq,
    ))
}

fn output_path_with_suffix(path: &Path, suffix: &str) -> PathBuf {
    let path = path.to_string_lossy();
    let stem = path
        .strip_suffix(".wdm.zst")
        .or_else(|| path.strip_suffix(".wdm"))
        .unwrap_or(&path);
    PathBuf::from(format!("{stem}{suffix}"))
}

fn path_to_str(path: &Path) -> Result<&str, Box<dyn Error>> {
    path.to_str()
        .ok_or_else(|| format!("Path is not UTF-8: {}", path.display()).into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_names_match_cuda_generator_lookup() {
        for input in ["sector.wdm", "sector.wdm.zst"] {
            assert_eq!(
                output_path_with_suffix(Path::new(input), "_generators.txt"),
                PathBuf::from("sector_generators.txt")
            );
            assert_eq!(
                output_path_with_suffix(Path::new(input), "_result.txt"),
                PathBuf::from("sector_result.txt")
            );
        }
    }
}

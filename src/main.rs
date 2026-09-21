//! Internal command-line stages used by the repository's reproduction script.

use clap::{Parser, Subcommand};
use prym_phi::{Result, cyclic, deformation};
use std::path::PathBuf;

#[derive(Parser)]
#[command(
    name = "prym-phi",
    about = "Internal exact algebra stages for Prym–Green reproduction"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Reconstruct and check a frozen cyclic input's small algebra.
    VerifyCyclic {
        #[arg(long)]
        instance: PathBuf,
    },
    /// Extract two independent original-coordinate base kernel columns.
    DeformationExtractBaseKernel {
        #[arg(long)]
        instance: PathBuf,
        #[arg(long)]
        kernel_vectors: PathBuf,
        #[arg(long)]
        kernel_report: Option<PathBuf>,
        #[arg(long)]
        out_dir: PathBuf,
    },
    /// Form F1*K and its two charged target-sector blocks.
    DeformationDeriveRhs {
        #[arg(long)]
        instance: PathBuf,
        #[arg(long)]
        kernel: PathBuf,
        #[arg(long)]
        out_dir: PathBuf,
        #[arg(long)]
        expected_corrections: Option<String>,
    },
    /// Normalize augmented kernels to solutions of F0*S=F1*K.
    DeformationNormalizeAugmented {
        #[arg(long)]
        instance: PathBuf,
        #[arg(long)]
        manifest: PathBuf,
        #[arg(long)]
        out_dir: PathBuf,
    },
    /// Form the quadratic obstruction F2*K-F1*S and replacement pivots.
    DeformationDeriveQuadratic {
        #[arg(long)]
        instance: PathBuf,
        #[arg(long)]
        kernel: PathBuf,
        #[arg(long)]
        solutions_manifest: PathBuf,
        #[arg(long)]
        out_dir: PathBuf,
    },
}

fn run() -> Result<()> {
    match Cli::parse().command {
        Command::VerifyCyclic { instance } => {
            let instance = cyclic::read_cyclic_instance(&instance)?;
            let report = cyclic::verify_cyclic_instance(&instance)?;
            println!(
                "{}",
                serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?
            );
            Ok(())
        }
        Command::DeformationExtractBaseKernel {
            instance,
            kernel_vectors,
            kernel_report,
            out_dir,
        } => deformation::extract_base_kernel(
            &instance,
            &kernel_vectors,
            kernel_report.as_deref(),
            &out_dir,
        ),
        Command::DeformationDeriveRhs {
            instance,
            kernel,
            out_dir,
            expected_corrections,
        } => deformation::derive_rhs(
            &instance,
            &kernel,
            &out_dir,
            expected_corrections.as_deref(),
        ),
        Command::DeformationNormalizeAugmented {
            instance,
            manifest,
            out_dir,
        } => deformation::normalize_augmented(&instance, &manifest, &out_dir),
        Command::DeformationDeriveQuadratic {
            instance,
            kernel,
            solutions_manifest,
            out_dir,
        } => deformation::derive_quadratic(&instance, &kernel, &solutions_manifest, &out_dir),
    }
}

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

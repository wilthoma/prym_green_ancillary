//! Shifted order-basis state and the rank lower bound from symmetric moments.
//! The initial V^T V term is discarded before processing; this convention is
//! shared with the CUDA producer and polynomial-generator recovery.

use crate::poly_mat_mul::{is_prime_valid_ntt, poly_mat_mul_red_adaptive};
use crate::sigma_basis::{
    ProgressReporter, analyze_delta, pm_basis, process_input_sequence, shift_trunc_in, unit_mat,
};
use std::cmp::min;
use std::error::Error;
use std::fmt;

pub type PolynomialMatrix = Vec<Vec<Vec<u64>>>;

pub const DEFAULT_PM_BASIS_STOPPING_THRESHOLD: u64 = 10;
pub const DEFAULT_MAX_NLEN_BUFFER: usize = 40;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WdmMetadata {
    pub prime: u32,
    pub n_rows: usize,
    pub n_cols: usize,
    pub num_vectors: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RankComputationOptions {
    pub stopping_threshold: u64,
    pub max_nlen_buffer: usize,
}

impl Default for RankComputationOptions {
    fn default() -> Self {
        Self {
            stopping_threshold: DEFAULT_PM_BASIS_STOPPING_THRESHOLD,
            max_nlen_buffer: DEFAULT_MAX_NLEN_BUFFER,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RankComputationStatus {
    Running,
    Success,
    RankExceedsMatrixDimension,
    SequenceLimitReached,
}

impl RankComputationStatus {
    pub fn is_running(self) -> bool {
        matches!(self, Self::Running)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RankStep {
    pub processed_sequence_len: usize,
    pub newly_processed_terms: usize,
    pub delta_spread: u64,
    pub rank_lower_bound: u64,
    pub status: RankComputationStatus,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RankComputationError {
    InvalidMetadata(String),
    InvalidSequence(String),
    PrimeInvalidForNtt {
        prime: u64,
        max_nlen: usize,
        num_vectors: usize,
    },
    NeedMoreSequence {
        available_terms: usize,
        processed_terms: usize,
    },
}

impl fmt::Display for RankComputationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidMetadata(message) => write!(f, "invalid WDM metadata: {message}"),
            Self::InvalidSequence(message) => write!(f, "invalid Wiedemann sequence: {message}"),
            Self::PrimeInvalidForNtt {
                prime,
                max_nlen,
                num_vectors,
            } => write!(
                f,
                "prime {prime} is not valid for NTT with max_nlen {max_nlen} and {num_vectors} vectors"
            ),
            Self::NeedMoreSequence {
                available_terms,
                processed_terms,
            } => write!(
                f,
                "need more Wiedemann sequence terms: available {available_terms}, already processed {processed_terms}"
            ),
        }
    }
}

impl Error for RankComputationError {}

pub struct RankComputation {
    metadata: WdmMetadata,
    options: RankComputationOptions,
    prime: u64,
    max_nlen: usize,
    delta: Vec<u64>,
    basis_matrix: PolynomialMatrix,
    processed_sequence_len: usize,
    delta_spread: u64,
    rank_lower_bound: u64,
    last_input_sequence_matrix: Option<PolynomialMatrix>,
}

impl RankComputation {
    pub fn new(
        metadata: WdmMetadata,
        options: RankComputationOptions,
    ) -> Result<Self, RankComputationError> {
        validate_metadata(metadata)?;
        let max_nlen = max_nlen(metadata, options.max_nlen_buffer);
        let prime = metadata.prime as u64;
        if !is_prime_valid_ntt::<u64>(prime, max_nlen, metadata.num_vectors) {
            return Err(RankComputationError::PrimeInvalidForNtt {
                prime,
                max_nlen,
                num_vectors: metadata.num_vectors,
            });
        }

        let num_vectors = metadata.num_vectors;
        let delta = std::iter::repeat_n(0, num_vectors)
            .chain(std::iter::repeat_n(1, num_vectors))
            .collect();

        Ok(Self {
            metadata,
            options,
            prime,
            max_nlen,
            delta,
            basis_matrix: unit_mat(2 * num_vectors),
            processed_sequence_len: 0,
            delta_spread: 0,
            rank_lower_bound: 0,
            last_input_sequence_matrix: None,
        })
    }

    pub fn step<P: ProgressReporter + ?Sized>(
        &mut self,
        prepared_sequence: &[Vec<u32>],
        progress: &mut P,
    ) -> Result<RankStep, RankComputationError> {
        validate_prepared_sequence(prepared_sequence, self.metadata.num_vectors)?;
        let available_sequence_len = prepared_sequence_len(prepared_sequence);
        if available_sequence_len <= self.processed_sequence_len {
            return Err(RankComputationError::NeedMoreSequence {
                available_terms: available_sequence_len,
                processed_terms: self.processed_sequence_len,
            });
        }

        let (g, num_vectors, sequence_len) = process_input_sequence(prepared_sequence);
        if num_vectors != self.metadata.num_vectors {
            return Err(RankComputationError::InvalidSequence(format!(
                "sequence encodes {num_vectors} vectors, but WDM metadata says {}",
                self.metadata.num_vectors
            )));
        }
        if sequence_len < self.processed_sequence_len {
            return Err(RankComputationError::InvalidSequence(format!(
                "sequence length went backwards from {} to {sequence_len}",
                self.processed_sequence_len
            )));
        }

        let newly_processed_terms = sequence_len - self.processed_sequence_len;
        if newly_processed_terms == 0 {
            return Err(RankComputationError::NeedMoreSequence {
                available_terms: sequence_len,
                processed_terms: self.processed_sequence_len,
            });
        }

        let mut shifted_product = poly_mat_mul_red_adaptive(
            &self.basis_matrix,
            &g,
            self.prime,
            0,
            self.processed_sequence_len + newly_processed_terms,
        );
        shift_trunc_in(
            &mut shifted_product,
            self.processed_sequence_len,
            newly_processed_terms,
        );

        let (basis_update, new_delta) = pm_basis(
            &shifted_product,
            newly_processed_terms,
            &self.delta,
            self.prime,
            progress,
        );
        self.delta = new_delta;

        (self.delta_spread, self.rank_lower_bound) = analyze_delta(&self.delta);

        self.basis_matrix = poly_mat_mul_red_adaptive(
            &basis_update,
            &self.basis_matrix,
            self.prime,
            0,
            self.basis_matrix[0][0].len(),
        );
        self.processed_sequence_len += newly_processed_terms;
        self.last_input_sequence_matrix = Some(g);

        Ok(RankStep {
            processed_sequence_len: self.processed_sequence_len,
            newly_processed_terms,
            delta_spread: self.delta_spread,
            rank_lower_bound: self.rank_lower_bound,
            status: self.status(),
        })
    }

    pub fn needs_more_sequence(&self, prepared_sequence: &[Vec<u32>]) -> bool {
        prepared_sequence_len(prepared_sequence) <= self.processed_sequence_len
    }

    pub fn status(&self) -> RankComputationStatus {
        if self.delta_spread >= self.options.stopping_threshold {
            RankComputationStatus::Success
        } else if self.rank_lower_bound > self.min_dimension() as u64 {
            RankComputationStatus::RankExceedsMatrixDimension
        } else if self.processed_sequence_len >= self.max_nlen {
            RankComputationStatus::SequenceLimitReached
        } else {
            RankComputationStatus::Running
        }
    }

    pub fn metadata(&self) -> WdmMetadata {
        self.metadata
    }

    pub fn prime(&self) -> u64 {
        self.prime
    }

    pub fn max_nlen(&self) -> usize {
        self.max_nlen
    }

    pub fn processed_sequence_len(&self) -> usize {
        self.processed_sequence_len
    }

    pub fn delta_spread(&self) -> u64 {
        self.delta_spread
    }

    pub fn rank_lower_bound(&self) -> u64 {
        self.rank_lower_bound
    }

    pub fn delta(&self) -> &Vec<u64> {
        &self.delta
    }

    pub fn basis_matrix(&self) -> &PolynomialMatrix {
        &self.basis_matrix
    }

    pub fn last_input_sequence_matrix(&self) -> Option<&PolynomialMatrix> {
        self.last_input_sequence_matrix.as_ref()
    }

    pub fn min_dimension(&self) -> usize {
        min(self.metadata.n_rows, self.metadata.n_cols)
    }
}

pub fn prepare_wdm_sequence_for_rank(seq: &mut [Vec<u32>]) -> Result<(), RankComputationError> {
    for sequence in seq.iter_mut() {
        if sequence.is_empty() {
            return Err(RankComputationError::InvalidSequence(
                "WDM sequence rows must contain the initial Gram entry".to_string(),
            ));
        }
        sequence.remove(0);
    }
    Ok(())
}

pub fn prepared_sequence_len(seq: &[Vec<u32>]) -> usize {
    seq.first().map_or(0, Vec::len)
}

pub fn max_nlen(metadata: WdmMetadata, buffer: usize) -> usize {
    (2 * min(metadata.n_rows, metadata.n_cols)) / metadata.num_vectors + buffer
}

fn validate_metadata(metadata: WdmMetadata) -> Result<(), RankComputationError> {
    if metadata.prime <= 1 {
        return Err(RankComputationError::InvalidMetadata(format!(
            "prime must be greater than 1, got {}",
            metadata.prime
        )));
    }
    if metadata.num_vectors == 0 {
        return Err(RankComputationError::InvalidMetadata(
            "num_vectors must be greater than 0".to_string(),
        ));
    }
    if metadata.n_rows == 0 || metadata.n_cols == 0 {
        return Err(RankComputationError::InvalidMetadata(format!(
            "matrix dimensions must be nonzero, got {}x{}",
            metadata.n_rows, metadata.n_cols
        )));
    }
    Ok(())
}

fn validate_prepared_sequence(
    seq: &[Vec<u32>],
    num_vectors: usize,
) -> Result<(), RankComputationError> {
    let expected_entries = num_vectors * (num_vectors + 1) / 2;
    if seq.len() != expected_entries {
        return Err(RankComputationError::InvalidSequence(format!(
            "expected {expected_entries} upper-triangular entries for {num_vectors} vectors, got {}",
            seq.len()
        )));
    }
    if seq.is_empty() {
        return Err(RankComputationError::InvalidSequence(
            "sequence must contain at least one upper-triangular entry".to_string(),
        ));
    }

    let len = seq[0].len();
    for (idx, row) in seq.iter().enumerate() {
        if row.len() != len {
            return Err(RankComputationError::InvalidSequence(format!(
                "sequence row {idx} has length {}, expected {len}",
                row.len()
            )));
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sigma_basis::{NoProgress, ProgressReporter};

    struct CountingProgress {
        ticks: usize,
    }

    impl ProgressReporter for CountingProgress {
        fn progress_tick(&mut self) {
            self.ticks += 1;
        }
    }

    #[test]
    fn prepare_sequence_removes_initial_gram_entry() {
        let mut seq = vec![vec![9, 1, 2], vec![8, 3, 4], vec![7, 5, 6]];

        prepare_wdm_sequence_for_rank(&mut seq).unwrap();

        assert_eq!(seq, vec![vec![1, 2], vec![3, 4], vec![5, 6]]);
        assert_eq!(prepared_sequence_len(&seq), 2);
    }

    #[test]
    fn computation_reports_need_more_for_initial_only_sequence() {
        let metadata = WdmMetadata {
            prime: 29,
            n_rows: 2,
            n_cols: 2,
            num_vectors: 1,
        };
        let mut computation =
            RankComputation::new(metadata, RankComputationOptions::default()).unwrap();
        let mut progress = NoProgress;
        let seq = vec![Vec::new()];

        let err = computation.step(&seq, &mut progress).unwrap_err();

        assert_eq!(
            err,
            RankComputationError::NeedMoreSequence {
                available_terms: 0,
                processed_terms: 0,
            }
        );
    }

    #[test]
    fn scalar_constant_sequence_computes_rank_lower_bound() {
        let metadata = WdmMetadata {
            prime: 29,
            n_rows: 2,
            n_cols: 2,
            num_vectors: 1,
        };
        let options = RankComputationOptions {
            stopping_threshold: 1,
            max_nlen_buffer: 4,
        };
        let mut computation = RankComputation::new(metadata, options).unwrap();
        let mut progress = NoProgress;
        let seq = vec![vec![2, 2, 2, 2, 2]];

        let step = computation.step(&seq, &mut progress).unwrap();

        assert_eq!(step.status, RankComputationStatus::Success);
        assert_eq!(computation.rank_lower_bound(), 1);
        assert!(computation.delta_spread() >= options.stopping_threshold);
        assert!(computation.last_input_sequence_matrix().is_some());
    }

    #[test]
    fn computation_uses_injected_progress_reporter() {
        let metadata = WdmMetadata {
            prime: 29,
            n_rows: 2,
            n_cols: 2,
            num_vectors: 1,
        };
        let mut computation = RankComputation::new(
            metadata,
            RankComputationOptions {
                stopping_threshold: 1,
                max_nlen_buffer: 4,
            },
        )
        .unwrap();
        let mut progress = CountingProgress { ticks: 0 };
        let seq = vec![vec![2, 2, 2]];

        computation.step(&seq, &mut progress).unwrap();

        assert_eq!(progress.ticks, 3);
    }
}

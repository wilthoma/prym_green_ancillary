//! Divide-and-conquer shifted order bases and CUDA-compatible generators.
//! Polynomial matrices use [row][column][degree]; generator output preserves
//! the upstream reversal and transpose conventions.

// Preserve the upstream nested-Vec API and explicit row/column/degree indices.
// These conventions make the extracted algebra directly comparable to its source.
#![allow(clippy::ptr_arg, clippy::needless_range_loop)]

use crate::modular_linalg::{lsp_decomposition, matrix_inverse};
use crate::poly_mat_mul::poly_mat_mul_red_adaptive;
use nalgebra::DMatrix;
use std::{io::Write, vec};

pub trait ProgressReporter {
    fn progress_tick(&mut self);
}

pub struct NoProgress;

impl ProgressReporter for NoProgress {
    #[inline(always)]
    fn progress_tick(&mut self) {}
}

/*
Inputs:
- g: input polynomial matrix
- prime: we compute everything modulo p

Output:
- mat: sigma-basis m[:][:][0] + m[:][:][1]*x + m[:][:][2]*x^2 + ...
    for a polynomial-matrix construced from the matrix sequence seq such that it encodes the rank of the original matrix A
    m = poly_mat.len()
    n = poly_mat[0].len
- delta: encodes "degrees" of the rows of the sigma-basis (?)
*/
pub fn pm_basis<P: ProgressReporter + ?Sized>(
    g: &Vec<Vec<Vec<u64>>>,
    d: usize,
    delta: &Vec<u64>,
    prime: u64,
    progress: &mut P,
) -> (Vec<Vec<Vec<u64>>>, Vec<u64>) {
    let n = g.len();

    if d == 0 {
        (unit_mat(n), delta.clone())
    } else if d == 1 {
        progress.progress_tick();
        basis(g, delta, prime)
        //m_basis(g, d, delta, prime, progress)
    } else {
        let d_ = d / 2; // d'
        let d__ = d - d_; // d''
        assert!(d_ >= 1 && d__ >= 1);

        // M', delta' = BlockSigmaBasis(G, d')
        let (mat_, delta_) = pm_basis(g, d_, delta, prime, progress);

        // G' = x^(-d') M * G  mod x^(d'')
        let mut g_ = poly_mat_mul_red_adaptive(&mat_, g, prime, 0, d);
        shift_trunc_in(&mut g_, d_, d__);

        // M'', delta'' = BlockSigmaBasis(G', d'')
        let (mat__, delta__) = pm_basis(&g_, d__, &delta_, prime, progress);

        // M = M'' * M'
        (
            poly_mat_mul_red_adaptive(&mat__, &mat_, prime, 0, mat_[0][0].len()),
            delta__,
        )
    }
}

pub fn process_input_sequence(seq: &[Vec<u32>]) -> (Vec<Vec<Vec<u64>>>, usize, usize) {
    // find dimension (num_v) of the elements of the Krylov-sequence using the condition: num_v * (num_v + 1) / 2 == seq[0].len()
    let num_entries = seq.len(); // number of integer entries per Krylov-sequence element
    let seq_len = seq[0].len(); // length of the Krylov-sequence
    let mut num_v = 0; // dimension of
    for i in 0..=num_entries {
        if i * (i + 1) / 2 == num_entries {
            num_v = i;
        }
    }
    assert!(
        num_v * (num_v + 1) / 2 == num_entries,
        "number of integer entries does not match square matrix size"
    );

    // compute largest d for the given sequence length
    // the sequence length has already been chosen minimally with some buffer

    // prepare input data. Also add a unit matrix of size nxn below the matrix
    let mut g = vec![vec![vec![0; seq_len]; num_v]; 2 * num_v];
    let mut ii = 0;
    for i in 0..num_v {
        for j in i..num_v {
            for k in 0..seq_len {
                g[i][j][k] = seq[ii][k].into();
                g[j][i][k] = g[i][j][k];
            }
            ii += 1;
        }
    }
    for i in 0..num_v {
        g[num_v + i][i][0] = 1;
    }

    (g, num_v, seq_len)
}

pub fn order_basis_to_generator_matrixlist(
    basis: &Vec<Vec<Vec<u64>>>,
    delta: &Vec<u64>,
) -> Vec<DMatrix<u64>> {
    // Reverse each row of the upper-left block using its shifted degree.
    let n = basis.len();
    assert_eq!(n, basis[0].len(), "Matrix of basis vectors must be square");
    let seq_len = basis[0][0].len();
    let m = n / 2;
    let mut basisrev = vec![vec![vec![0; seq_len]; m]; m];
    let mut maxdel = 0;
    for i in 0..m {
        let del = delta[i] as usize;
        if del > maxdel {
            maxdel = del;
        }
        for j in 0..m {
            for k in 0..=del {
                basisrev[i][j][k] = basis[i][j][del - k];
            }
        }
    }

    // Note that we transpose to get the minimal matrix generator as defined in the Skipness-Thesis
    let mut min_mat_gen = vec![DMatrix::zeros(m, m); maxdel + 1];
    let mut maxdeg = 0;
    for i in 0..m {
        for j in 0..m {
            for k in 0..=maxdel {
                min_mat_gen[k][(j, i)] = basisrev[i][j][k];
                if basisrev[j][i][k] != 0 && k > maxdeg {
                    maxdeg = k;
                }
            }
        }
    }
    min_mat_gen.resize(maxdeg + 1, DMatrix::zeros(m, m));
    min_mat_gen
}

pub fn save_generator_list(basis: &Vec<Vec<Vec<u64>>>, delta: &Vec<u64>, filepath: &str) {
    // extract reversed basis for original sequence
    let min_mat_gen = order_basis_to_generator_matrixlist(basis, delta);

    // save to file. Every line of the file is one matrix in the sequence.
    // Every matrix is stored row-wise, i.e., first row 0, then row 1, etc, with all entries separated by spaces.
    let mut file = std::fs::File::create(filepath).expect("Failed to create file.");
    let m = min_mat_gen[0].nrows();
    for k in 0..min_mat_gen.len() {
        for i in 0..m {
            for j in 0..m {
                // Note: since the min_mat_gen computation was transposed, we again transpose when writing to keep it compatible downstream
                write!(file, "{} ", min_mat_gen[k][(j, i)]).expect("Failed to write to file.");
            }
        }
        writeln!(file).expect("Failed to write to file.");
    }
}

pub struct ProgressData {
    total: usize,
    current: usize,
    start_time: std::time::Instant,
}
impl ProgressData {
    pub fn new(total: usize) -> Self {
        ProgressData {
            total,
            current: 0,
            start_time: std::time::Instant::now(),
        }
    }
}

impl ProgressReporter for ProgressData {
    #[inline(always)]
    fn progress_tick(&mut self) {
        self.current += 1;
        if self.current.is_multiple_of(100) {
            let elapsed_time = self.start_time.elapsed();
            let percent = (self.current as f64 / self.total as f64) * 100.0;
            print!(
                "\rSigma basis progress: {:.2}% ({} of {}), Time elapsed: {:?}    ",
                percent, self.current, self.total, elapsed_time
            );
            std::io::stdout().flush().unwrap();
        }
    }
}

// creates unit (polynomial) matrix
pub fn unit_mat(n: usize) -> Vec<Vec<Vec<u64>>> {
    let mut mat = vec![vec![vec![0]; n]; n];
    for i in 0..n {
        mat[i][i][0] = 1;
    }
    mat
}

/*
Input: matrix-polynomial M = mat[:][:][0] + mat[:][:][1]*x^1 + mat[:][:][2]*x^2 ...
Output: ( x^(-shiftd) * M ) mod x^truncd

Example:
shiftd = 1
tuncd = 2
1 + 2x + 3x^2 + 4x^3 --> 2 + 3x
*/
pub fn shift_trunc_in(mat: &mut Vec<Vec<Vec<u64>>>, shiftd: usize, truncd: usize) {
    let m = mat.len();
    let n = mat[0].len();
    for i in 0..m {
        for j in 0..n {
            for k in 0..truncd {
                mat[i][j][k] = mat[i][j][k + shiftd];
            }
            mat[i][j].resize(truncd, 0);
        }
    }
}

pub fn basis(
    g: &Vec<Vec<Vec<u64>>>,
    delta: &Vec<u64>,
    prime: u64,
) -> (Vec<Vec<Vec<u64>>>, Vec<u64>) {
    //println!("running basis -----");
    let m = g.len();
    assert!(m > 0, "Must have #rows > 0");
    let n = g[0].len();
    assert!(m >= n, "Must have #rows >= #cols");
    let mut delta = delta.clone();

    // sort delta in descending order and remember the permutation
    let mut perm = (0..m).collect::<Vec<_>>();
    //perm.sort_by(|&a, &b| delta[a].cmp(&delta[b]).reverse());
    //delta.sort_by(|a, b| b.cmp(a));
    perm.sort_by(|&a, &b| delta[a].cmp(&delta[b]));
    delta.sort();
    //println!("{:?}", perm);
    //println!("{:?}", delta);

    let mut pi_m = DMatrix::zeros(m, m);
    for i in 0..m {
        pi_m[(i, perm[i])] = 1;
    }
    // let mut Gk_inv = pi_m.clone();
    // Gk_inv = Gk_inv.try_inverse().unwrap();

    // delta.sort_by(|a, b| b.cmp(a));

    // let Delta0 = Gk_inv * M * (G[k-1].clone());
    let mut delta_matrix = DMatrix::from_fn(m, n, |i, j| g[i][j][0]);
    delta_matrix = &pi_m * &delta_matrix;
    let mut delta_aug = DMatrix::zeros(m, m);
    for i in 0..m {
        for j in 0..n {
            delta_aug[(i, j)] = delta_matrix[(i, j)];
        }
    }

    //println!("{:?}", delta_matrix);
    //println!("{:?}", delta_aug);

    let (l, s, _p) = lsp_decomposition(&delta_aug, prime);
    let linv = matrix_inverse(&l, prime);

    // D = D2 + x Dx
    let mut d1 = DMatrix::zeros(m, m);
    let mut dx = DMatrix::zeros(m, m);
    for i in 0..m {
        if s.row(i).iter().all(|&x| x == 0) {
            d1[(i, i)] = 1;
        } else {
            dx[(i, i)] = 1;
        }
    }

    let m1 = &d1 * &linv * &pi_m;
    let mx = &dx * &linv * &pi_m;

    // fill output again in vects
    let mut ret = vec![vec![vec![0; 2]; m]; m];
    for i in 0..m {
        for j in 0..m {
            ret[i][j][0] = m1[(i, j)];
            ret[i][j][1] = mx[(i, j)];
        }
    }

    for i in 0..m {
        // delta[i] += Dx[(i,i)] as i128;
        //delta[i] -= if dx[(i,i)] == 1 { 1 } else { 0 };
        delta[i] += if dx[(i, i)] == 1 { 1 } else { 0 };
    }

    (ret, delta)
}

// orders the values of delta and computes the distance between the smaller half and the bigger half
pub fn analyze_delta(delta: &Vec<u64>) -> (u64, u64) {
    // sort delta in ascending order
    let mut delta = delta.clone();
    delta.sort_by(|&a, &b| a.cmp(&b));

    // compute distance between blocks
    let len = delta.len();
    assert!(len.is_multiple_of(2));
    assert!(delta[len / 2] >= delta[len / 2 - 1]);
    let dist = delta[len / 2] - delta[len / 2 - 1];
    let rank_lower_bound = delta.iter().take(len / 2).sum();
    (dist, rank_lower_bound)
}

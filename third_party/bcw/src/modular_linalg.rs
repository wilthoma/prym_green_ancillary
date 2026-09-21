//! Exact small-matrix inversion and LSP decomposition over the finite field.

use crate::ntt::{NTTInteger, modinv};
use nalgebra::DMatrix;

/// Computes the modular inverse of a matrix modulo `p`.
/// Panics if the matrix is not invertible.
pub fn matrix_inverse<T: NTTInteger>(mat: &DMatrix<T>, p: T) -> DMatrix<T> {
    let mat = mat.clone();
    let n = mat.nrows();
    assert_eq!(n, mat.ncols(), "Matrix must be square");

    // Augment the matrix with the identity matrix
    let mut aug = DMatrix::zeros(n, 2 * n);
    for i in 0..n {
        for j in 0..n {
            aug[(i, j)] = mat[(i, j)] % p;
        }
        aug[(i, n + i)] = T::one();
    }

    // Perform Gaussian elimination
    for i in 0..n {
        // Find pivot
        if aug[(i, i)] == T::zero() {
            // Try to swap with a lower row
            let mut found = false;
            for j in (i + 1)..n {
                if aug[(j, i)] != T::zero() {
                    aug.swap_rows(i, j);
                    found = true;
                    break;
                }
            }
            if !found {
                panic!("Matrix is singular and not invertible");
            }
        }

        // Normalize pivot row
        let inv = modinv(aug[(i, i)], p);
        for k in 0..2 * n {
            aug[(i, k)] = (aug[(i, k)] * inv) % p;
        }

        // Eliminate other rows
        for j in 0..n {
            if j != i {
                let factor = aug[(j, i)];
                for k in 0..2 * n {
                    let tmp = (aug[(i, k)] * factor) % p;
                    aug[(j, k)] = (aug[(j, k)] + p - tmp) % p;
                }
            }
        }
    }

    // Extract inverse matrix
    let mut inv_mat = DMatrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            inv_mat[(i, j)] = aug[(i, n + j)];
        }
    }

    inv_mat
}

/// Computes the LSP decomposition of a matrix modulo p.
/// LSP decomposition: A = L * S * P (mod p)
pub fn lsp_decomposition<T: NTTInteger>(
    a: &DMatrix<T>,
    p: T,
) -> (DMatrix<T>, DMatrix<T>, DMatrix<T>) {
    let mut a = a.clone();
    let m = a.nrows();
    assert_eq!(m, a.ncols(), "Matrix must be square");

    let mut l = DMatrix::<T>::identity(m, m);
    let mut perm = (0..m).collect::<Vec<usize>>();

    let mut rank = 0;

    for k in 0..m {
        // Find pivot in row k (column-wise search)
        let mut pivot_col = None;
        for j in rank..m {
            if a[(k, j)] % p != T::zero() {
                pivot_col = Some(j);
                break;
            }
        }

        if let Some(j) = pivot_col {
            // Swap columns k <-> j in A and perm
            a.swap_columns(rank, j);
            perm.swap(rank, j);

            // Eliminate below
            let pivot_inv = modinv(a[(k, rank)], p);
            for i in (k + 1)..m {
                let factor = (a[(i, rank)] * pivot_inv) % p;
                l[(i, k)] = factor;
                for col in rank..m {
                    let sub = (factor * a[(k, col)]) % p;
                    a[(i, col)] = (a[(i, col)] + p - sub) % p;
                }
            }

            rank += 1;
        }
    }

    // Build the permutation matrix P
    let mut pmat = DMatrix::<T>::zeros(m, m);
    for (i, &j) in perm.iter().enumerate() {
        // pmat[(i, j)] = 1;
        pmat[(j, i)] = T::one();
    }

    // Final modular reduction
    let s = a.map(|x| x % p);

    (l, s, pmat)
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DMatrix;

    fn is_unit_lower_triangular(l: &DMatrix<u128>, p: u128) -> bool {
        let n = l.nrows();
        for i in 0..n {
            for j in 0..n {
                let val = l[(i, j)] % p;
                if (i == j && val != 1) || (i < j && val != 0) {
                    return false;
                }
            }
        }
        true
    }

    fn is_valid_s_matrix(s: &DMatrix<u128>, p: u128) -> bool {
        let m = s.nrows();

        // Count nonzero rows from the top
        let mut rank = 0;
        for i in 0..m {
            // check that all entries to the left are zero
            for j in 0..rank {
                if !s[(i, j)].is_multiple_of(p) {
                    return false;
                }
            }
            // if row is nonzero, then rank entry must be nonzero
            if s.row(i).iter().any(|&x| x % p != 0) {
                if s[(i, rank)].is_multiple_of(p) {
                    return false;
                }

                rank += 1;
            }
        }

        true
    }

    fn check_lsp(a: DMatrix<u128>, p: u128) {
        let (l, s, pmat) = lsp_decomposition(&a, p);

        // println!("{} {} {} {}", a,l,s,pmat);

        let reconstructed = (l.clone() * s.clone() * pmat.clone()).map(|x| x % p);
        let original_modp = a.map(|x| x % p);

        assert_eq!(
            original_modp, reconstructed,
            "Reconstructed matrix does not match original"
        );
        assert!(
            is_unit_lower_triangular(&l, p),
            "L is not unit lower triangular"
        );
        assert!(is_valid_s_matrix(&s, p), "S matrix is not in expected form");
    }

    #[test]
    fn test_lsp_decomposition_small() {
        let p = 101;
        let a = DMatrix::<u128>::from_row_slice(3, 3, &[2, 4, 6, 1, 3, 5, 0, 0, 1]);
        check_lsp(a, p);
    }

    #[test]
    fn test_lsp_decomposition_larger() {
        let p = 97;
        let a = DMatrix::<u128>::from_row_slice(
            5,
            5,
            &[
                10, 20, 30, 40, 50, 5, 10, 15, 20, 25, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 99, 99, 99,
                99, 99, // mod 97 => 2
            ],
        );
        check_lsp(a, p);
    }

    fn is_identity_mod(mat: &DMatrix<u128>, p: u128) -> bool {
        let n = mat.nrows();
        if mat.ncols() != n {
            return false;
        }
        for i in 0..n {
            for j in 0..n {
                let val = mat[(i, j)] % p;
                if (i == j && val != 1) || (i != j && val != 0) {
                    return false;
                }
            }
        }
        true
    }

    #[test]
    fn test_modular_inverse_small_matrix() {
        let p = 97;
        let a = DMatrix::<u128>::from_row_slice(3, 3, &[2, 3, 1, 1, 1, 1, 3, 5, 2]);

        let inv = matrix_inverse(&a, p);
        let product = (&a * &inv).map(|x| x % p);

        assert!(is_identity_mod(&product, p), "A * A⁻¹ != I mod p");
    }

    #[test]
    fn test_modular_inverse_identity() {
        let p = 97;
        let a = DMatrix::<u128>::identity(4, 4);
        let inv = matrix_inverse(&a, p);
        assert_eq!(inv, a, "Inverse of identity should be identity");
    }

    #[test]
    fn test_modinv_basic() {
        let p = 97u64;

        for a in 1..p {
            let inv = modinv(a, p);
            assert_eq!((a * inv) % p, 1, "modinv({a}, {p}) * {a} != 1 mod {p}");
        }
    }
}

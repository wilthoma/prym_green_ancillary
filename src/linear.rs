//! Deterministic finite-field row reduction and reusable column-coordinate solves.
//! Leftmost pivots fix bases reproducibly; coordinate solves verify their residuals.

// Explicit indices and argument lists follow the mathematical matrix identities.
#![allow(clippy::needless_range_loop)]

use serde::{Deserialize, Serialize};

use crate::{Result, field::Field};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Rref {
    pub matrix: Vec<Vec<u64>>,
    pub pivot_cols: Vec<usize>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ColumnSolver {
    row_count: usize,
    col_count: usize,
    pivot_rows: Vec<usize>,
    inverse: Vec<Vec<u64>>,
}

impl ColumnSolver {
    pub fn new(columns: &[Vec<u64>], field: Field) -> Result<Self> {
        if columns.is_empty() {
            return Err("cannot build a column solver with no columns".to_string());
        }
        let row_count = columns[0].len();
        if row_count == 0 {
            return Err("columns must be nonempty".to_string());
        }
        if columns.iter().any(|c| c.len() != row_count) {
            return Err("columns have inconsistent lengths".to_string());
        }

        let col_count = columns.len();
        let transpose: Vec<Vec<u64>> = columns.to_vec();
        let rref = rref(transpose, field)?;
        if rref.pivot_cols.len() != col_count {
            return Err(format!(
                "column matrix has rank {}, expected full column rank {col_count}",
                rref.pivot_cols.len()
            ));
        }

        let pivot_rows = rref.pivot_cols;
        let mut square = vec![vec![0; col_count]; col_count];
        for (i, &row_idx) in pivot_rows.iter().enumerate() {
            for j in 0..col_count {
                square[i][j] = columns[j][row_idx];
            }
        }
        let inverse = invert_square(square, field)?;
        Ok(Self {
            row_count,
            col_count,
            pivot_rows,
            inverse,
        })
    }

    pub fn solve(&self, rhs: &[u64], field: Field) -> Result<Vec<u64>> {
        if rhs.len() != self.row_count {
            return Err(format!(
                "rhs length {}, expected {}",
                rhs.len(),
                self.row_count
            ));
        }
        let mut projected = vec![0; self.col_count];
        for (i, &row_idx) in self.pivot_rows.iter().enumerate() {
            projected[i] = rhs[row_idx];
        }
        Ok(mat_vec_mul(&self.inverse, &projected, field))
    }

    pub fn solve_and_verify(
        &self,
        columns: &[Vec<u64>],
        rhs: &[u64],
        field: Field,
    ) -> Result<Vec<u64>> {
        let solution = self.solve(rhs, field)?;
        let reconstructed = linear_combination(columns, &solution, field)?;
        if reconstructed != rhs {
            return Err("rhs is not in the span of the column matrix".to_string());
        }
        Ok(solution)
    }
}

pub fn rref(mut matrix: Vec<Vec<u64>>, field: Field) -> Result<Rref> {
    if matrix.is_empty() {
        return Ok(Rref {
            matrix,
            pivot_cols: Vec::new(),
        });
    }
    let cols = matrix[0].len();
    if matrix.iter().any(|row| row.len() != cols) {
        return Err("matrix rows have inconsistent lengths".to_string());
    }

    let mut pivot_cols = Vec::new();
    let mut pivot_row = 0;
    for col in 0..cols {
        let Some(found) = (pivot_row..matrix.len()).find(|&r| matrix[r][col] != 0) else {
            continue;
        };
        matrix.swap(pivot_row, found);
        let inv = field.inv(matrix[pivot_row][col])?;
        for c in col..cols {
            matrix[pivot_row][c] = field.mul(matrix[pivot_row][c], inv);
        }
        for r in 0..matrix.len() {
            if r == pivot_row {
                continue;
            }
            let factor = matrix[r][col];
            if factor == 0 {
                continue;
            }
            for c in col..cols {
                matrix[r][c] = field.sub(matrix[r][c], field.mul(factor, matrix[pivot_row][c]));
            }
        }
        pivot_cols.push(col);
        pivot_row += 1;
        if pivot_row == matrix.len() {
            break;
        }
    }
    Ok(Rref { matrix, pivot_cols })
}

pub fn rank_rows(matrix: Vec<Vec<u64>>, field: Field) -> Result<usize> {
    Ok(rref(matrix, field)?.pivot_cols.len())
}

pub fn rank_columns(columns: &[Vec<u64>], field: Field) -> Result<usize> {
    if columns.is_empty() {
        return Ok(0);
    }
    let row_count = columns[0].len();
    if columns.iter().any(|c| c.len() != row_count) {
        return Err("columns have inconsistent lengths".to_string());
    }
    let mut rows = vec![vec![0; columns.len()]; row_count];
    for (j, column) in columns.iter().enumerate() {
        for (i, value) in column.iter().copied().enumerate() {
            rows[i][j] = value;
        }
    }
    rank_rows(rows, field)
}

pub fn nullspace(matrix: Vec<Vec<u64>>, field: Field) -> Result<Vec<Vec<u64>>> {
    let cols = matrix.first().map_or(0, |row| row.len());
    let r = rref(matrix, field)?;
    let mut is_pivot = vec![false; cols];
    for &col in &r.pivot_cols {
        is_pivot[col] = true;
    }

    let mut basis = Vec::new();
    for free_col in 0..cols {
        if is_pivot[free_col] {
            continue;
        }
        let mut vector = vec![0; cols];
        vector[free_col] = 1;
        for (row_idx, &pivot_col) in r.pivot_cols.iter().enumerate() {
            vector[pivot_col] = field.neg(r.matrix[row_idx][free_col]);
        }
        basis.push(vector);
    }
    Ok(basis)
}

pub fn invert_square(matrix: Vec<Vec<u64>>, field: Field) -> Result<Vec<Vec<u64>>> {
    let n = matrix.len();
    if n == 0 {
        return Err("cannot invert empty matrix".to_string());
    }
    if matrix.iter().any(|row| row.len() != n) {
        return Err("matrix is not square".to_string());
    }
    let mut augmented = vec![vec![0; 2 * n]; n];
    for i in 0..n {
        for j in 0..n {
            augmented[i][j] = matrix[i][j];
        }
        augmented[i][n + i] = 1;
    }
    let r = rref(augmented, field)?;
    if r.pivot_cols.len() < n || r.pivot_cols[..n] != (0..n).collect::<Vec<_>>() {
        return Err("matrix is singular".to_string());
    }
    let mut inverse = vec![vec![0; n]; n];
    for i in 0..n {
        for j in 0..n {
            inverse[i][j] = r.matrix[i][n + j];
        }
    }
    Ok(inverse)
}

pub fn mat_vec_mul(matrix: &[Vec<u64>], vector: &[u64], field: Field) -> Vec<u64> {
    matrix
        .iter()
        .map(|row| {
            row.iter()
                .zip(vector)
                .fold(0, |acc, (&a, &b)| field.add(acc, field.mul(a, b)))
        })
        .collect()
}

pub fn linear_combination(columns: &[Vec<u64>], coeffs: &[u64], field: Field) -> Result<Vec<u64>> {
    if columns.len() != coeffs.len() {
        return Err(format!(
            "linear combination has {} columns but {} coefficients",
            columns.len(),
            coeffs.len()
        ));
    }
    if columns.is_empty() {
        return Ok(Vec::new());
    }
    let row_count = columns[0].len();
    let mut out = vec![0; row_count];
    for (column, &scale) in columns.iter().zip(coeffs) {
        if column.len() != row_count {
            return Err("columns have inconsistent lengths".to_string());
        }
        if scale == 0 {
            continue;
        }
        for (dst, &value) in out.iter_mut().zip(column) {
            *dst = field.add(*dst, field.mul(scale, value));
        }
    }
    Ok(out)
}

pub fn solve_full_column_rank_slow(
    columns: &[Vec<u64>],
    rhs: &[u64],
    field: Field,
) -> Result<Vec<u64>> {
    if columns.is_empty() {
        return if rhs.iter().all(|&x| x == 0) {
            Ok(Vec::new())
        } else {
            Err("inconsistent empty column solve".to_string())
        };
    }
    let row_count = columns[0].len();
    let col_count = columns.len();
    if rhs.len() != row_count {
        return Err(format!("rhs length {}, expected {row_count}", rhs.len()));
    }
    if columns.iter().any(|c| c.len() != row_count) {
        return Err("columns have inconsistent lengths".to_string());
    }

    let mut augmented = vec![vec![0; col_count + 1]; row_count];
    for i in 0..row_count {
        for j in 0..col_count {
            augmented[i][j] = columns[j][i];
        }
        augmented[i][col_count] = rhs[i];
    }

    let mut pivot_row = 0;
    let mut pivot_cols = Vec::new();
    for col in 0..col_count {
        let Some(found) = (pivot_row..row_count).find(|&r| augmented[r][col] != 0) else {
            continue;
        };
        augmented.swap(pivot_row, found);
        let inv = field.inv(augmented[pivot_row][col])?;
        for c in col..=col_count {
            augmented[pivot_row][c] = field.mul(augmented[pivot_row][c], inv);
        }
        for r in 0..row_count {
            if r == pivot_row {
                continue;
            }
            let factor = augmented[r][col];
            if factor == 0 {
                continue;
            }
            for c in col..=col_count {
                augmented[r][c] =
                    field.sub(augmented[r][c], field.mul(factor, augmented[pivot_row][c]));
            }
        }
        pivot_cols.push(col);
        pivot_row += 1;
        if pivot_row == row_count {
            break;
        }
    }

    for row in &augmented {
        if row[..col_count].iter().all(|&x| x == 0) && row[col_count] != 0 {
            return Err("linear system is inconsistent".to_string());
        }
    }
    if pivot_cols.len() != col_count {
        return Err(format!(
            "column matrix rank {}, expected full column rank {col_count}",
            pivot_cols.len()
        ));
    }
    let mut solution = vec![0; col_count];
    for (row_idx, &col) in pivot_cols.iter().enumerate() {
        solution[col] = augmented[row_idx][col_count];
    }
    Ok(solution)
}

pub fn rows_from_columns(columns: &[Vec<u64>]) -> Result<Vec<Vec<u64>>> {
    if columns.is_empty() {
        return Ok(Vec::new());
    }
    let row_count = columns[0].len();
    if columns.iter().any(|c| c.len() != row_count) {
        return Err("columns have inconsistent lengths".to_string());
    }
    let mut rows = vec![vec![0; columns.len()]; row_count];
    for (j, column) in columns.iter().enumerate() {
        for (i, value) in column.iter().copied().enumerate() {
            rows[i][j] = value;
        }
    }
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nullspace_uses_leftmost_pivots() {
        let f = Field::new(11).unwrap();
        let matrix = vec![vec![1, 2, 3], vec![2, 4, 6]];
        let ns = nullspace(matrix, f).unwrap();
        assert_eq!(ns, vec![vec![9, 1, 0], vec![8, 0, 1]]);
    }

    #[test]
    fn column_solver_recovers_coordinates() {
        let f = Field::new(17).unwrap();
        let cols = vec![vec![1, 2, 0], vec![0, 1, 3]];
        let coeffs = vec![5, 7];
        let rhs = linear_combination(&cols, &coeffs, f).unwrap();
        let solver = ColumnSolver::new(&cols, f).unwrap();
        assert_eq!(solver.solve_and_verify(&cols, &rhs, f).unwrap(), coeffs);
        assert_eq!(solve_full_column_rank_slow(&cols, &rhs, f).unwrap(), coeffs);
    }
}

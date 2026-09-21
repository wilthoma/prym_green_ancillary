//! Dense multiplication tensor for the Artinian section-space quotient.
//! Entries use ((i*n)+beta)*g+a: a varies fastest, then beta, then i.

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct MuData {
    pub n: usize,
    pub g: usize,
    pub layout: String,
    pub values: Vec<u64>,
}

impl MuData {
    pub fn new(n: usize, g: usize) -> Self {
        Self {
            n,
            g,
            layout: "i-beta-a with a varying fastest".to_string(),
            values: vec![0; n * n * g],
        }
    }

    pub fn index(&self, i: usize, beta: usize, a: usize) -> usize {
        debug_assert!(i < self.n);
        debug_assert!(beta < self.n);
        debug_assert!(a < self.g);
        (i * self.n + beta) * self.g + a
    }

    pub fn get(&self, i: usize, beta: usize, a: usize) -> u64 {
        self.values[self.index(i, beta, a)]
    }

    pub fn set(&mut self, i: usize, beta: usize, a: usize, value: u64) {
        let idx = self.index(i, beta, a);
        self.values[idx] = value;
    }

    pub fn nonzero_coefficients(&self) -> u64 {
        self.values.iter().filter(|&&x| x != 0).count() as u64
    }
}

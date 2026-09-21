//! Projective node coordinates and distinctness checks.
//! The paper fixes every point explicitly through two cyclic orbits.

// Explicit indices and argument lists follow the mathematical matrix identities.
#![allow(clippy::needless_range_loop)]

use crate::{Result, field::Field};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ProjectivePoint {
    pub x0: u64,
    pub x1: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct PointPair {
    pub p: ProjectivePoint,
    pub q: ProjectivePoint,
}

impl ProjectivePoint {
    pub fn new(x0: u64, x1: u64, field: Field) -> Result<Self> {
        let point = Self {
            x0: x0 % field.modulus(),
            x1: x1 % field.modulus(),
        };
        if point.x0 == 0 && point.x1 == 0 {
            return Err("projective point (0:0) is invalid".to_string());
        }
        Ok(point)
    }

    pub fn affine(z: u64, field: Field) -> Self {
        Self {
            x0: 1,
            x1: z % field.modulus(),
        }
    }

    pub fn determinant(&self, other: &Self, field: Field) -> u64 {
        field.sub(field.mul(self.x0, other.x1), field.mul(self.x1, other.x0))
    }

    pub fn projectively_equals(&self, other: &Self, field: Field) -> bool {
        self.determinant(other, field) == 0
    }
}

pub fn validate_point_pairs(point_pairs: &[PointPair], field: Field) -> Result<()> {
    for (idx, pair) in point_pairs.iter().enumerate() {
        if pair.p.x0 >= field.modulus()
            || pair.p.x1 >= field.modulus()
            || pair.q.x0 >= field.modulus()
            || pair.q.x1 >= field.modulus()
        {
            return Err(format!(
                "point pair {idx} has coordinates outside the field"
            ));
        }
        if pair.p.x0 == 0 && pair.p.x1 == 0 {
            return Err(format!("P_{idx} is (0:0)"));
        }
        if pair.q.x0 == 0 && pair.q.x1 == 0 {
            return Err(format!("Q_{idx} is (0:0)"));
        }
    }

    for i in 0..point_pairs.len() {
        let pi = [&point_pairs[i].p, &point_pairs[i].q];
        for j in i..point_pairs.len() {
            let start = if i == j { 1 } else { 0 };
            let pj = [&point_pairs[j].p, &point_pairs[j].q];
            for a in 0..2 {
                for b in start..2 {
                    if i == j && a >= b {
                        continue;
                    }
                    if pi[a].projectively_equals(pj[b], field) {
                        return Err(format!("points ({i},{a}) and ({j},{b}) are not distinct"));
                    }
                }
            }
        }
    }
    Ok(())
}

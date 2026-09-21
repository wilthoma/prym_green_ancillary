//! Homogeneous binary forms in coefficients [x0^d, x0^(d-1)x1, ..., x1^d].
//! Products, evaluations, and projective common-zero checks include infinity.

use serde::{Deserialize, Serialize};

use crate::{Result, field::Field, points::ProjectivePoint};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct BinaryForm {
    pub degree: usize,
    pub coeffs: Vec<u64>,
}

impl BinaryForm {
    pub fn new(coeffs: Vec<u64>, field: Field) -> Result<Self> {
        if coeffs.is_empty() {
            return Err("binary form needs at least one coefficient".to_string());
        }
        let coeffs: Vec<_> = coeffs.into_iter().map(|c| c % field.modulus()).collect();
        Ok(Self {
            degree: coeffs.len() - 1,
            coeffs,
        })
    }

    pub fn zero(degree: usize) -> Self {
        Self {
            degree,
            coeffs: vec![0; degree + 1],
        }
    }

    pub fn one() -> Self {
        Self {
            degree: 0,
            coeffs: vec![1],
        }
    }

    pub fn linear_for_point(point: &ProjectivePoint, field: Field) -> Self {
        Self {
            degree: 1,
            coeffs: vec![field.neg(point.x1), point.x0],
        }
    }

    pub fn evaluate(&self, point: &ProjectivePoint, field: Field) -> u64 {
        let mut value = 0;
        for (i, coeff) in self.coeffs.iter().copied().enumerate() {
            let x0_pow = field.pow(point.x0, (self.degree - i) as u64);
            let x1_pow = field.pow(point.x1, i as u64);
            value = field.add(value, field.mul(coeff, field.mul(x0_pow, x1_pow)));
        }
        value
    }

    pub fn monomial_evaluations(degree: usize, point: &ProjectivePoint, field: Field) -> Vec<u64> {
        (0..=degree)
            .map(|i| {
                let x0_pow = field.pow(point.x0, (degree - i) as u64);
                let x1_pow = field.pow(point.x1, i as u64);
                field.mul(x0_pow, x1_pow)
            })
            .collect()
    }

    pub fn mul(&self, other: &Self, field: Field) -> Self {
        let mut coeffs = vec![0; self.degree + other.degree + 1];
        for (i, a) in self.coeffs.iter().copied().enumerate() {
            for (j, b) in other.coeffs.iter().copied().enumerate() {
                coeffs[i + j] = field.add(coeffs[i + j], field.mul(a, b));
            }
        }
        Self {
            degree: self.degree + other.degree,
            coeffs,
        }
    }

    pub fn add_scaled(&mut self, scale: u64, other: &Self, field: Field) -> Result<()> {
        if self.degree != other.degree {
            return Err(format!(
                "degree mismatch in add_scaled: {} vs {}",
                self.degree, other.degree
            ));
        }
        if scale == 0 {
            return Ok(());
        }
        for (dst, src) in self.coeffs.iter_mut().zip(&other.coeffs) {
            *dst = field.add(*dst, field.mul(scale, *src));
        }
        Ok(())
    }

    pub fn sub_scaled(&mut self, scale: u64, other: &Self, field: Field) -> Result<()> {
        self.add_scaled(field.neg(scale), other, field)
    }

    pub fn is_zero(&self) -> bool {
        self.coeffs.iter().all(|&c| c == 0)
    }

    pub fn dehomogenized_coeffs(&self) -> Vec<u64> {
        self.coeffs.clone()
    }
}

pub fn product(forms: &[BinaryForm], field: Field) -> BinaryForm {
    let mut acc = BinaryForm::one();
    for form in forms {
        acc = acc.mul(form, field);
    }
    acc
}

pub fn has_common_projective_zero(a: &BinaryForm, b: &BinaryForm, field: Field) -> bool {
    let common_infinity = a.coeffs[a.degree] == 0 && b.coeffs[b.degree] == 0;
    if common_infinity {
        return true;
    }
    polynomial_gcd_degree(a.dehomogenized_coeffs(), b.dehomogenized_coeffs(), field) > 0
}

fn polynomial_gcd_degree(mut a: Vec<u64>, mut b: Vec<u64>, field: Field) -> isize {
    trim_poly(&mut a);
    trim_poly(&mut b);
    if a.is_empty() && b.is_empty() {
        return -1;
    }
    if a.is_empty() {
        return b.len() as isize - 1;
    }
    if b.is_empty() {
        return a.len() as isize - 1;
    }

    while !b.is_empty() {
        let r = polynomial_rem(&a, &b, field);
        a = b;
        b = r;
    }
    a.len() as isize - 1
}

fn polynomial_rem(a: &[u64], b: &[u64], field: Field) -> Vec<u64> {
    let mut r = a.to_vec();
    let mut divisor = b.to_vec();
    trim_poly(&mut r);
    trim_poly(&mut divisor);
    if divisor.is_empty() {
        return r;
    }
    let divisor_degree = divisor.len() - 1;
    let divisor_lc_inv = field
        .inv(divisor[divisor_degree])
        .expect("nonzero leading coeff");
    while r.len() >= divisor.len() && !r.is_empty() {
        let degree_delta = r.len() - divisor.len();
        let scale = field.mul(*r.last().unwrap(), divisor_lc_inv);
        if scale != 0 {
            for (j, coeff) in divisor.iter().copied().enumerate().take(divisor_degree + 1) {
                let idx = degree_delta + j;
                r[idx] = field.sub(r[idx], field.mul(scale, coeff));
            }
        }
        trim_poly(&mut r);
    }
    r
}

fn trim_poly(poly: &mut Vec<u64>) {
    while poly.last().copied() == Some(0) {
        poly.pop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_form_vanishes_on_point() {
        let f = Field::new(101).unwrap();
        let p = ProjectivePoint::new(3, 7, f).unwrap();
        let ell = BinaryForm::linear_for_point(&p, f);
        assert_eq!(ell.coeffs, vec![94, 3]);
        assert_eq!(ell.evaluate(&p, f), 0);
    }

    #[test]
    fn multiplication_and_evaluation_match_hand_example() {
        let f = Field::new(101).unwrap();
        let a = BinaryForm::new(vec![1, 2, 3], f).unwrap();
        let b = BinaryForm::new(vec![4, 5], f).unwrap();
        let product = a.mul(&b, f);
        assert_eq!(product.coeffs, vec![4, 13, 22, 15]);

        let p = ProjectivePoint::new(2, 3, f).unwrap();
        assert_eq!(
            product.evaluate(&p, f),
            f.mul(a.evaluate(&p, f), b.evaluate(&p, f))
        );
    }

    #[test]
    fn common_projective_zero_detects_finite_and_infinity() {
        let f = Field::new(101).unwrap();
        let x1_minus_2x0 = BinaryForm::new(vec![99, 1], f).unwrap();
        let x1_minus_2x0_squared = x1_minus_2x0.mul(&x1_minus_2x0, f);
        let x1_minus_3x0 = BinaryForm::new(vec![98, 1], f).unwrap();
        assert!(has_common_projective_zero(
            &x1_minus_2x0,
            &x1_minus_2x0_squared,
            f
        ));
        assert!(!has_common_projective_zero(&x1_minus_2x0, &x1_minus_3x0, f));

        let x0 = BinaryForm::new(vec![1, 0], f).unwrap();
        let x0_squared = x0.mul(&x0, f);
        assert!(has_common_projective_zero(&x0, &x0_squared, f));
    }
}

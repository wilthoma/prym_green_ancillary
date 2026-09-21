//! Exact odd-prime-field arithmetic for the small section-space calculations.
//! Residues are canonical in 0..p; multiplication uses a u128 intermediate.

use crate::Result;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Field {
    modulus: u64,
}

impl Field {
    pub fn new(modulus: u64) -> Result<Self> {
        if modulus < 3 || modulus.is_multiple_of(2) || !is_prime(modulus) {
            return Err(format!("modulus {modulus} is not an odd prime"));
        }
        Ok(Self { modulus })
    }

    pub fn modulus(self) -> u64 {
        self.modulus
    }

    pub fn normalize(self, value: i128) -> u64 {
        let p = self.modulus as i128;
        let mut value = value % p;
        if value < 0 {
            value += p;
        }
        value as u64
    }

    pub fn add(self, a: u64, b: u64) -> u64 {
        let sum = a as u128 + b as u128;
        if a < self.modulus && b < self.modulus {
            let modulus = self.modulus as u128;
            if sum >= modulus {
                (sum - modulus) as u64
            } else {
                sum as u64
            }
        } else {
            self.reduce_u128(sum)
        }
    }

    pub fn sub(self, a: u64, b: u64) -> u64 {
        if a >= b {
            a - b
        } else {
            self.modulus - (b - a)
        }
    }

    pub fn neg(self, a: u64) -> u64 {
        if a == 0 { 0 } else { self.modulus - a }
    }

    pub fn mul(self, a: u64, b: u64) -> u64 {
        self.reduce_u128(a as u128 * b as u128)
    }

    pub fn pow(self, mut base: u64, mut exp: u64) -> u64 {
        let mut acc = 1;
        while exp > 0 {
            if exp & 1 == 1 {
                acc = self.mul(acc, base);
            }
            base = self.mul(base, base);
            exp >>= 1;
        }
        acc
    }

    pub fn inv(self, a: u64) -> Result<u64> {
        if a == 0 {
            return Err("attempted to invert zero".to_string());
        }
        Ok(self.pow(a, self.modulus - 2))
    }

    pub fn div(self, a: u64, b: u64) -> Result<u64> {
        Ok(self.mul(a, self.inv(b)?))
    }

    #[inline(always)]
    fn reduce_u128(self, value: u128) -> u64 {
        (value % self.modulus as u128) as u64
    }
}

pub fn is_prime(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37] {
        if n == p {
            return true;
        }
        if n.is_multiple_of(p) {
            return false;
        }
    }
    let d = n - 1;
    let s = d.trailing_zeros();
    let d = d >> s;
    let field = Field { modulus: n };
    for a in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37] {
        if a >= n {
            continue;
        }
        let mut x = field.pow(a, d);
        if x == 1 || x == n - 1 {
            continue;
        }
        let mut maybe_prime = false;
        for _ in 1..s {
            x = field.mul(x, x);
            if x == n - 1 {
                maybe_prime = true;
                break;
            }
        }
        if !maybe_prime {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arithmetic_is_canonical() {
        let f = Field::new(17).unwrap();
        assert_eq!(f.normalize(-3), 14);
        assert_eq!(f.add(16, 3), 2);
        assert_eq!(f.sub(2, 5), 14);
        assert_eq!(f.neg(5), 12);
        assert_eq!(f.mul(9, 4), 2);
        assert_eq!(f.div(10, 5).unwrap(), 2);
    }
}

//! Auxiliary-prime number theoretic transforms and exact modular arithmetic.

use num_traits::{One, Zero};
use rand::distr::uniform::SampleUniform;
use rayon::prelude::*;
use std::fmt::{Debug, Display};
use std::iter::Sum;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Rem, Sub, SubAssign};
use std::ops::{Shl, Shr};

pub trait ModMul:
    Add<Output = Self>
    + DivAssign
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div
    + Rem<Output = Self>
    + Zero
    + One
    + Copy
    + AddAssign
    + SubAssign
    + PartialOrd
{
    const PRIME: Self;
    const ROOT: Self;
    fn mulmod(self, other: Self) -> Self;
    #[inline(always)]
    fn addmod(self, other: Self) -> Self {
        let a = self + other;
        if a >= Self::PRIME { a - Self::PRIME } else { a }
    }
    #[inline(always)]
    fn submod(self, other: Self) -> Self {
        if self < other {
            (self + Self::PRIME) - other
        } else {
            self - other
        }
    }
    #[inline(always)]
    fn powmod(self, mut exp: Self) -> Self {
        let mut base = self;
        let mut result = Self::one();
        let two = Self::one() + Self::one();
        while exp > Self::zero() {
            if exp % two == Self::one() {
                result = result.mulmod(base);
            }
            base = base.mulmod(base);
            exp /= two;
        }
        result
    }
}
impl ModMul for u32 {
    const PRIME: u32 = 2013265921;
    const ROOT: u32 = 31;
    #[inline(always)]
    fn mulmod(self, other: Self) -> Self {
        ((self as u64 * other as u64) % Self::PRIME as u64) as u32
    }
}
impl ModMul for u64 {
    const PRIME: u64 = 2305843009146585089; // 2^61 - 2^26 + 1
    const ROOT: u64 = 3;

    #[inline(always)]
    fn mulmod(self, other: Self) -> Self {
        let a = self as u128 * other as u128;
        let cc = ((a << 67) >> 67) as u64; // lowest 61 bits, i.e., bits 0-60
        let bb = ((a << 32) >> 93) as u64; // bits 61-95
        let aa = (a >> 96) as u64; // bits 96-127
        // We have self = cc + bb*2^61 + aa*2^96
        // 2^96 = 2^26 - 1 - 2^35 (mod p) and aa * 2^35 < p (-> we can use our modular addition/subtraction)
        // 2^61 = 2^26 -1 (mod p)
        cc.addmod(bb << 26)
            .submod(bb)
            .addmod(aa << 26)
            .submod(aa)
            .submod(aa << 35)
    }
}
impl ModMul for u128 {
    const PRIME: u128 = 18446744069414584321; // 2^64 - 2^32 + 1
    const ROOT: u128 = 7;
    #[inline(always)]
    fn mulmod(self, other: Self) -> Self {
        let a = self * other;
        let cc = (a << 64) >> 64;
        let bb = (a << 32) >> 96;
        let aa = a >> 96;
        cc.addmod(bb << 32).submod(bb).submod(aa)
    }
}

pub trait NTTInteger:
    ModMul
    + Shl<u32, Output = Self>
    + Shr<u32, Output = Self>
    + Debug
    + Into<u128>
    + From<u32>
    + From<u32>
    + MulAssign
    + Div<Output = Self>
    + DivAssign
    + Sub<Output = Self>
    + SubAssign
    + Display
    + Sum
    + PartialOrd
    + SampleUniform
    + Copy
    + Zero
    + One
    + Rem<Output = Self>
    + Add<Output = Self>
    + AddAssign
    + Mul<Output = Self>
    + Send
    + Sync
    + 'static
{
}
impl<
    T: ModMul
        + Shl<u32, Output = Self>
        + Shr<u32, Output = Self>
        + Debug
        + Into<u128>
        + From<u32>
        + MulAssign
        + Div<Output = Self>
        + DivAssign
        + Sub<Output = Self>
        + SubAssign
        + Display
        + Sum
        + PartialOrd
        + SampleUniform
        + Copy
        + Zero
        + One
        + Rem<Output = Self>
        + Add<Output = Self>
        + AddAssign
        + Mul<Output = Self>
        + Send
        + Sync
        + 'static,
> NTTInteger for T
{
}

#[inline(always)]
pub fn mod_add<T: NTTInteger>(mut a: T, b: T, p: T) -> T {
    a += b;
    if a >= p {
        a -= p;
    }
    a
}

#[inline(always)] // Total HACK here
pub fn mod_mul<T: NTTInteger>(a: T, b: T, p: T) -> T {
    (a * b) % p
    //return a.mulmod(b,p);
    // if p > (T::one() << 32) {
    //     mod_fast(a * b,p)
    // } else  {
    //     (a * b) % p
    // }

    //(a * b) % p // !!!!! this % is responsible for >2/3 of the runtime -- optimize
    //mod_fast(a * b,p) // !!!!! this % is responsible for 2/3 of the runtime -- optimize
}

// #[inline(always)]
// pub fn mod_fast<T : NTTInteger >(a: T, p:T) -> T {
//     // only valid for 18446744069414584321
//     let cc = ((a << 64) >> 64);
//     let bb = ((a<<32) >> 96);
//     let aa = (a >> 96);
//     mod_sub(mod_sub(mod_add(cc,  bb << 32,p), bb,p), aa,p)
// }

#[inline(always)]
pub fn mod_pow<T: NTTInteger>(mut base: T, mut exp: T, p: T) -> T {
    let mut result = T::one();
    let two = T::one() + T::one();
    while exp > T::zero() {
        if exp % two == T::one() {
            result = mod_mul(result, base, p);
        }
        base = mod_mul(base, base, p);
        exp /= two;
    }
    result
}

// Modular inverse using the extended Euclidean algorithm
#[inline(always)]
pub fn modinv<T: NTTInteger>(a: T, p: T) -> T {
    let two = T::one() + T::one();
    mod_pow(a, p - two, p)
}

#[inline(always)]
pub fn bit_reverse(mut x: usize, bits: usize) -> usize {
    let mut result = 0;
    for _ in 0..bits {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    result
}

// in-place number theoretic transform
pub fn ntt<T: NTTInteger>(a: &mut [T], invert: bool) {
    let p = T::PRIME;
    let root = T::ROOT;
    let n = a.len();
    let bits = n.trailing_zeros() as usize;
    assert_eq!(n, 1 << bits, "Length of input array must be a power of 2");
    let two = T::one() + T::one();

    // Bit reversal permutation
    for i in 0..n {
        let j = bit_reverse(i, bits);
        if i < j {
            a.swap(i, j);
        }
    }

    let root = if invert {
        root.powmod(p - T::one() - (p - T::one()) / (T::from(n as u32)))
    } else {
        root.powmod((p - T::one()) / (T::from(n as u32)))
    };

    let mut len = 2;
    while len <= n {
        let wlen = root.powmod(T::from((n / len) as u32));

        a.chunks_mut(len).for_each(|chunk| {
            let mut w = T::one();
            let (left, right) = chunk.split_at_mut(len / 2);
            for (l, r) in left.iter_mut().zip(right.iter_mut()) {
                let u = *l;
                let v = (*r).mulmod(w);
                *l = u.addmod(v);
                *r = u.submod(v);
                w = w.mulmod(wlen);
            }
        });

        len <<= 1;
    }

    if invert {
        let n_inv = T::from(n as u32).powmod(p - two);
        a.iter_mut().for_each(|x| {
            *x = (*x).mulmod(n_inv);
        });
    }
}

// pub fn ntt_par<T : NTTInteger>(a: &mut [T], invert: bool, p : T, root : T) {
//     let n = a.len();
//     let bits = n.trailing_zeros() as usize;
//     assert_eq!(n, 1 << bits, "Length of input array must be a power of 2");
//     let two = T::one() + T::one();

//     // Bit reversal permutation
//     for i in 0..n {
//         let j = bit_reverse(i, bits);
//         if i < j {
//             a.swap(i, j);
//         }
//     }

//     let root = if invert {
//         mod_pow(root, p - T::one() - (p - T::one()) / (T::from(n as u32)),p)
//     } else {
//         mod_pow(root, (p - T::one()) / (T::from(n as u32)),p)
//     };

//     let mut len = 2;
//     while len <= n {
//         let wlen = mod_pow(root, T::from((n / len) as u32),p);

//         // Parallel over blocks
//         a.par_chunks_mut(len).for_each(|chunk| {
//             let mut w = T::one();
//             let (left, right) = chunk.split_at_mut(len/2);
//             for (l, r) in left.iter_mut().zip(right.iter_mut()) {
//                 let u = *l;
//                 let v = mod_mul(*r, w, p);
//                 *l = mod_add(u, v, p);
//                 *r = mod_sub(u, v, p);
//                 w = mod_mul(w, wlen, p);
//             }
//         });

//         len <<= 1;
//     }

//     if invert {
//         let n_inv = mod_pow(T::from(n as u32), p - two, p);
//         a.par_iter_mut().for_each(|x| {
//             *x = mod_mul(*x, n_inv, p);
//         });
//     }
// }

pub fn matrix_ntt_parallel<T: NTTInteger>(a: &mut Vec<Vec<Vec<T>>>, invert: bool) {
    // apply ntt to all elements in parallel

    a.par_iter_mut()
        .flat_map(|row| row.par_iter_mut())
        .for_each(|x| {
            ntt(x, invert);
        });
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng};

    fn generate_random_data(len: usize, p: u64) -> Vec<u64> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(0x5052594d);
        (0..len).map(|_| rng.random_range(0..p)).collect()
    }

    #[test]
    fn test_ntt_inverse() {
        let p = u64::PRIME; // A prime modulus
        let len = 16; // Must be a power of two
        let mut data = generate_random_data(len, p);

        let original_data = data.clone();
        ntt(&mut data, false); // Forward NTT
        ntt(&mut data, true); // Inverse NTT

        assert_eq!(
            data, original_data,
            "NTT followed by inverse NTT should return the original data"
        );
    }
}

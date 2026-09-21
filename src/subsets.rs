//! Lexicographic exterior-subset enumeration used for Koszul coordinates.
//! Subset order determines the source/target layouts shared with CUDA.

use crate::Result;

pub fn binom(n: usize, k: usize) -> u64 {
    if k > n {
        return 0;
    }
    let k = k.min(n - k);
    let mut out: u128 = 1;
    for i in 0..k {
        out = out * (n - i) as u128 / (i + 1) as u128;
    }
    u64::try_from(out).expect("binomial coefficient does not fit in u64")
}

pub fn rank_subset(subset: &[usize], n: usize, k: usize) -> Result<u64> {
    if subset.len() != k {
        return Err(format!("subset has length {}, expected {k}", subset.len()));
    }
    let mut rank = 0;
    let mut min_value = 0;
    for (pos, &value) in subset.iter().enumerate() {
        if value < min_value || value >= n {
            return Err(format!(
                "invalid lexicographic subset {subset:?} of {n} choose {k}"
            ));
        }
        for candidate in min_value..value {
            rank += binom(n - candidate - 1, k - pos - 1);
        }
        min_value = value + 1;
    }
    Ok(rank)
}

pub fn unrank_subset(mut rank: u64, n: usize, k: usize) -> Result<Vec<usize>> {
    let total = binom(n, k);
    if rank >= total {
        return Err(format!("subset rank {rank} is outside 0..{total}"));
    }
    let mut subset = Vec::with_capacity(k);
    let mut start = 0;
    for pos in 0..k {
        let remaining = k - pos - 1;
        let mut chosen = None;
        for value in start..n {
            let count = binom(n - value - 1, remaining);
            if rank < count {
                chosen = Some(value);
                break;
            }
            rank -= count;
        }
        let value = chosen.ok_or_else(|| "failed to unrank subset".to_string())?;
        subset.push(value);
        start = value + 1;
    }
    Ok(subset)
}

pub fn next_subset(subset: &mut [usize], n: usize) -> bool {
    let k = subset.len();
    for i in (0..k).rev() {
        if subset[i] < n - k + i {
            subset[i] += 1;
            for j in i + 1..k {
                subset[j] = subset[j - 1] + 1;
            }
            return true;
        }
    }
    false
}

pub fn remove_at(subset: &[usize], idx: usize) -> Vec<usize> {
    subset
        .iter()
        .enumerate()
        .filter_map(|(j, &value)| if j == idx { None } else { Some(value) })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rank_unrank_are_inverse() {
        for n in 1..9 {
            for k in 0..=n {
                let total = binom(n, k);
                let mut previous = None;
                for rank in 0..total {
                    let subset = unrank_subset(rank, n, k).unwrap();
                    assert_eq!(rank_subset(&subset, n, k).unwrap(), rank);
                    if let Some(prev) = previous {
                        assert!(prev < subset);
                    }
                    previous = Some(subset);
                }
            }
        }
    }

    #[test]
    fn next_subset_matches_unrank() {
        let n = 7;
        let k = 3;
        let mut subset = vec![0, 1, 2];
        for rank in 0..binom(n, k) {
            assert_eq!(subset, unrank_subset(rank, n, k).unwrap());
            if rank + 1 < binom(n, k) {
                assert!(next_subset(&mut subset, n));
            } else {
                assert!(!next_subset(&mut subset, n));
            }
        }
    }
}

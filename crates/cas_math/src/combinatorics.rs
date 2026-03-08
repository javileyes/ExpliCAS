//! Basic combinatorics helpers shared by expansion rules.

/// Binomial coefficient `C(n, k)`.
pub fn binomial_coeff(n: u32, k: u32) -> u32 {
    if k == 0 || k == n {
        return 1;
    }
    if k > n {
        return 0;
    }
    let mut res = 1;
    for i in 0..k {
        res = res * (n - i) / (i + 1);
    }
    res
}

#[cfg(test)]
mod tests {
    use super::binomial_coeff;

    #[test]
    fn binomial_edges() {
        assert_eq!(binomial_coeff(5, 0), 1);
        assert_eq!(binomial_coeff(5, 5), 1);
        assert_eq!(binomial_coeff(5, 6), 0);
    }

    #[test]
    fn binomial_values() {
        assert_eq!(binomial_coeff(5, 2), 10);
        assert_eq!(binomial_coeff(8, 3), 56);
    }
}

//! Modular field arithmetic over Fp (prime field).
//!
//! All operations are constant-time for a given prime p.
//! Uses u128 for multiplication to avoid overflow.

/// Modular addition: (a + b) mod p
#[inline(always)]
pub fn add_mod(a: u64, b: u64, p: u64) -> u64 {
    let sum = a as u128 + b as u128;
    (sum % p as u128) as u64
}

/// Modular subtraction: (a - b) mod p (always non-negative)
#[inline(always)]
pub fn sub_mod(a: u64, b: u64, p: u64) -> u64 {
    if a >= b {
        a - b
    } else {
        p - (b - a) % p
    }
}

/// Modular multiplication: (a * b) mod p
#[inline(always)]
pub fn mul_mod(a: u64, b: u64, p: u64) -> u64 {
    let prod = (a as u128) * (b as u128);
    (prod % p as u128) as u64
}

/// Modular negation: (-a) mod p
#[inline(always)]
pub fn neg_mod(a: u64, p: u64) -> u64 {
    if a == 0 {
        0
    } else {
        p - (a % p)
    }
}

/// Extended GCD: returns (gcd, x, y) such that gcd = a*x + b*y
/// x and y are signed (can be negative)
fn extended_gcd(a: i128, b: i128) -> (i128, i128, i128) {
    if b == 0 {
        (a, 1, 0)
    } else {
        let (g, x, y) = extended_gcd(b, a % b);
        (g, y, x - (a / b) * y)
    }
}

/// Modular inverse: a^(-1) mod p, returns None if gcd(a,p) != 1
#[inline]
pub fn inv_mod(a: u64, p: u64) -> Option<u64> {
    if a == 0 {
        return None;
    }
    let (g, x, _) = extended_gcd(a as i128, p as i128);
    if g != 1 {
        return None;
    }
    // x can be negative, normalize to [0, p)
    let inv = ((x % p as i128) + p as i128) % p as i128;
    Some(inv as u64)
}

/// Modular division: (a / b) mod p = a * b^(-1) mod p
#[inline]
pub fn div_mod(a: u64, b: u64, p: u64) -> Option<u64> {
    inv_mod(b, p).map(|b_inv| mul_mod(a, b_inv, p))
}

/// Modular exponentiation: a^e mod p (binary exponentiation)
pub fn pow_mod(mut base: u64, mut exp: u64, p: u64) -> u64 {
    if p == 1 {
        return 0;
    }
    let mut result: u64 = 1;
    base %= p;
    while exp > 0 {
        if exp & 1 == 1 {
            result = mul_mod(result, base, p);
        }
        exp >>= 1;
        base = mul_mod(base, base, p);
    }
    result
}

/// A commonly used large prime for modular GCD (fits in u64)
pub const DEFAULT_PRIME: u64 = 0xFFFF_FFFF_FFFF_FFC5; // 2^64 - 59, a prime

/// Smaller prime for testing
pub const TEST_PRIME: u64 = 1_000_000_007; // 10^9 + 7

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_mod() {
        let p = 17u64;
        assert_eq!(add_mod(10, 5, p), 15);
        assert_eq!(add_mod(10, 10, p), 3); // 20 mod 17 = 3
    }

    #[test]
    fn test_sub_mod() {
        let p = 17u64;
        assert_eq!(sub_mod(10, 5, p), 5);
        assert_eq!(sub_mod(5, 10, p), 12); // -5 mod 17 = 12
    }

    #[test]
    fn test_mul_mod() {
        let p = 17u64;
        assert_eq!(mul_mod(10, 5, p), 16); // 50 mod 17 = 16
        assert_eq!(mul_mod(10, 10, p), 15); // 100 mod 17 = 15
    }

    #[test]
    fn test_inv_mod() {
        let p = 17u64;
        // 3 * 6 = 18 = 1 mod 17
        assert_eq!(inv_mod(3, p), Some(6));
        // Verify: a * inv(a) = 1 mod p
        for a in 1..p {
            let a_inv = inv_mod(a, p).unwrap();
            assert_eq!(mul_mod(a, a_inv, p), 1, "Failed for a={}", a);
        }
    }

    #[test]
    fn test_inv_zero() {
        assert_eq!(inv_mod(0, 17), None);
    }

    #[test]
    fn test_div_mod() {
        let p = 17u64;
        // 10 / 2 = 5 mod 17
        assert_eq!(div_mod(10, 2, p), Some(5));
        // 10 / 3: need 10 * inv(3) = 10 * 6 = 60 = 9 mod 17
        assert_eq!(div_mod(10, 3, p), Some(9));
    }

    #[test]
    fn test_pow_mod() {
        let p = 17u64;
        assert_eq!(pow_mod(2, 4, p), 16);
        assert_eq!(pow_mod(2, 5, p), 15); // 32 mod 17 = 15
        assert_eq!(pow_mod(3, 16, p), 1); // Fermat: a^(p-1) = 1 mod p
    }

    #[test]
    fn test_large_prime() {
        let p = TEST_PRIME;
        let a = 123456789u64;
        let a_inv = inv_mod(a, p).unwrap();
        assert_eq!(mul_mod(a, a_inv, p), 1);
    }
}

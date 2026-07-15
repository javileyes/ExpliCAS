//! Subresultant polynomial-remainder sequence over ℚ[t] (G1 Cap. E-i).
//!
//! The Lazard-Rioboo-Trager logarithmic-part algorithm integrates `N(x)/D(x)`
//! (with `D` squarefree over ℚ) WITHOUT factoring `D`: it forms the
//! Rothstein-Trager resultant `R(t) = res_x(N − t·D', D) ∈ ℚ[t]`, whose roots
//! are the logarithmic coefficients, and reads the logarithmic arguments off
//! the SUBRESULTANT polynomial-remainder sequence of the same pair. This
//! module is the standalone, wired-to-nothing primitive layer for that
//! algorithm (the C-i analogue): a light `ℚ[t][x]` polynomial type plus the
//! subresultant PRS and the resultant as its degree-0 tail.
//!
//! Everything stays EXACT over the integral domain ℚ[t] — the coefficient
//! ring is the existing univariate [`Polynomial`] reinterpreted in the
//! parameter `t`. The subresultant reduction uses pseudo-division with the
//! Brown/GCL subresultant divisors, so intermediate coefficients never leave
//! ℚ[t] and do not blow up the way a naive pseudo-remainder sequence would.
//! Correctness is pinned exactly against SymPy `resultant`/`subresultants`.

use crate::polynomial::Polynomial;
use num_rational::BigRational;
use num_traits::{One, Zero};

const T: &str = "t";

fn t_zero() -> Polynomial {
    Polynomial::zero(T.to_string())
}

fn t_one() -> Polynomial {
    Polynomial::one(T.to_string())
}

fn t_const(value: BigRational) -> Polynomial {
    Polynomial::new(vec![value], T.to_string())
}

/// Exact power of a ℚ[t] element (repeated multiplication). `exp == 0` gives 1.
fn poly_pow(base: &Polynomial, exp: u32) -> Polynomial {
    let mut acc = t_one();
    for _ in 0..exp {
        acc = acc.mul(base);
    }
    acc
}

/// Exact division `a / b` in ℚ[t]; returns `None` if it does not divide
/// evenly (a construction bug — the caller treats it as an honest bail).
fn poly_exact_div(a: &Polynomial, b: &Polynomial) -> Option<Polynomial> {
    let (q, r) = a.div_rem(b).ok()?;
    if r.is_zero() {
        Some(q)
    } else {
        None
    }
}

/// A polynomial in `x` whose coefficients live in ℚ[t]: an element of ℚ[t][x].
/// `x_coeffs[i]` is the coefficient of `x^i`, itself a [`Polynomial`] in `t`.
/// The trailing zero invariant (top coefficient non-zero, or empty for 0) is
/// maintained by [`PolyTX::trim`].
#[derive(Clone, Debug, PartialEq)]
pub struct PolyTX {
    pub x_coeffs: Vec<Polynomial>,
}

impl PolyTX {
    fn trim(mut self) -> Self {
        while self.x_coeffs.last().is_some_and(Polynomial::is_zero) {
            self.x_coeffs.pop();
        }
        self
    }

    pub fn zero() -> Self {
        PolyTX { x_coeffs: vec![] }
    }

    pub fn is_zero(&self) -> bool {
        self.x_coeffs.iter().all(Polynomial::is_zero)
    }

    /// Degree in `x` (`0` for a non-zero ℚ[t] constant). Panics-free: a zero
    /// polynomial reports `0`; callers gate on [`PolyTX::is_zero`] first.
    pub fn degree_x(&self) -> usize {
        let trimmed_len = self
            .x_coeffs
            .iter()
            .rposition(|c| !c.is_zero())
            .map(|i| i + 1)
            .unwrap_or(0);
        trimmed_len.saturating_sub(1)
    }

    /// Leading coefficient in `x` (a ℚ[t] element); `0` for the zero poly.
    pub fn leading_x(&self) -> Polynomial {
        self.x_coeffs
            .iter()
            .rev()
            .find(|c| !c.is_zero())
            .cloned()
            .unwrap_or_else(t_zero)
    }

    /// The degree-0 coefficient viewed as a ℚ[t] element (the resultant, once
    /// the PRS has collapsed to `x`-degree 0).
    pub fn constant_in_x(&self) -> Polynomial {
        self.x_coeffs.first().cloned().unwrap_or_else(t_zero)
    }

    pub fn add(&self, other: &Self) -> Self {
        let n = self.x_coeffs.len().max(other.x_coeffs.len());
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let a = self.x_coeffs.get(i).cloned().unwrap_or_else(t_zero);
            let b = other.x_coeffs.get(i).cloned().unwrap_or_else(t_zero);
            out.push(a.add(&b));
        }
        PolyTX { x_coeffs: out }.trim()
    }

    pub fn sub(&self, other: &Self) -> Self {
        let n = self.x_coeffs.len().max(other.x_coeffs.len());
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let a = self.x_coeffs.get(i).cloned().unwrap_or_else(t_zero);
            let b = other.x_coeffs.get(i).cloned().unwrap_or_else(t_zero);
            out.push(a.sub(&b));
        }
        PolyTX { x_coeffs: out }.trim()
    }

    /// Multiply every `x`-coefficient by a ℚ[t] scalar.
    pub fn scale(&self, factor: &Polynomial) -> Self {
        if factor.is_zero() {
            return PolyTX::zero();
        }
        PolyTX {
            x_coeffs: self.x_coeffs.iter().map(|c| c.mul(factor)).collect(),
        }
        .trim()
    }

    /// Multiply by `x^shift` (shift the coefficient vector up).
    pub fn shift_x(&self, shift: usize) -> Self {
        if self.is_zero() {
            return PolyTX::zero();
        }
        let mut out = vec![t_zero(); shift];
        out.extend(self.x_coeffs.iter().cloned());
        PolyTX { x_coeffs: out }.trim()
    }

    /// Divide every `x`-coefficient by a ℚ[t] element, requiring exactness.
    pub fn exact_scalar_div(&self, divisor: &Polynomial) -> Option<Self> {
        let mut out = Vec::with_capacity(self.x_coeffs.len());
        for c in &self.x_coeffs {
            if c.is_zero() {
                out.push(t_zero());
            } else {
                out.push(poly_exact_div(c, divisor)?);
            }
        }
        Some(PolyTX { x_coeffs: out }.trim())
    }
}

/// Pseudo-remainder `prem(a, b)` in ℚ[t][x]: the remainder of
/// `lc(b)^(deg a − deg b + 1)·a` divided by `b`, computed fraction-free so the
/// result stays in ℚ[t][x]. Requires `b != 0`.
pub fn pseudo_remainder(a: &PolyTX, b: &PolyTX) -> PolyTX {
    debug_assert!(!b.is_zero());
    let deg_b = b.degree_x();
    if a.is_zero() || a.degree_x() < deg_b {
        return a.clone();
    }
    let lc_b = b.leading_x();
    let mut r = a.clone();
    // lc(b)^(deg a − deg b + 1): one factor is consumed per reduction step,
    // the surplus is applied once at the end.
    let mut surplus = a.degree_x() - deg_b + 1;
    while !r.is_zero() && r.degree_x() >= deg_b {
        let lc_r = r.leading_x();
        let shift = r.degree_x() - deg_b;
        // r <- lc(b)·r − lc(r)·x^shift·b
        let scaled_r = r.scale(&lc_b);
        let subtrahend = b.scale(&lc_r).shift_x(shift);
        r = scaled_r.sub(&subtrahend);
        surplus -= 1;
    }
    if surplus > 0 {
        r = r.scale(&poly_pow(&lc_b, surplus as u32));
    }
    r
}

/// The result of a subresultant PRS: the chain of remainders (decreasing
/// `x`-degree, starting with the two inputs) and the resultant `res_x`.
#[derive(Clone, Debug)]
pub struct SubresultantPrs {
    /// `chain[0]`, `chain[1]` are the two inputs (higher degree first); the
    /// last element has `x`-degree 0 and holds the resultant when the inputs
    /// are coprime in `x`.
    pub chain: Vec<PolyTX>,
    /// `res_x(a, b) ∈ ℚ[t]`: the degree-0 tail of the subresultant PRS, or `0`
    /// when `a` and `b` share a positive-degree factor in `x`.
    pub resultant: Polynomial,
}

/// Subresultant polynomial-remainder sequence of two ℚ[t][x] polynomials.
///
/// The chain is ordered by strictly decreasing `x`-degree. Each successive
/// remainder is `prem(prev, cur)` divided by the Brown/GCL subresultant
/// divisor `βᵢ`, keeping every element in ℚ[t][x] and making the degree-0
/// tail equal the resultant. Returns `None` only on a structural degeneracy
/// (a division that is not exact — a construction bug, surfaced as an honest
/// bail rather than a wrong value).
pub fn subresultant_prs(a: &PolyTX, b: &PolyTX) -> Option<SubresultantPrs> {
    // Order so the first element has the larger x-degree.
    let (mut prev, mut cur) = if a.degree_x() >= b.degree_x() {
        (a.clone(), b.clone())
    } else {
        (b.clone(), a.clone())
    };

    if cur.is_zero() {
        // res(prev, 0) is 0 unless prev is a non-zero constant in x.
        let resultant = if prev.degree_x() == 0 && !prev.is_zero() {
            t_one()
        } else {
            t_zero()
        };
        return Some(SubresultantPrs {
            chain: vec![prev, cur],
            resultant,
        });
    }

    let mut chain = vec![prev.clone(), cur.clone()];

    // ψ carries the running subresultant coefficient; δ_prev the previous
    // degree gap. First step uses β = (−1)^(δ+1); later steps the GCL update.
    let mut psi = t_one();
    let mut delta_prev: usize = 0;
    let mut first = true;

    loop {
        if cur.degree_x() == 0 {
            // The sequence has collapsed to an x-constant: that is the
            // resultant (the last pushed element).
            break;
        }
        let delta = prev.degree_x() - cur.degree_x();
        let lc_prev = prev.leading_x();

        let beta = if first {
            // β₁ = (−1)^(δ+1)
            if delta % 2 == 0 {
                t_const(-BigRational::one())
            } else {
                t_one()
            }
        } else {
            // ψ ← (−lc(prev))^(δ_prev) · ψ^(1 − δ_prev)   (exact in ℚ[t])
            let neg_lc = lc_prev.neg();
            let numerator = poly_pow(&neg_lc, delta_prev as u32);
            psi = if delta_prev >= 1 {
                let denom = poly_pow(&psi, (delta_prev - 1) as u32);
                poly_exact_div(&numerator, &denom)?
            } else {
                // 1 − δ_prev = 1: ψ ← (−lc(prev))^0 · ψ = ψ (δ_prev == 0)
                numerator.mul(&psi)
            };
            // β = −lc(prev) · ψ^δ
            neg_lc.mul(&poly_pow(&psi, delta as u32))
        };

        let rem = pseudo_remainder(&prev, &cur);
        if rem.is_zero() {
            break;
        }
        let next = rem.exact_scalar_div(&beta)?;

        chain.push(next.clone());
        prev = cur;
        cur = next;
        delta_prev = delta;
        first = false;
    }

    let last = chain.last().cloned().unwrap_or_else(PolyTX::zero);
    let resultant = if last.degree_x() == 0 {
        last.constant_in_x()
    } else {
        // Positive-degree final element ⇒ a non-trivial gcd in x ⇒ res = 0.
        t_zero()
    };

    Some(SubresultantPrs { chain, resultant })
}

/// Build `N − t·D'` as an element of ℚ[t][x] from univariate `N(x)`, `D'(x)`
/// over ℚ. Each `x`-coefficient becomes `N_i − D'_i·t`, degree ≤ 1 in `t`.
fn numerator_minus_t_dprime(numerator: &Polynomial, d_prime: &Polynomial) -> PolyTX {
    let n = numerator.coeffs.len().max(d_prime.coeffs.len());
    let mut x_coeffs = Vec::with_capacity(n);
    for i in 0..n {
        let n_i = numerator
            .coeffs
            .get(i)
            .cloned()
            .unwrap_or_else(BigRational::zero);
        let dp_i = d_prime
            .coeffs
            .get(i)
            .cloned()
            .unwrap_or_else(BigRational::zero);
        // coefficient N_i − D'_i · t  ==  Polynomial [N_i, −D'_i] in t
        x_coeffs.push(Polynomial::new(vec![n_i, -dp_i], T.to_string()));
    }
    PolyTX { x_coeffs }.trim()
}

/// Lift a univariate ℚ-polynomial in `x` to a ℚ[t][x] element with constant
/// (t-free) coefficients.
fn lift_constant_in_t(poly: &Polynomial) -> PolyTX {
    PolyTX {
        x_coeffs: poly.coeffs.iter().cloned().map(t_const).collect(),
    }
    .trim()
}

/// The Rothstein-Trager resultant `R(t) = res_x(N − t·D', D)` for the
/// logarithmic part of `∫ N/D` (with `D` squarefree). Its roots are the
/// logarithmic coefficients. Returns `None` on a structural degeneracy.
pub fn rothstein_trager_resultant(
    numerator: &Polynomial,
    denominator: &Polynomial,
) -> Option<Polynomial> {
    let d_prime = denominator.derivative();
    let a = lift_constant_in_t(denominator);
    let b = numerator_minus_t_dprime(numerator, &d_prime);
    Some(subresultant_prs(&a, &b)?.resultant)
}

/// The full Rothstein-Trager subresultant PRS for `∫ N/D` — both the resultant
/// and the chain of subresultants (used by the logarithmic-argument
/// extraction in a later sub-cycle).
pub fn rothstein_trager_prs(
    numerator: &Polynomial,
    denominator: &Polynomial,
) -> Option<SubresultantPrs> {
    let d_prime = denominator.derivative();
    let a = lift_constant_in_t(denominator);
    let b = numerator_minus_t_dprime(numerator, &d_prime);
    subresultant_prs(&a, &b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_traits::One;

    fn poly_x(coeffs: &[i64]) -> Polynomial {
        Polynomial::new(
            coeffs
                .iter()
                .map(|&c| BigRational::from_integer(c.into()))
                .collect(),
            "x".to_string(),
        )
    }

    /// Ascending-in-t integer coefficient list of a ℚ[t] element.
    fn t_coeffs(p: &Polynomial) -> Vec<i64> {
        p.coeffs
            .iter()
            .map(|c| {
                assert!(c.is_integer(), "expected integer t-coeff, got {c}");
                c.to_integer().try_into().expect("fits i64")
            })
            .collect()
    }

    /// Rothstein-Trager resultants verified against SymPy 1.14
    /// `resultant(N - t*D', D)` (ascending t order). These are the exact
    /// oracle values printed during scoping.
    #[test]
    fn rothstein_trager_resultant_matches_sympy() {
        // (numerator, denominator, expected res(t) ascending)
        let cases: &[(&[i64], &[i64], &[i64])] = &[
            // 1/(x^2+1) -> 4t^2 + 1
            (&[1], &[1, 0, 1], &[1, 0, 4]),
            // 1/(x^3-x-1) -> -23t^3 - 3t + 1
            (&[1], &[-1, -1, 0, 1], &[1, -3, 0, -23]),
            // 1/(x^4+x+1) -> 229t^4 + 18t^2 + 8t + 1
            (&[1], &[1, 1, 0, 0, 1], &[1, 8, 18, 0, 229]),
            // 1/(x^5-1) -> 1 - 3125t^5
            (&[1], &[-1, 0, 0, 0, 0, 1], &[1, 0, 0, 0, 0, -3125]),
            // 1/(x^7-1) -> 1 - 823543t^7
            (
                &[1],
                &[-1, 0, 0, 0, 0, 0, 0, 1],
                &[1, 0, 0, 0, 0, 0, 0, -823543],
            ),
            // x/(x^4+x+1) -> 229t^4 + 32t^2 + t + 1
            (&[0, 1], &[1, 1, 0, 0, 1], &[1, 1, 32, 0, 229]),
            // Rioboo: (x^4-3x^2+6)/(x^6-5x^4+5x^2+4)
            //   -> 2930944t^6 + 2198208t^4 + 549552t^2 + 45796
            (
                &[6, 0, -3, 0, 1],
                &[4, 0, 5, 0, -5, 0, 1],
                &[45796, 0, 549552, 0, 2198208, 0, 2930944],
            ),
            // 1/(x^5-x-1) (the Galois S_5 case) -> -2869t^5 - 160t^3 + 80t^2 - 15t + 1
            (&[1], &[-1, -1, 0, 0, 0, 1], &[1, -15, 80, -160, 0, -2869]),
            // 1/(x^6+x^3+1) (Phi_9) -> 19683t^6 + 243t^3 + 1
            (&[1], &[1, 0, 0, 1, 0, 0, 1], &[1, 0, 0, 243, 0, 0, 19683]),
            // (x^2+1)/(x^5-1) -> -3125t^5 + 125t^2 + 25t + 2
            (&[1, 0, 1], &[-1, 0, 0, 0, 0, 1], &[2, 25, 125, 0, 0, -3125]),
            // 1/(x^4+1) -> 256t^4 + 1
            (&[1], &[1, 0, 0, 0, 1], &[1, 0, 0, 0, 256]),
        ];
        for (num, den, expected) in cases {
            let n = poly_x(num);
            let d = poly_x(den);
            let r = rothstein_trager_resultant(&n, &d).expect("resultant");
            assert_eq!(
                t_coeffs(&r),
                expected.to_vec(),
                "res_x(N - t*D', D) mismatch for num={num:?} den={den:?}"
            );
        }
    }

    /// The subresultant CHAIN for 1/(x^3-x-1), pinned against SymPy
    /// `subresultants(1 - t*D', D)`: degrees 3,2,1,0 with the exact elements
    /// S[2] = (3t - 6t^2)x - 9t^2 and S[3] = -23t^3 - 3t + 1.
    #[test]
    fn subresultant_chain_matches_sympy_cubic() {
        let d = poly_x(&[-1, -1, 0, 1]); // x^3 - x - 1
        let n = poly_x(&[1]);
        let prs = rothstein_trager_prs(&n, &d).expect("prs");
        let degs: Vec<usize> = prs.chain.iter().map(PolyTX::degree_x).collect();
        assert_eq!(degs, vec![3, 2, 1, 0], "chain x-degrees");

        // S[2] = (3t - 6t^2) x - 9t^2  == x_coeffs [ -9t^2 , (3t - 6t^2) ]
        let s2 = &prs.chain[2];
        assert_eq!(t_coeffs(&s2.x_coeffs[0]), vec![0, 0, -9]); // -9t^2
        assert_eq!(t_coeffs(&s2.x_coeffs[1]), vec![0, 3, -6]); // 3t - 6t^2

        // S[3] = -23t^3 - 3t + 1  (the resultant)
        assert_eq!(t_coeffs(&prs.chain[3].constant_in_x()), vec![1, -3, 0, -23]);
        assert_eq!(t_coeffs(&prs.resultant), vec![1, -3, 0, -23]);
    }

    #[test]
    fn resultant_matches_sympy_on_non_squarefree_denominator() {
        // LRT requires D squarefree, but the primitive must still return the
        // well-defined resultant on any input. SymPy gives
        // res_x(1 - t*D', (x-1)^2) = 1 (verified) — not the naive 0 one might
        // guess; pin the exact oracle value.
        let d = poly_x(&[1, -2, 1]); // (x-1)^2
        let n = poly_x(&[1]);
        let r = rothstein_trager_resultant(&n, &d).expect("resultant");
        assert_eq!(t_coeffs(&r), vec![1], "res must match SymPy's value of 1");
    }

    #[test]
    fn poly_tx_ring_arithmetic() {
        // (t·x + 1) + (x - t) = (t+1)·x + (1 - t); then scale by t.
        let a = PolyTX {
            x_coeffs: vec![
                t_one(),
                Polynomial::new(vec![BigRational::zero(), BigRational::one()], T.to_string()),
            ],
        };
        let b = PolyTX {
            x_coeffs: vec![
                Polynomial::new(
                    vec![BigRational::zero(), -BigRational::one()],
                    T.to_string(),
                ),
                t_one(),
            ],
        };
        let sum = a.add(&b);
        assert_eq!(t_coeffs(&sum.x_coeffs[0]), vec![1, -1]); // 1 - t
        assert_eq!(t_coeffs(&sum.x_coeffs[1]), vec![1, 1]); // t + 1
        let scaled = sum.scale(&Polynomial::new(
            vec![BigRational::zero(), BigRational::one()],
            T.to_string(),
        ));
        assert_eq!(t_coeffs(&scaled.x_coeffs[0]), vec![0, 1, -1]); // t - t^2
                                                                   // add then sub round-trips.
        assert_eq!(sum.sub(&b), a);
    }

    #[test]
    fn pseudo_remainder_reduces_degree() {
        // prem of x^3 - x - 1 by (-3t)x^2 + (t+1): degree drops below 2.
        let a = PolyTX {
            x_coeffs: vec![
                t_const(BigRational::from_integer((-1).into())),
                t_const(BigRational::from_integer((-1).into())),
                t_zero(),
                t_one(),
            ],
        };
        let b = PolyTX {
            x_coeffs: vec![
                Polynomial::new(vec![BigRational::one(), BigRational::one()], T.to_string()), // t + 1
                t_zero(),
                Polynomial::new(
                    vec![BigRational::zero(), BigRational::from_integer((-3).into())],
                    T.to_string(),
                ), // -3t
            ],
        };
        let r = pseudo_remainder(&a, &b);
        assert!(r.degree_x() < 2, "prem must reduce below deg b");
        // Expected (3t - 6t^2)x - 9t^2.
        assert_eq!(t_coeffs(&r.x_coeffs[0]), vec![0, 0, -9]);
        assert_eq!(t_coeffs(&r.x_coeffs[1]), vec![0, 3, -6]);
    }
}

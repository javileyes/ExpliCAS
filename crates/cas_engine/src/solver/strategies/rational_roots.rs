//! RationalRootsStrategy — solves polynomial equations of degree ≥ 3
//! with all-numeric (rational) coefficients using the Rational Root Theorem
//! plus synthetic division (deflation).
//!
//! Pipeline:
//! 1. Extract univariate polynomial coefficients from `simplify(lhs - rhs)`
//! 2. Normalize to integer coefficients (scale by LCM of denominators)
//! 3. Enumerate candidate rational roots ±p/q
//! 4. Verify each candidate via exact Horner evaluation
//! 5. Deflate by confirmed roots (synthetic division)
//! 6. Delegate residual polynomial (degree ≤ 2) to existing strategies

use crate::engine::Simplifier;
use crate::error::CasError;
use crate::solver::isolation::contains_var;
use crate::solver::strategy::SolverStrategy;
use crate::solver::{SolveStep, SolverOptions};
use cas_ast::{Equation, Expr, ExprId, RelOp, SolutionSet};
use num_rational::BigRational;
use num_traits::{One, Signed, Zero};

/// Maximum number of candidate rational roots to try before bailing.
/// Prevents combinatorial blowup on polynomials with large leading/constant coefficients.
const MAX_CANDIDATES: usize = 200;

/// Maximum polynomial degree we attempt.
const MAX_DEGREE: usize = 10;

pub struct RationalRootsStrategy;

impl SolverStrategy for RationalRootsStrategy {
    fn name(&self) -> &str {
        "Rational Roots"
    }

    fn apply(
        &self,
        eq: &Equation,
        var: &str,
        simplifier: &mut Simplifier,
        _opts: &SolverOptions,
    ) -> Option<Result<(SolutionSet, Vec<SolveStep>), CasError>> {
        // Only handle equality
        if eq.op != RelOp::Eq {
            return None;
        }

        // Move everything to LHS: lhs - rhs = 0
        let diff = simplifier.context.add(Expr::Sub(eq.lhs, eq.rhs));
        let (sim_diff, _) = simplifier.simplify(diff);

        // Expand to canonical polynomial form
        let expanded = crate::expand::expand(&mut simplifier.context, sim_diff);

        // Extract polynomial coefficients: [a0, a1, ..., an] where poly = a0 + a1*x + ... + an*x^n
        let coeffs = extract_poly_coefficients(&mut simplifier.context, expanded, var)?;

        let degree = coeffs.len() - 1;

        // Only handle degree ≥ 3 (degree ≤ 2 is handled by QuadraticStrategy/Linear)
        if !(3..=MAX_DEGREE).contains(&degree) {
            return None;
        }

        // All coefficients must be numeric (rational)
        let rat_coeffs: Vec<BigRational> = coeffs
            .iter()
            .map(|&c| get_rational(&simplifier.context, c))
            .collect::<Option<Vec<_>>>()?;

        // All zeros check
        if rat_coeffs.iter().all(|c| c.is_zero()) {
            return Some(Ok((SolutionSet::AllReals, vec![])));
        }

        // Find rational roots
        let mut roots = Vec::new();
        let mut current_coeffs = rat_coeffs;

        loop {
            if current_coeffs.len() <= 1 {
                break;
            }

            // Strip trailing zeros (factor out x): each trailing zero = root at x=0
            while current_coeffs.len() > 1 && current_coeffs[0].is_zero() {
                current_coeffs.remove(0);
                let zero_expr = simplifier.context.num(0);
                roots.push(zero_expr);
            }

            if current_coeffs.len() <= 1 {
                break;
            }

            let deg = current_coeffs.len() - 1;
            if deg <= 2 {
                // Delegate to quadratic/linear
                break;
            }

            // Normalize to integer coefficients
            let int_coeffs = normalize_to_integers(&current_coeffs);

            // Generate candidates
            let candidates = rational_root_candidates(&int_coeffs);
            if candidates.is_empty() {
                break;
            }

            // Try each candidate
            let mut found_root = false;
            for candidate in &candidates {
                if horner_eval(&current_coeffs, candidate).is_zero() {
                    // Confirmed root! Add to results
                    let root_expr = rational_to_expr(&mut simplifier.context, candidate);
                    let (sim_root, _) = simplifier.simplify(root_expr);
                    roots.push(sim_root);

                    // Deflate
                    current_coeffs = synthetic_division(&current_coeffs, candidate);
                    found_root = true;
                    break; // restart candidate search on deflated polynomial
                }
            }

            if !found_root {
                break; // no rational roots found in remaining polynomial
            }
        }

        // Handle residual polynomial (degree ≤ 2)
        if current_coeffs.len() == 3 {
            // Quadratic: use quadratic formula directly
            let a = &current_coeffs[2];
            let b = &current_coeffs[1];
            let c = &current_coeffs[0];

            let discriminant =
                b.clone() * b.clone() - BigRational::from_integer(4.into()) * a.clone() * c.clone();

            if discriminant.is_zero() {
                let root = -b.clone() / (BigRational::from_integer(2.into()) * a.clone());
                let root_expr = rational_to_expr(&mut simplifier.context, &root);
                let (sim_root, _) = simplifier.simplify(root_expr);
                roots.push(sim_root);
            } else if discriminant.is_positive() {
                // Two real roots — compute symbolically for cleaner output
                let neg_b = rational_to_expr(&mut simplifier.context, &(-b.clone()));
                let disc_expr = rational_to_expr(&mut simplifier.context, &discriminant);
                let two_a = rational_to_expr(
                    &mut simplifier.context,
                    &(BigRational::from_integer(2.into()) * a.clone()),
                );

                let one = simplifier.context.num(1);
                let two = simplifier.context.num(2);
                let half = simplifier.context.add(Expr::Div(one, two));
                let sqrt_disc = simplifier.context.add(Expr::Pow(disc_expr, half));

                let num1 = simplifier.context.add(Expr::Sub(neg_b, sqrt_disc));
                let sol1 = simplifier.context.add(Expr::Div(num1, two_a));
                let (sim1, _) = simplifier.simplify(sol1);

                let num2 = simplifier.context.add(Expr::Add(neg_b, sqrt_disc));
                let sol2 = simplifier.context.add(Expr::Div(num2, two_a));
                let (sim2, _) = simplifier.simplify(sol2);

                roots.push(sim1);
                roots.push(sim2);
            }
            // else: discriminant < 0 → no real roots from this factor (complex)
        } else if current_coeffs.len() == 2 {
            // Linear: ax + b = 0 → x = -b/a
            let a = &current_coeffs[1];
            let b = &current_coeffs[0];
            if !a.is_zero() {
                let root = -b.clone() / a.clone();
                let root_expr = rational_to_expr(&mut simplifier.context, &root);
                let (sim_root, _) = simplifier.simplify(root_expr);
                roots.push(sim_root);
            }
        }
        // else: degree ≥ 3 with no rational roots — can't solve further, but we may have partial roots

        if roots.is_empty() {
            return None; // No roots found, let other strategies try
        }

        // Dedup roots
        use crate::ordering::compare_expr;
        use std::cmp::Ordering;
        roots.sort_by(|a, b| compare_expr(&simplifier.context, *a, *b));
        roots.dedup_by(|a, b| compare_expr(&simplifier.context, *a, *b) == Ordering::Equal);

        let steps = if simplifier.collect_steps() {
            vec![SolveStep {
                description: format!(
                    "Applied Rational Root Theorem to degree-{} polynomial",
                    degree
                ),
                equation_after: Equation {
                    lhs: expanded,
                    rhs: simplifier.context.num(0),
                    op: RelOp::Eq,
                },
                importance: crate::step::ImportanceLevel::Medium,
                substeps: vec![],
            }]
        } else {
            vec![]
        };

        Some(Ok((SolutionSet::Discrete(roots), steps)))
    }

    fn should_verify(&self) -> bool {
        true // Verify roots against original equation
    }
}

// =============================================================================
// Polynomial coefficient extraction
// =============================================================================

/// Extract polynomial coefficients from an expression in `var`.
/// Returns `[a0, a1, ..., an]` where `expr = a0 + a1*var + ... + an*var^n`.
/// Returns None if the expression isn't a polynomial in `var`.
fn extract_poly_coefficients(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    var: &str,
) -> Option<Vec<ExprId>> {
    // Collect (coefficient_expr, degree) pairs
    let mut terms: Vec<(ExprId, usize)> = Vec::new();
    let mut stack: Vec<(ExprId, bool)> = vec![(expr, true)];

    while let Some((curr, positive)) = stack.pop() {
        let curr_data = ctx.get(curr).clone();
        match curr_data {
            Expr::Add(l, r) => {
                stack.push((r, positive));
                stack.push((l, positive));
            }
            Expr::Sub(l, r) => {
                stack.push((r, !positive));
                stack.push((l, positive));
            }
            _ => {
                let (coeff, degree) = analyze_term_for_poly(ctx, curr, var)?;
                let term_coeff = if positive {
                    coeff
                } else {
                    ctx.add(Expr::Neg(coeff))
                };
                terms.push((term_coeff, degree as usize));
            }
        }
    }

    if terms.is_empty() {
        return None;
    }

    let max_degree = terms.iter().map(|(_, d)| *d).max().unwrap_or(0);
    if max_degree > MAX_DEGREE {
        return None;
    }

    // Build coefficient vector [a0, a1, ..., an]
    let zero = ctx.num(0);
    let mut coeffs = vec![zero; max_degree + 1];

    for (coeff, degree) in terms {
        coeffs[degree] = ctx.add(Expr::Add(coeffs[degree], coeff));
    }

    // Simplify each coefficient
    // (We need a simplifier, but we only have ctx here. We'll simplify in the caller.)
    Some(coeffs)
}

/// Analyze a single term to extract (coefficient, degree) for polynomial extraction.
/// Similar to `analyze_term_mut` but accepts any non-negative integer degree.
fn analyze_term_for_poly(
    ctx: &mut cas_ast::Context,
    term: ExprId,
    var: &str,
) -> Option<(ExprId, i32)> {
    if !contains_var(ctx, term, var) {
        return Some((term, 0));
    }

    let term_data = ctx.get(term).clone();

    match term_data {
        Expr::Variable(sym_id) if ctx.sym_name(sym_id) == var => Some((ctx.num(1), 1)),
        Expr::Pow(base, exp) => {
            if let Expr::Variable(sym_id) = ctx.get(base) {
                if ctx.sym_name(*sym_id) == var && !contains_var(ctx, exp, var) {
                    if let Expr::Number(n) = ctx.get(exp) {
                        if n.is_integer() && n.is_positive() {
                            let d: i32 = n.to_integer().try_into().ok()?;
                            return Some((ctx.num(1), d));
                        }
                    }
                }
            }
            None // Not a simple x^n
        }
        Expr::Mul(l, r) => {
            let l_has = contains_var(ctx, l, var);
            let r_has = contains_var(ctx, r, var);

            if l_has && r_has {
                let (c1, d1) = analyze_term_for_poly(ctx, l, var)?;
                let (c2, d2) = analyze_term_for_poly(ctx, r, var)?;
                let new_coeff = crate::build::mul2_raw(ctx, c1, c2);
                Some((new_coeff, d1 + d2))
            } else if l_has {
                let (c, d) = analyze_term_for_poly(ctx, l, var)?;
                let new_coeff = crate::build::mul2_raw(ctx, c, r);
                Some((new_coeff, d))
            } else {
                let (c, d) = analyze_term_for_poly(ctx, r, var)?;
                let new_coeff = crate::build::mul2_raw(ctx, l, c);
                Some((new_coeff, d))
            }
        }
        Expr::Div(l, r) => {
            if contains_var(ctx, r, var) {
                return None; // x in denominator → not a polynomial
            }
            let (c, d) = analyze_term_for_poly(ctx, l, var)?;
            let new_coeff = ctx.add(Expr::Div(c, r));
            Some((new_coeff, d))
        }
        Expr::Neg(inner) => {
            let (c, d) = analyze_term_for_poly(ctx, inner, var)?;
            let new_coeff = ctx.add(Expr::Neg(c));
            Some((new_coeff, d))
        }
        _ => None,
    }
}

// =============================================================================
// Rational number helpers
// =============================================================================

/// Try to extract a rational number from an expression.
fn get_rational(ctx: &cas_ast::Context, expr: ExprId) -> Option<BigRational> {
    match ctx.get(expr) {
        Expr::Number(n) => Some(n.clone()),
        Expr::Neg(inner) => get_rational(ctx, *inner).map(|n| -n),
        Expr::Div(l, r) => {
            let ln = get_rational(ctx, *l)?;
            let rn = get_rational(ctx, *r)?;
            if rn.is_zero() {
                None
            } else {
                Some(ln / rn)
            }
        }
        Expr::Add(l, r) => {
            let ln = get_rational(ctx, *l)?;
            let rn = get_rational(ctx, *r)?;
            Some(ln + rn)
        }
        Expr::Sub(l, r) => {
            let ln = get_rational(ctx, *l)?;
            let rn = get_rational(ctx, *r)?;
            Some(ln - rn)
        }
        Expr::Mul(l, r) => {
            let ln = get_rational(ctx, *l)?;
            let rn = get_rational(ctx, *r)?;
            Some(ln * rn)
        }
        _ => None,
    }
}

/// Convert a BigRational to an expression.
fn rational_to_expr(ctx: &mut cas_ast::Context, r: &BigRational) -> ExprId {
    ctx.add(Expr::Number(r.clone()))
}

// =============================================================================
// Rational Root Theorem implementation
// =============================================================================

/// Normalize rational coefficients to integers by multiplying by LCM of denominators.
fn normalize_to_integers(coeffs: &[BigRational]) -> Vec<num_bigint::BigInt> {
    use num_integer::Integer;

    // Find LCM of all denominators
    let mut lcm = num_bigint::BigInt::one();
    for c in coeffs {
        if !c.is_zero() {
            let d = c.denom().clone();
            lcm = lcm.clone() / lcm.clone().gcd(&d) * d;
        }
    }

    // Scale all coefficients
    coeffs
        .iter()
        .map(|c| (c * BigRational::from_integer(lcm.clone())).to_integer())
        .collect()
}

/// Generate candidate rational roots using the Rational Root Theorem.
/// Candidates: ±(divisors of |a0|) / (divisors of |an|)
fn rational_root_candidates(int_coeffs: &[num_bigint::BigInt]) -> Vec<BigRational> {
    let a0 = &int_coeffs[0]; // constant term
    let an = int_coeffs.last().unwrap(); // leading coefficient

    if an.is_zero() {
        return vec![];
    }

    // If constant term is 0, x=0 is a root (handled by caller's zero-stripping)
    if a0.is_zero() {
        return vec![BigRational::zero()];
    }

    let a0_abs = a0.abs();
    let an_abs = an.abs();

    let divisors_a0 = small_divisors(&a0_abs);
    let divisors_an = small_divisors(&an_abs);

    if divisors_a0.is_empty() || divisors_an.is_empty() {
        return vec![];
    }

    // Check candidate count doesn't exceed limit
    let candidate_count = divisors_a0.len() * divisors_an.len() * 2;
    if candidate_count > MAX_CANDIDATES {
        return vec![]; // Too many candidates, bail
    }

    let mut candidates = Vec::with_capacity(candidate_count);
    let mut seen = std::collections::HashSet::new();

    for p in &divisors_a0 {
        for q in &divisors_an {
            let candidate = BigRational::new(p.clone(), q.clone());
            // Reduce to canonical form for dedup
            let key = (candidate.numer().clone(), candidate.denom().clone());
            if seen.insert(key.clone()) {
                candidates.push(candidate.clone());
                let neg_key = (-key.0.clone(), key.1);
                if seen.insert(neg_key) {
                    candidates.push(-candidate);
                }
            }
        }
    }

    candidates
}

/// Get all positive divisors of |n|. Returns empty if n has too many divisors.
fn small_divisors(n: &num_bigint::BigInt) -> Vec<num_bigint::BigInt> {
    use num_bigint::BigInt;

    if n.is_zero() {
        return vec![];
    }

    let n_abs = n.abs();

    // For efficiency, limit to numbers that fit in u64
    let n_u64: u64 = match n_abs.try_into() {
        Ok(v) => v,
        Err(_) => return vec![], // Number too large
    };

    if n_u64 == 0 {
        return vec![];
    }

    let mut divs = Vec::new();
    let sqrt_n = (n_u64 as f64).sqrt() as u64;

    for i in 1..=sqrt_n {
        if n_u64.is_multiple_of(i) {
            divs.push(BigInt::from(i));
            if i != n_u64 / i {
                divs.push(BigInt::from(n_u64 / i));
            }
        }
    }

    // Limit total divisors
    if divs.len() > 50 {
        return vec![]; // Highly composite number, too many candidates
    }

    divs
}

// =============================================================================
// Polynomial evaluation and deflation
// =============================================================================

/// Evaluate polynomial at a rational point using Horner's method.
/// coeffs = [a0, a1, ..., an] → a0 + a1*x + ... + an*x^n
fn horner_eval(coeffs: &[BigRational], x: &BigRational) -> BigRational {
    // Horner: start from highest degree
    let mut result = BigRational::zero();
    for c in coeffs.iter().rev() {
        result = result * x.clone() + c.clone();
    }
    result
}

/// Synthetic division: divide polynomial by (x - root).
/// Input: coeffs = [a0, a1, ..., an] (low-to-high degree)
/// Output: quotient coefficients [b0, b1, ..., b(n-1)]
fn synthetic_division(coeffs: &[BigRational], root: &BigRational) -> Vec<BigRational> {
    let n = coeffs.len();
    if n <= 1 {
        return vec![];
    }

    // Work from highest to lowest degree
    // Standard synthetic division: b_{n-1} = a_n, then b_{i-1} = a_i + root * b_i
    let mut quotient = vec![BigRational::zero(); n - 1];

    // Start with leading coefficient
    quotient[n - 2] = coeffs[n - 1].clone();

    // Descend through degrees
    for i in (0..n - 2).rev() {
        quotient[i] = coeffs[i + 1].clone() + root.clone() * quotient[i + 1].clone();
    }

    quotient
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_horner_eval() {
        // p(x) = x^3 - x = 0 + (-1)*x + 0*x^2 + 1*x^3
        // coeffs = [0, -1, 0, 1]
        let coeffs = vec![
            BigRational::zero(),
            BigRational::from_integer((-1).into()),
            BigRational::zero(),
            BigRational::one(),
        ];

        assert!(horner_eval(&coeffs, &BigRational::zero()).is_zero());
        assert!(horner_eval(&coeffs, &BigRational::one()).is_zero());
        assert!(horner_eval(&coeffs, &BigRational::from_integer((-1).into())).is_zero());
        assert!(!horner_eval(&coeffs, &BigRational::from_integer(2.into())).is_zero());
    }

    #[test]
    fn test_synthetic_division() {
        // (x^3 - x) / (x - 0) = x^2 - 1
        // coeffs = [0, -1, 0, 1], root = 0
        let coeffs = vec![
            BigRational::zero(),
            BigRational::from_integer((-1).into()),
            BigRational::zero(),
            BigRational::one(),
        ];
        let quotient = synthetic_division(&coeffs, &BigRational::zero());
        // Expected: x^2 - 1 = [-1, 0, 1]
        assert_eq!(quotient.len(), 3);
        assert_eq!(quotient[0], BigRational::from_integer((-1).into()));
        assert!(quotient[1].is_zero());
        assert_eq!(quotient[2], BigRational::one());

        // (x^2 - 1) / (x - 1) = x + 1
        let quotient2 = synthetic_division(&quotient, &BigRational::one());
        assert_eq!(quotient2.len(), 2);
        assert_eq!(quotient2[0], BigRational::one()); // constant = 1
        assert_eq!(quotient2[1], BigRational::one()); // x coeff = 1
    }

    #[test]
    fn test_small_divisors() {
        let divs = small_divisors(&num_bigint::BigInt::from(12));
        let mut vals: Vec<u64> = divs.iter().map(|d| d.try_into().unwrap()).collect();
        vals.sort();
        assert_eq!(vals, vec![1, 2, 3, 4, 6, 12]);
    }
}

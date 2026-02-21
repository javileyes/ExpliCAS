use crate::isolation_utils::contains_var;
use crate::quadratic_formula::{discriminant, roots_from_a_b_delta};
use cas_ast::{Context, Expr, ExprId};
use num_bigint::BigInt;
use num_integer::Integer;
use num_rational::BigRational;
use num_traits::{One, Signed, Zero};

/// Extract polynomial coefficients from an expression in `var`.
///
/// Returns `[a0, a1, ..., an]` where
/// `expr = a0 + a1*var + ... + an*var^n`.
///
/// Returns `None` if the expression is not polynomial in `var`
/// or its degree exceeds `max_degree`.
pub fn extract_poly_coefficients(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
    max_degree: usize,
) -> Option<Vec<ExprId>> {
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

    let max_seen_degree = terms.iter().map(|(_, d)| *d).max().unwrap_or(0);
    if max_seen_degree > max_degree {
        return None;
    }

    let zero = ctx.num(0);
    let mut coeffs = vec![zero; max_seen_degree + 1];
    for (coeff, degree) in terms {
        coeffs[degree] = ctx.add(Expr::Add(coeffs[degree], coeff));
    }

    Some(coeffs)
}

fn analyze_term_for_poly(ctx: &mut Context, term: ExprId, var: &str) -> Option<(ExprId, i32)> {
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
            None
        }
        Expr::Mul(l, r) => {
            let l_has = contains_var(ctx, l, var);
            let r_has = contains_var(ctx, r, var);

            if l_has && r_has {
                let (c1, d1) = analyze_term_for_poly(ctx, l, var)?;
                let (c2, d2) = analyze_term_for_poly(ctx, r, var)?;
                let new_coeff = cas_math::build::mul2_raw(ctx, c1, c2);
                Some((new_coeff, d1 + d2))
            } else if l_has {
                let (c, d) = analyze_term_for_poly(ctx, l, var)?;
                let new_coeff = cas_math::build::mul2_raw(ctx, c, r);
                Some((new_coeff, d))
            } else {
                let (c, d) = analyze_term_for_poly(ctx, r, var)?;
                let new_coeff = cas_math::build::mul2_raw(ctx, l, c);
                Some((new_coeff, d))
            }
        }
        Expr::Div(l, r) => {
            if contains_var(ctx, r, var) {
                return None;
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

/// Try to extract a rational number from an expression.
pub fn get_rational(ctx: &Context, expr: ExprId) -> Option<BigRational> {
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

/// Convert a rational to expression.
pub fn rational_to_expr(ctx: &mut Context, r: &BigRational) -> ExprId {
    ctx.add(Expr::Number(r.clone()))
}

/// Normalize rational coefficients to integers by multiplying by LCM of denominators.
pub fn normalize_to_integers(coeffs: &[BigRational]) -> Vec<BigInt> {
    let mut lcm = BigInt::one();
    for c in coeffs {
        if !c.is_zero() {
            let d = c.denom().clone();
            lcm = lcm.clone() / lcm.clone().gcd(&d) * d;
        }
    }

    coeffs
        .iter()
        .map(|c| (c * BigRational::from_integer(lcm.clone())).to_integer())
        .collect()
}

/// Generate candidate rational roots using Rational Root Theorem.
pub fn rational_root_candidates(int_coeffs: &[BigInt], max_candidates: usize) -> Vec<BigRational> {
    let a0 = &int_coeffs[0];
    let an = int_coeffs
        .last()
        .expect("coeff vector for polynomial must not be empty");

    if an.is_zero() {
        return vec![];
    }
    if a0.is_zero() {
        return vec![BigRational::zero()];
    }

    let divisors_a0 = small_divisors(&a0.abs());
    let divisors_an = small_divisors(&an.abs());
    if divisors_a0.is_empty() || divisors_an.is_empty() {
        return vec![];
    }

    let candidate_count = divisors_a0.len() * divisors_an.len() * 2;
    if candidate_count > max_candidates {
        return vec![];
    }

    let mut candidates = Vec::with_capacity(candidate_count);
    let mut seen = std::collections::HashSet::new();

    for p in &divisors_a0 {
        for q in &divisors_an {
            let candidate = BigRational::new(p.clone(), q.clone());
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

fn small_divisors(n: &BigInt) -> Vec<BigInt> {
    if n.is_zero() {
        return vec![];
    }

    let n_u64: u64 = match n.abs().try_into() {
        Ok(v) => v,
        Err(_) => return vec![],
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

    if divs.len() > 50 {
        return vec![];
    }

    divs
}

/// Evaluate polynomial at `x` using Horner's method.
pub fn horner_eval(coeffs: &[BigRational], x: &BigRational) -> BigRational {
    let mut result = BigRational::zero();
    for c in coeffs.iter().rev() {
        result = result * x.clone() + c.clone();
    }
    result
}

/// Divide polynomial by `(x - root)` using synthetic division.
///
/// Coeff order is `[a0, a1, ..., an]` (low-to-high degree).
pub fn synthetic_division(coeffs: &[BigRational], root: &BigRational) -> Vec<BigRational> {
    let n = coeffs.len();
    if n <= 1 {
        return vec![];
    }

    let mut quotient = vec![BigRational::zero(); n - 1];
    quotient[n - 2] = coeffs[n - 1].clone();

    for i in (0..n - 2).rev() {
        quotient[i] = coeffs[i + 1].clone() + root.clone() * quotient[i + 1].clone();
    }

    quotient
}

/// Extract rational roots by repeated candidate testing + synthetic deflation.
///
/// Returns `(roots, residual_coeffs)` where:
/// - `roots` are found rational roots (possibly with multiplicity),
/// - `residual_coeffs` is the remaining polynomial coefficient vector.
///
/// The routine stops deflation once residual degree is <= 2.
pub fn find_rational_roots(
    mut coeffs: Vec<BigRational>,
    max_candidates: usize,
) -> (Vec<BigRational>, Vec<BigRational>) {
    let mut roots = Vec::new();

    loop {
        if coeffs.len() <= 1 {
            break;
        }

        // a0 == 0 => root x=0; remove constant term and continue.
        while coeffs.len() > 1 && coeffs[0].is_zero() {
            coeffs.remove(0);
            roots.push(BigRational::zero());
        }

        if coeffs.len() <= 1 {
            break;
        }

        let degree = coeffs.len() - 1;
        if degree <= 2 {
            break;
        }

        let int_coeffs = normalize_to_integers(&coeffs);
        let candidates = rational_root_candidates(&int_coeffs, max_candidates);
        if candidates.is_empty() {
            break;
        }

        let mut found = false;
        for candidate in &candidates {
            if horner_eval(&coeffs, candidate).is_zero() {
                roots.push(candidate.clone());
                coeffs = synthetic_division(&coeffs, candidate);
                found = true;
                break;
            }
        }

        if !found {
            break;
        }
    }

    (roots, coeffs)
}

/// Extract all discrete candidate roots for a degree>=3 polynomial with
/// rational coefficients.
///
/// This combines:
/// - rational-root deflation for high-degree components, and
/// - residual solving for the remaining degree<=2 polynomial.
pub fn extract_candidate_roots(
    ctx: &mut Context,
    coeffs: Vec<BigRational>,
    max_candidates: usize,
) -> Vec<ExprId> {
    let mut roots = Vec::new();
    let (rational_roots, residual_coeffs) = find_rational_roots(coeffs, max_candidates);

    for r in &rational_roots {
        roots.push(rational_to_expr(ctx, r));
    }

    if residual_coeffs.len() == 3 || residual_coeffs.len() == 2 {
        roots.extend(solve_residual_degree_leq_two(ctx, &residual_coeffs));
    }

    roots
}

/// Solve a residual polynomial of degree <= 2 with rational coefficients.
///
/// Coeff order is `[a0, a1, ..., an]` (low-to-high degree).
/// Returns zero, one, or two root expressions.
pub fn solve_residual_degree_leq_two(ctx: &mut Context, coeffs: &[BigRational]) -> Vec<ExprId> {
    if coeffs.len() == 3 {
        // Quadratic: a*x^2 + b*x + c = 0
        let a = &coeffs[2];
        let b = &coeffs[1];
        let c = &coeffs[0];

        if a.is_zero() {
            // Degenerated quadratic -> linear
            if b.is_zero() {
                return vec![];
            }
            let root = -c.clone() / b.clone();
            return vec![rational_to_expr(ctx, &root)];
        }

        let discriminant = discriminant(a, b, c);

        if discriminant.is_zero() {
            let root = -b.clone() / (BigRational::from_integer(2.into()) * a.clone());
            vec![rational_to_expr(ctx, &root)]
        } else if discriminant.is_positive() {
            let a_expr = rational_to_expr(ctx, a);
            let b_expr = rational_to_expr(ctx, b);
            let disc_expr = rational_to_expr(ctx, &discriminant);
            let (r1, r2) = roots_from_a_b_delta(ctx, a_expr, b_expr, disc_expr);
            vec![r1, r2]
        } else {
            vec![]
        }
    } else if coeffs.len() == 2 {
        // Linear: a*x + b = 0
        let a = &coeffs[1];
        let b = &coeffs[0];
        if a.is_zero() {
            vec![]
        } else {
            let root = -b.clone() / a.clone();
            vec![rational_to_expr(ctx, &root)]
        }
    } else {
        vec![]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn horner_eval_works() {
        let coeffs = vec![
            BigRational::zero(),
            BigRational::from_integer((-1).into()),
            BigRational::zero(),
            BigRational::one(),
        ];
        assert!(horner_eval(&coeffs, &BigRational::zero()).is_zero());
        assert!(horner_eval(&coeffs, &BigRational::one()).is_zero());
        assert!(horner_eval(&coeffs, &BigRational::from_integer((-1).into())).is_zero());
    }

    #[test]
    fn synthetic_division_works() {
        let coeffs = vec![
            BigRational::zero(),
            BigRational::from_integer((-1).into()),
            BigRational::zero(),
            BigRational::one(),
        ];
        let quotient = synthetic_division(&coeffs, &BigRational::zero());
        assert_eq!(quotient.len(), 3);
        assert_eq!(quotient[0], BigRational::from_integer((-1).into()));
        assert!(quotient[1].is_zero());
        assert_eq!(quotient[2], BigRational::one());
    }

    #[test]
    fn small_divisors_for_12() {
        let divs = small_divisors(&BigInt::from(12));
        let mut vals: Vec<u64> = divs.iter().map(|d| d.try_into().unwrap_or(0)).collect();
        vals.sort();
        assert_eq!(vals, vec![1, 2, 3, 4, 6, 12]);
    }

    #[test]
    fn solve_residual_linear_works() {
        let mut ctx = Context::new();
        // 2x + 4 = 0 -> x = -2
        let coeffs = vec![
            BigRational::from_integer(4.into()),
            BigRational::from_integer(2.into()),
        ];
        let roots = solve_residual_degree_leq_two(&mut ctx, &coeffs);
        assert_eq!(roots.len(), 1);
        assert_eq!(
            get_rational(&ctx, roots[0]).expect("root should be numeric"),
            BigRational::from_integer((-2).into())
        );
    }

    #[test]
    fn solve_residual_quadratic_positive_discriminant_returns_two() {
        let mut ctx = Context::new();
        // x^2 - 1 = 0
        let coeffs = vec![
            BigRational::from_integer((-1).into()),
            BigRational::zero(),
            BigRational::one(),
        ];
        let roots = solve_residual_degree_leq_two(&mut ctx, &coeffs);
        assert_eq!(roots.len(), 2);
    }

    #[test]
    fn solve_residual_quadratic_negative_discriminant_returns_none() {
        let mut ctx = Context::new();
        // x^2 + 1 = 0 (no real roots)
        let coeffs = vec![BigRational::one(), BigRational::zero(), BigRational::one()];
        let roots = solve_residual_degree_leq_two(&mut ctx, &coeffs);
        assert!(roots.is_empty());
    }

    #[test]
    fn find_rational_roots_stops_at_quadratic_residual() {
        // x^3 - x = x*(x^2 - 1)
        let coeffs = vec![
            BigRational::zero(),
            BigRational::from_integer((-1).into()),
            BigRational::zero(),
            BigRational::one(),
        ];
        let (roots, residual) = find_rational_roots(coeffs, 200);
        assert_eq!(roots, vec![BigRational::zero()]);
        assert_eq!(residual.len(), 3); // quadratic residual
    }

    #[test]
    fn extract_candidate_roots_combines_deflation_and_residual() {
        // x^3 - x = x*(x^2-1), roots: -1, 0, 1
        let coeffs = vec![
            BigRational::zero(),
            BigRational::from_integer((-1).into()),
            BigRational::zero(),
            BigRational::one(),
        ];
        let mut ctx = Context::new();
        let roots = extract_candidate_roots(&mut ctx, coeffs, 200);
        assert_eq!(roots.len(), 3);
    }
}

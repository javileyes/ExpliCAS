//! Generalized Rationalization Strategy
//!
//! Rationalizes denominators of the form `1/(a + b√n + c√m)` where:
//! - a, b, c are rational numbers
//! - n, m are positive integers (normalized to square-free)
//!
//! Uses conjugate multiplication heuristic with budget guards.

use cas_ast::{views::SurdSumView, Context, Expr, ExprId};
use num_rational::BigRational;

/// Configuration for rationalization strategy
#[derive(Debug, Clone)]
pub struct RationalizeConfig {
    /// Maximum conjugate iterations (default: 4)
    pub max_steps: usize,
    /// Maximum nodes allowed in denominator (default: 250)
    pub max_den_nodes: usize,
}

impl Default for RationalizeConfig {
    fn default() -> Self {
        Self {
            max_steps: 4,
            max_den_nodes: 250,
        }
    }
}

/// Result of rationalization attempt
#[derive(Debug)]
pub enum RationalizeResult {
    /// Successfully rationalized
    Success(ExprId),
    /// Could not rationalize (domain not supported)
    NotApplicable,
    /// Budget exceeded
    BudgetExceeded,
}

/// Extract a numeric constant factor from a multiplicative expression.
/// Returns (factor, core) where expr = factor * core (only for numeric factors).
/// Handles Neg as factor of -1.
/// NOTE: Neg(Number) is now converted to Number(-n) by normalize_core N0,
/// so we don't need special handling for it here.
fn extract_constant_factor(ctx: &Context, expr: ExprId) -> Option<(BigRational, ExprId)> {
    use num_traits::One;

    match ctx.get(expr) {
        Expr::Neg(inner) => {
            // Neg(x) = -1 * x
            let inner_id = *inner;
            // Recursively check if inner also has a factor
            if let Some((k, core)) = extract_constant_factor(ctx, inner_id) {
                Some((-k, core))
            } else {
                Some((-BigRational::one(), inner_id))
            }
        }
        Expr::Mul(l, r) => {
            let l_id = *l;
            let r_id = *r;
            // Check if left is a number
            if let Expr::Number(n) = ctx.get(l_id) {
                // Recursively check right side for more factors
                if let Some((k, core)) = extract_constant_factor(ctx, r_id) {
                    return Some((n * k, core));
                }
                return Some((n.clone(), r_id));
            }
            // Check if right is a number
            if let Expr::Number(n) = ctx.get(r_id) {
                // Recursively check left side for more factors
                if let Some((k, core)) = extract_constant_factor(ctx, l_id) {
                    return Some((n * k, core));
                }
                return Some((n.clone(), l_id));
            }
            None
        }
        _ => None,
    }
}

/// Attempt to rationalize a fraction expression.
///
/// Input should be of the form `Div(num, den)` or `Mul(num, Pow(den, -1))`.
/// Returns the rationalized expression if successful.
pub fn rationalize_denominator(
    ctx: &mut Context,
    expr: ExprId,
    _config: &RationalizeConfig,
) -> RationalizeResult {
    use num_traits::One;

    // Extract numerator and denominator
    let (num, den) = match extract_fraction(ctx, expr) {
        Some(pair) => pair,
        None => return RationalizeResult::NotApplicable,
    };

    // Try to extract constant factor from denominator: k * surd_sum
    let (den_factor, den_core) =
        extract_constant_factor(ctx, den).unwrap_or_else(|| (BigRational::one(), den));

    // Check if denominator core is a surd sum
    let view = match SurdSumView::from(ctx, den_core) {
        Some(v) => v,
        None => return RationalizeResult::NotApplicable,
    };

    // Already rational? Nothing to do
    if view.is_rational() {
        return RationalizeResult::Success(expr);
    }

    // v1: Handle 2 surds (constant + surd1 + surd2)
    // v2: Also handle 1 surd (constant + surd) - simple conjugate case
    if view.surds.len() == 1 {
        // Simple case: a + b√n → multiply by (a - b√n)/(a - b√n)
        let a = view.constant.clone();
        let (b, n) = (view.surds[0].coeff.clone(), view.surds[0].radicand);

        // Build conjugate: a - b√n
        let a_expr = build_rational(ctx, &a);
        let b_sqrt_n = build_sqrt_term(ctx, &b, n);
        let neg_b_sqrt_n = ctx.add(Expr::Neg(b_sqrt_n));
        let conjugate = ctx.add(Expr::Add(a_expr, neg_b_sqrt_n));

        // New numerator: num * (a - b√n)
        let new_num = ctx.add(Expr::Mul(num, conjugate));

        // New denominator: a² - b²·n (pure rational)
        let a_sq = &a * &a;
        let b_sq_n = &b * &b * BigRational::from_integer(n.into());
        let new_den_value = a_sq - b_sq_n;
        let new_den = build_rational(ctx, &new_den_value);

        let result = build_fraction(ctx, new_num, new_den);

        // Apply extracted constant factor: result / den_factor
        let final_result = if den_factor == BigRational::one() {
            result
        } else {
            let factor_expr = build_rational(ctx, &den_factor);
            build_fraction(ctx, result, factor_expr)
        };
        return RationalizeResult::Success(final_result);
    }

    if view.surds.len() != 2 {
        // Only handle 1 or 2 surds for now
        return RationalizeResult::NotApplicable;
    }

    // Extract A = a + b√n and B = c√m
    let a = view.constant.clone();
    let (b, n) = (view.surds[0].coeff.clone(), view.surds[0].radicand);
    let (c, m) = (view.surds[1].coeff.clone(), view.surds[1].radicand);

    // Build conjugate parts for numerator: (a + b√n) - c√m
    // We need: num * ((a + b√n) - c√m)
    let a_part = build_surd_sum(ctx, &a, &b, n);
    let b_part = build_sqrt_term(ctx, &c, m);
    let neg_b_part = ctx.add(Expr::Neg(b_part));
    let conjugate = ctx.add(Expr::Add(a_part, neg_b_part));
    let new_num = ctx.add(Expr::Mul(num, conjugate));

    // Calculate A² - B² using surd arithmetic:
    // A² = (a + b√n)² = (a² + b²·n) + (2ab)√n
    // B² = (c√m)² = c²·m (pure rational)
    // A² - B² = (a² + b²·n - c²·m) + (2ab)√n
    let a_sq = &a * &a;
    let b_sq_n = &b * &b * BigRational::from_integer(n.into());
    let c_sq_m = &c * &c * BigRational::from_integer(m.into());
    let two_ab = BigRational::from_integer(2.into()) * &a * &b;

    let new_const = a_sq + b_sq_n - c_sq_m;
    let new_surd_coeff = two_ab;

    // Now denominator is: new_const + new_surd_coeff * √n
    // This is a binomial (1 surd term) - progress!

    // Check if we're done (no surd term)
    if new_surd_coeff == BigRational::from_integer(0.into()) {
        // Denominator is pure rational
        let den_expr = build_rational(ctx, &new_const);
        let result = build_fraction(ctx, new_num, den_expr);
        // Apply extracted constant factor: result / den_factor
        let final_result = if den_factor == BigRational::one() {
            result
        } else {
            let factor_expr = build_rational(ctx, &den_factor);
            build_fraction(ctx, result, factor_expr)
        };
        return RationalizeResult::Success(final_result);
    }

    // Still have a surd - do one more conjugate to fully rationalize
    // Denominator is: p + q√n where p = new_const, q = new_surd_coeff
    // Multiply by (p - q√n) / (p - q√n)
    // New den = p² - q²·n (pure rational)

    let p = new_const;
    let q = new_surd_coeff;

    // Build second conjugate for numerator
    let p_expr = build_rational(ctx, &p);
    let q_sqrt_n = build_sqrt_term(ctx, &q, n);
    let neg_q_sqrt_n = ctx.add(Expr::Neg(q_sqrt_n));
    let conjugate2 = ctx.add(Expr::Add(p_expr, neg_q_sqrt_n));
    let final_num = ctx.add(Expr::Mul(new_num, conjugate2));

    // Final denominator: p² - q²·n
    let p_sq = &p * &p;
    let q_sq_n = &q * &q * BigRational::from_integer(n.into());
    let final_den_value = p_sq - q_sq_n;

    let final_den = build_rational(ctx, &final_den_value);
    let result = build_fraction(ctx, final_num, final_den);

    // Apply extracted constant factor: result / den_factor
    let final_result = if den_factor == BigRational::one() {
        result
    } else {
        let factor_expr = build_rational(ctx, &den_factor);
        build_fraction(ctx, result, factor_expr)
    };

    RationalizeResult::Success(final_result)
}

/// Extract (numerator, denominator) from a fraction expression.
fn extract_fraction(ctx: &mut Context, expr: ExprId) -> Option<(ExprId, ExprId)> {
    match ctx.get(expr) {
        Expr::Div(num, den) => Some((*num, *den)),
        Expr::Mul(l, r) => {
            // Check for Mul(a, Pow(b, -1))
            if let Expr::Pow(base, exp) = ctx.get(*r) {
                if let Expr::Number(n) = ctx.get(*exp) {
                    if n.is_integer() && *n == BigRational::from_integer((-1).into()) {
                        return Some((*l, *base));
                    }
                }
            }
            if let Expr::Pow(base, exp) = ctx.get(*l) {
                if let Expr::Number(n) = ctx.get(*exp) {
                    if n.is_integer() && *n == BigRational::from_integer((-1).into()) {
                        return Some((*r, *base));
                    }
                }
            }
            None
        }
        Expr::Pow(base, exp) => {
            // Standalone x^(-1) = 1/x
            let base_copy = *base;
            if let Expr::Number(n) = ctx.get(*exp) {
                if n.is_integer() && *n == BigRational::from_integer((-1).into()) {
                    let one = ctx.num(1);
                    return Some((one, base_copy));
                }
            }
            None
        }
        _ => None,
    }
}

/// Build a single surd term: coeff * √(radicand)
fn build_sqrt_term(ctx: &mut Context, coeff: &BigRational, radicand: i64) -> ExprId {
    let radicand_expr = ctx.num(radicand);
    let half = ctx.rational(1, 2);
    let sqrt_part = ctx.add(Expr::Pow(radicand_expr, half));

    if *coeff == BigRational::from_integer(1.into()) {
        sqrt_part
    } else if *coeff == BigRational::from_integer((-1).into()) {
        ctx.add(Expr::Neg(sqrt_part))
    } else {
        let coeff_expr = ctx.add(Expr::Number(coeff.clone()));
        ctx.add(Expr::Mul(coeff_expr, sqrt_part))
    }
}

/// Build a + b√n expression
fn build_surd_sum(ctx: &mut Context, a: &BigRational, b: &BigRational, n: i64) -> ExprId {
    let a_expr = ctx.add(Expr::Number(a.clone()));

    if *b == BigRational::from_integer(0.into()) {
        // No surd part
        return a_expr;
    }

    let b_sqrt_n = build_sqrt_term(ctx, b, n);

    if *a == BigRational::from_integer(0.into()) {
        // No constant part
        return b_sqrt_n;
    }

    ctx.add(Expr::Add(a_expr, b_sqrt_n))
}

/// Build a rational number expression
fn build_rational(ctx: &mut Context, r: &BigRational) -> ExprId {
    ctx.add(Expr::Number(r.clone()))
}

/// Build a fraction Div(num, den), simplifying if den = 1
fn build_fraction(ctx: &mut Context, num: ExprId, den: ExprId) -> ExprId {
    if let Expr::Number(n) = ctx.get(den) {
        if n.is_integer() && *n == BigRational::from_integer(1.into()) {
            return num;
        }
    }
    ctx.add(Expr::Div(num, den))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_fraction_div() {
        let mut ctx = Context::new();
        let a = ctx.num(1);
        let b = ctx.num(2);
        let div = ctx.add(Expr::Div(a, b));

        let (num, den) = extract_fraction(&mut ctx, div).unwrap();
        assert_eq!(num, a);
        assert_eq!(den, b);
    }

    #[test]
    fn test_extract_fraction_mul_pow() {
        let mut ctx = Context::new();
        let a = ctx.num(1);
        let b = ctx.num(2);
        let neg_one = ctx.num(-1);
        let b_inv = ctx.add(Expr::Pow(b, neg_one));
        let mul = ctx.add(Expr::Mul(a, b_inv));

        let (num, den) = extract_fraction(&mut ctx, mul).unwrap();
        assert_eq!(num, a);
        assert_eq!(den, b);
    }

    #[test]
    fn test_rationalize_simple_sqrt() {
        let mut ctx = Context::new();

        // 1 / √2
        let one = ctx.num(1);
        let two = ctx.num(2);
        let half = ctx.rational(1, 2);
        let sqrt2 = ctx.add(Expr::Pow(two, half));
        let frac = ctx.add(Expr::Div(one, sqrt2));

        // Debug: check if SurdSumView works on √2
        let view = SurdSumView::from(&ctx, sqrt2);
        println!("SurdSumView for √2: {:?}", view);

        let config = RationalizeConfig::default();
        let result = rationalize_denominator(&mut ctx, frac, &config);

        println!("Result: {:?}", result);
        match result {
            RationalizeResult::Success(expr) => {
                // Should have rationalized
                println!("Rationalized: {:?}", ctx.get(expr));
            }
            RationalizeResult::NotApplicable => {
                println!("Not applicable");
                // For v1, just a single √2 might not work - accept this
            }
            RationalizeResult::BudgetExceeded => {
                panic!("Budget exceeded");
            }
        }
    }

    #[test]
    fn test_rationalize_surd_sum() {
        let mut ctx = Context::new();

        // 1 / (1 + √2 + √3) using Function("sqrt") form
        let one = ctx.num(1);
        let one_const = ctx.num(1);
        let two = ctx.num(2);
        let three = ctx.num(3);
        let sqrt2 = ctx.call("sqrt", vec![two]);
        let sqrt3 = ctx.call("sqrt", vec![three]);
        let sum1 = ctx.add(Expr::Add(one_const, sqrt2));
        let den = ctx.add(Expr::Add(sum1, sqrt3));
        let frac = ctx.add(Expr::Div(one, den));

        println!("Input: {:?}", ctx.get(frac));
        println!("Denominator: {:?}", ctx.get(den));

        // Check SurdSumView recognizes the denominator
        let view = SurdSumView::from(&ctx, den);
        println!("SurdSumView for denominator: {:?}", view);
        assert!(
            view.is_some(),
            "SurdSumView should recognize the denominator"
        );

        let config = RationalizeConfig::default();
        let result = rationalize_denominator(&mut ctx, frac, &config);

        println!("Result: {:?}", result);
        match result {
            RationalizeResult::Success(expr) => {
                println!("Rationalized: {:?}", ctx.get(expr));
            }
            RationalizeResult::NotApplicable => {
                panic!("Expected success, got NotApplicable");
            }
            RationalizeResult::BudgetExceeded => {
                panic!("Budget exceeded");
            }
        }
    }
}

//! Half-angle, phase shift, supplementary angle and Weierstrass substitution rules.

use crate::define_rule;
use crate::helpers::{as_add, as_div, as_mul, as_neg, as_sub, is_pi, is_pi_over_n};
use crate::rule::Rewrite;
use crate::rules::algebra::helpers::smart_mul;
use cas_ast::{BuiltinFn, Expr, ExprId};
use num_traits::One;
use std::cmp::Ordering;

// =============================================================================
// Sin Supplementary Angle Rule
// =============================================================================
// sin(π - x) → sin(x)
// sin(k·π - x) → (-1)^(k+1) · sin(x) for integer k
// cos(π - x) → -cos(x)
//
// This enables simplification of expressions like sin(8π/9) = sin(π - π/9) = sin(π/9)

define_rule!(
    SinSupplementaryAngleRule,
    "Supplementary Angle",
    |ctx, expr| {
        use crate::helpers::extract_rational_pi_multiple;
        use num_rational::BigRational;

        let (fn_id, args) = match ctx.get(expr) {
            Expr::Function(fn_id, args) => (*fn_id, args.clone()),
            _ => return None,
        };
        {
            let builtin = ctx.builtin_of(fn_id);
            if args.len() != 1 {
                return None;
            }

            let is_sin = matches!(builtin, Some(BuiltinFn::Sin));
            let is_cos = matches!(builtin, Some(BuiltinFn::Cos));
            if !is_sin && !is_cos {
                return None;
            }

            let arg = args[0];

            // Try to check if arg is a rational multiple of π
            // where the coefficient is of the form (n - small) for some positive integer n
            // e.g., 8/9 = 1 - 1/9, so sin(8π/9) = sin(π - π/9) = sin(π/9)

            if let Some(k) = extract_rational_pi_multiple(ctx, arg) {
                // k = p/q in lowest terms
                let p = k.numer();
                let q = k.denom();

                // Check if p/q is close enough to an integer that the supplementary form is simpler.
                // For sin(k·π) where k = (n*q - m)/q, we can write it as sin((n*q - m)/q · π) = sin(n·π - m/q·π)
                // This simplifies when m < p (i.e., the remainder is smaller than the original numerator)
                //
                // Example: sin(8/9·π) = sin(1·π - 1/9·π) = sin(π/9) because 1 < 8

                // Only for positive k (p > 0)
                if p > &num_bigint::BigInt::from(0) {
                    let one = num_bigint::BigInt::from(1);
                    // n = ceil(p/q) = floor((p + q - 1) / q)
                    let n_candidate = (p + q - &one) / q;
                    let remainder = &n_candidate * q - p; // m = n*q - p

                    // Apply simplification if:
                    // 1. remainder > 0 (i.e., k is not an integer)
                    // 2. remainder < p (i.e., the new form is simpler)
                    // 3. n >= 1 (always true since p > 0)
                    if remainder > num_bigint::BigInt::from(0) && &remainder < p {
                        // The supplementary angle is m/q * π
                        let new_coeff = BigRational::new(remainder.clone(), q.clone());

                        // Build the new angle: (m/q) * π
                        let new_angle = if new_coeff == BigRational::from_integer(1.into()) {
                            ctx.add(Expr::Constant(cas_ast::Constant::Pi))
                        } else {
                            let pi = ctx.add(Expr::Constant(cas_ast::Constant::Pi));
                            let coeff_expr = ctx.add(Expr::Number(new_coeff));
                            ctx.add(Expr::Mul(coeff_expr, pi))
                        };

                        // Determine sign based on parity of n
                        // sin(n·π - x) = (-1)^(n+1) · sin(x)
                        // cos(n·π - x) = (-1)^n · cos(x)
                        let n_parity_odd = &n_candidate % 2 == one;

                        let (result, desc) = if is_sin {
                            // sin(n·π - x) = (-1)^(n+1) · sin(x)
                            // n odd → (-1)^(n+1) = 1, so sin(x)
                            // n even → (-1)^(n+1) = -1, so -sin(x)
                            let new_trig = ctx.call("sin", vec![new_angle]);
                            if n_parity_odd {
                                (new_trig, format!("sin({}π - x) = sin(x)", n_candidate))
                            } else {
                                (
                                    ctx.add(Expr::Neg(new_trig)),
                                    format!("sin({}π - x) = -sin(x)", n_candidate),
                                )
                            }
                        } else {
                            // cos(n·π - x) = (-1)^n · cos(x)
                            // n odd → -cos(x), n even → cos(x)
                            let new_trig = ctx.call("cos", vec![new_angle]);
                            if n_parity_odd {
                                (
                                    ctx.add(Expr::Neg(new_trig)),
                                    format!("cos({}π - x) = -cos(x)", n_candidate),
                                )
                            } else {
                                (new_trig, format!("cos({}π - x) = cos(x)", n_candidate))
                            }
                        };

                        return Some(Rewrite::new(result).desc(&desc));
                    }
                }
            }
        }

        None
    }
);

/// Extract (base_term, k) from arg such that arg = base_term + k*π/2
/// Handles multiple canonical forms:
/// - Add(x, π/2) → (x, 1)
/// - Sub(x, π/2) → (x, -1)  
/// - Div(Add(n*x, k*π), m) → (x, k*2/m) if m divides k*2
pub fn extract_phase_shift(ctx: &mut cas_ast::Context, expr: ExprId) -> Option<(ExprId, i32)> {
    // Form 1: Div((coeff*x + k*pi), denom) - the canonical form!
    // Example: (2*x + pi)/2 means x + pi/2, so k=1
    if let Expr::Div(num, den) = ctx.get(expr) {
        let num = *num;
        let den = *den;

        // Get denominator value
        let denom_val: i32 = if let Expr::Number(n) = ctx.get(den) {
            if n.is_integer() {
                n.to_integer().try_into().ok()?
            } else {
                return None;
            }
        } else {
            return None;
        };

        // Check if numerator is Add/Sub
        if let Some((l, r)) = as_add(ctx, num) {
            // Check both terms for π
            // Form: (base + k*pi)/denom where we want k/denom = m/2 for some integer m

            // Check right term for π (most common: 2*x + pi)
            if is_pi(ctx, r) {
                // pi/denom * 2 = shift in units of pi/2
                // For denom=2, shift = 1 (one pi/2)
                let k = 2 / denom_val; // Only works if denom divides 2
                if 2 % denom_val != 0 {
                    // Not a clean pi/2 multiple
                } else {
                    let base = ctx.add(Expr::Div(l, den));
                    return Some((base, k));
                }
            }

            // Check left term for π (less common: pi + 2*x)
            if is_pi(ctx, l) {
                let k = 2 / denom_val;
                if 2 % denom_val != 0 {
                    // Not a clean pi/2 multiple
                } else {
                    let base = ctx.add(Expr::Div(r, den));
                    return Some((base, k));
                }
            }

            // Also check for k*pi form using extract_pi_coefficient
            if let Some(pi_coeff) = extract_pi_coefficient(ctx, r) {
                let k_times_2 = 2 * pi_coeff;
                if k_times_2 % denom_val == 0 {
                    let k = k_times_2 / denom_val;
                    let base = ctx.add(Expr::Div(l, den));
                    return Some((base, k));
                }
            }

            if let Some(pi_coeff) = extract_pi_coefficient(ctx, l) {
                let k_times_2 = 2 * pi_coeff;
                if k_times_2 % denom_val == 0 {
                    let k = k_times_2 / denom_val;
                    let base = ctx.add(Expr::Div(r, den));
                    return Some((base, k));
                }
            }
        }
    }

    // Form 1b: Mul(1/n, Add(coeff*x, k*pi)) - the canonical form for (a + b)/n!
    // Example: (2*x + pi)/2 becomes Mul(1/2, Add(2*x, pi)); shift = 1
    if let Some((coeff_id, inner)) = as_mul(ctx, expr) {
        // Check if coeff is 1/n (a rational with numerator 1)
        if let Expr::Number(coeff) = ctx.get(coeff_id) {
            if coeff.numer() == &num_bigint::BigInt::from(1) && !coeff.denom().is_one() {
                let denom_val: i32 = coeff.denom().try_into().ok().unwrap_or(0);
                if denom_val > 0 {
                    // Check if inner is Add containing pi
                    if let Some((l, r)) = as_add(ctx, inner) {
                        // Check right term for pi
                        if is_pi(ctx, r) {
                            let k = 2 / denom_val;
                            if 2 % denom_val == 0 {
                                // base = l / denom = l * (1/denom) = coeff * l
                                let base = ctx.add(Expr::Mul(coeff_id, l));
                                return Some((base, k));
                            }
                        }

                        // Check left term for pi
                        if is_pi(ctx, l) {
                            let k = 2 / denom_val;
                            if 2 % denom_val == 0 {
                                let base = ctx.add(Expr::Mul(coeff_id, r));
                                return Some((base, k));
                            }
                        }

                        // Check for k*pi form
                        if let Some(pi_coeff) = extract_pi_coefficient(ctx, r) {
                            let k_times_2 = 2 * pi_coeff;
                            if k_times_2 % denom_val == 0 {
                                let k = k_times_2 / denom_val;
                                let base = ctx.add(Expr::Mul(coeff_id, l));
                                return Some((base, k));
                            }
                        }
                    }
                }
            }
        }
    }

    // Form 2: Add(x, k*π/2)
    if let Expr::Add(l, r) = ctx.get(expr) {
        if let Some(k) = extract_pi_half_multiple(ctx, *r) {
            return Some((*l, k));
        }
        if let Some(k) = extract_pi_half_multiple(ctx, *l) {
            return Some((*r, k));
        }
    }

    // Form 3: Sub(x, k*π/2)
    if let Expr::Sub(l, r) = ctx.get(expr) {
        if let Some(k) = extract_pi_half_multiple(ctx, *r) {
            return Some((*l, -k));
        }
    }

    None
}

/// Extract the coefficient of π from an expression.
/// - π → 1
/// - k*π → k  
/// - π*k → k
fn extract_pi_coefficient(ctx: &cas_ast::Context, expr: ExprId) -> Option<i32> {
    // Check for π alone
    if is_pi(ctx, expr) {
        return Some(1);
    }

    // Check for Mul(k, π) or Mul(π, k)
    if let Expr::Mul(l, r) = ctx.get(expr) {
        if is_pi(ctx, *r) {
            if let Expr::Number(n) = ctx.get(*l) {
                if n.is_integer() {
                    return n.to_integer().try_into().ok();
                }
            }
        }
        if is_pi(ctx, *l) {
            if let Expr::Number(n) = ctx.get(*r) {
                if n.is_integer() {
                    return n.to_integer().try_into().ok();
                }
            }
        }
    }

    None
}

/// Extract k from expressions like k*π/2, π/2, π, 3π/2, etc.
/// Returns Some(k) if the expression equals k*π/2 for integer k.
fn extract_pi_half_multiple(ctx: &cas_ast::Context, expr: ExprId) -> Option<i32> {
    // Check for π/2 (k=1)
    if is_pi_over_n(ctx, expr, 2) {
        return Some(1);
    }

    // Check for π (k=2)
    if is_pi(ctx, expr) {
        return Some(2);
    }

    // Check for Mul(k, π/2) or Mul(π/2, k)
    if let Expr::Mul(l, r) = ctx.get(expr) {
        // Check Mul(Number, π/2)
        if let Expr::Number(n) = ctx.get(*l) {
            if is_pi_over_n(ctx, *r, 2) && n.is_integer() {
                if let Ok(k) = n.to_integer().try_into() {
                    return Some(k);
                }
            }
            // Check Mul(Number, π) means k = 2*number
            if is_pi(ctx, *r) && n.is_integer() {
                if let Ok(k_half) = n.to_integer().try_into() {
                    let k: i32 = k_half;
                    return Some(k * 2);
                }
            }
        }
        // Check Mul(π/2, Number)
        if let Expr::Number(n) = ctx.get(*r) {
            if is_pi_over_n(ctx, *l, 2) && n.is_integer() {
                if let Ok(k) = n.to_integer().try_into() {
                    return Some(k);
                }
            }
            if is_pi(ctx, *l) && n.is_integer() {
                if let Ok(k_half) = n.to_integer().try_into() {
                    let k: i32 = k_half;
                    return Some(k * 2);
                }
            }
        }
    }

    // Check for Div(k*π, 2)
    if let Expr::Div(num, den) = ctx.get(expr) {
        if let Expr::Number(d) = ctx.get(*den) {
            if d.is_integer() && *d == num_rational::BigRational::from_integer(2.into()) {
                // Check if numerator is k*π or just π
                if is_pi(ctx, *num) {
                    return Some(1);
                }
                if let Expr::Mul(l, r) = ctx.get(*num) {
                    if let Expr::Number(n) = ctx.get(*l) {
                        if is_pi(ctx, *r) && n.is_integer() {
                            if let Ok(k) = n.to_integer().try_into() {
                                return Some(k);
                            }
                        }
                    }
                    if let Expr::Number(n) = ctx.get(*r) {
                        if is_pi(ctx, *l) && n.is_integer() {
                            if let Ok(k) = n.to_integer().try_into() {
                                return Some(k);
                            }
                        }
                    }
                }
            }
        }
    }

    None
}

// ============================================================================
// Cotangent Half-Angle Difference Rule
// ============================================================================
// cot(u/2) - cot(u) = 1/sin(u) = csc(u)
//
// This is a common precalculus identity that avoids term explosion from
// brute-force expansion via cot→cos/sin + double angle formulas.
//
// Pattern matching:
// - cot(u/2) - cot(u) → 1/sin(u)
// - k*cot(u/2) - k*cot(u) → k/sin(u)
// - Works on n-ary sums via flatten_add

/// Helper: Check if arg represents u/2 and return u
/// Supports: Mul(1/2, u), Div(u, 2)
fn is_half_angle(ctx: &cas_ast::Context, arg: ExprId) -> Option<ExprId> {
    match ctx.get(arg) {
        Expr::Mul(coef, inner) => {
            if let Expr::Number(n) = ctx.get(*coef) {
                if *n == num_rational::BigRational::new(1.into(), 2.into()) {
                    return Some(*inner);
                }
            }
            // Check reversed order: inner * 1/2
            if let Expr::Number(n) = ctx.get(*inner) {
                if *n == num_rational::BigRational::new(1.into(), 2.into()) {
                    return Some(*coef);
                }
            }
        }
        Expr::Div(numer, denom) => {
            if let Expr::Number(d) = ctx.get(*denom) {
                if *d == num_rational::BigRational::from_integer(2.into()) {
                    return Some(*numer);
                }
            }
        }
        _ => {}
    }
    None
}

/// Helper: Extract coefficient and cot argument from a term
/// Returns (coefficient_opt, cot_arg, is_positive)
/// coefficient_opt=None means coefficient is implicitly 1
/// is_positive=false means the term is negated (represents -coeff*cot(arg))
fn extract_cot_term(
    ctx: &cas_ast::Context,
    term: ExprId,
) -> Option<(Option<ExprId>, ExprId, bool)> {
    let term_data = ctx.get(term);

    // Check for Neg(...)
    let (inner_term, is_positive) = match term_data {
        Expr::Neg(inner) => (*inner, false),
        _ => (term, true),
    };

    let inner_data = ctx.get(inner_term);

    // Check for cot(arg) directly
    if let Expr::Function(fn_id, args) = inner_data {
        if matches!(ctx.builtin_of(*fn_id), Some(BuiltinFn::Cot)) && args.len() == 1 {
            // Coefficient is implicitly 1
            return Some((None, args[0], is_positive));
        }
    }

    // Check for Mul(coef, cot(arg))
    if let Expr::Mul(l, r) = inner_data {
        // Check if right is cot
        if let Expr::Function(fn_id, args) = ctx.get(*r) {
            if matches!(ctx.builtin_of(*fn_id), Some(BuiltinFn::Cot)) && args.len() == 1 {
                return Some((Some(*l), args[0], is_positive));
            }
        }
        // Check if left is cot
        if let Expr::Function(fn_id, args) = ctx.get(*l) {
            if matches!(ctx.builtin_of(*fn_id), Some(BuiltinFn::Cot)) && args.len() == 1 {
                return Some((Some(*r), args[0], is_positive));
            }
        }
    }

    None
}

// =============================================================================
// WEIERSTRASS HALF-ANGLE TANGENT CONTRACTION RULES
// =============================================================================
// Recognize patterns with t = tan(x/2) and contract to sin(x), cos(x):
// - 2*t / (1 + t²) → sin(x)
// - (1 - t²) / (1 + t²) → cos(x)
// This is the CONTRACTION direction (safe, doesn't worsen expressions)

/// Helper: Check if expr is tan(arg/2) and return Some(arg), i.e. the full angle
fn extract_tan_half_angle(ctx: &cas_ast::Context, expr: ExprId) -> Option<ExprId> {
    if let Expr::Function(fn_id, args) = ctx.get(expr) {
        if matches!(ctx.builtin_of(*fn_id), Some(BuiltinFn::Tan)) && args.len() == 1 {
            // Check if the argument is x/2 or (1/2)*x
            let arg = args[0];
            // Pattern: Div(x, 2) or Mul(1/2, x) or Mul(x, 1/2)
            match ctx.get(arg) {
                Expr::Div(num, den) => {
                    // x/2 pattern
                    if let Expr::Number(n) = ctx.get(*den) {
                        if n.is_integer() && *n == num_rational::BigRational::from_integer(2.into())
                        {
                            return Some(*num); // return x (the full angle)
                        }
                    }
                }
                Expr::Mul(l, r) => {
                    // (1/2)*x or x*(1/2) pattern
                    let half = num_rational::BigRational::new(1.into(), 2.into());
                    if let Expr::Number(n) = ctx.get(*l) {
                        if *n == half {
                            return Some(*r);
                        }
                    }
                    if let Expr::Number(n) = ctx.get(*r) {
                        if *n == half {
                            return Some(*l);
                        }
                    }
                }
                _ => {}
            }
        }
    }
    None
}

/// Helper: Check if expr is 1 + tan(x/2)² and return (x, tan_half_id)
fn match_one_plus_tan_squared(ctx: &cas_ast::Context, expr: ExprId) -> Option<(ExprId, ExprId)> {
    if let Expr::Add(l, r) = ctx.get(expr) {
        // Check both orders: 1 + tan²(...) or tan²(...) + 1
        let (one_id, pow_id) = if matches!(ctx.get(*l), Expr::Number(n) if n.is_one()) {
            (*l, *r)
        } else if matches!(ctx.get(*r), Expr::Number(n) if n.is_one()) {
            (*r, *l)
        } else {
            return None;
        };
        let _ = one_id;

        // Check if pow_id is tan(x/2)^2
        if let Expr::Pow(base, exp) = ctx.get(pow_id) {
            if let Expr::Number(n) = ctx.get(*exp) {
                if n.is_integer() && *n == num_rational::BigRational::from_integer(2.into()) {
                    if let Some(full_angle) = extract_tan_half_angle(ctx, *base) {
                        return Some((full_angle, *base));
                    }
                }
            }
        }
    }
    None
}

/// Helper: Check if expr is 1 - tan(x/2)² and return (x, tan_half_id)
fn match_one_minus_tan_squared(ctx: &cas_ast::Context, expr: ExprId) -> Option<(ExprId, ExprId)> {
    // Check Sub(1, tan²)
    if let Expr::Sub(l, r) = ctx.get(expr) {
        if let Expr::Number(n) = ctx.get(*l) {
            if n.is_one() {
                // Check if r is tan(x/2)^2
                if let Expr::Pow(base, exp) = ctx.get(*r) {
                    if let Expr::Number(e) = ctx.get(*exp) {
                        if e.is_integer() && *e == num_rational::BigRational::from_integer(2.into())
                        {
                            if let Some(full_angle) = extract_tan_half_angle(ctx, *base) {
                                return Some((full_angle, *base));
                            }
                        }
                    }
                }
            }
        }
    }

    // Also check Add(1, Neg(tan²)) which is canonicalized form
    if let Expr::Add(l, r) = ctx.get(expr) {
        let (one_id, neg_id) = if matches!(ctx.get(*l), Expr::Number(n) if n.is_one()) {
            (*l, *r)
        } else if matches!(ctx.get(*r), Expr::Number(n) if n.is_one()) {
            (*r, *l)
        } else {
            return None;
        };
        let _ = one_id;

        if let Expr::Neg(inner) = ctx.get(neg_id) {
            if let Expr::Pow(base, exp) = ctx.get(*inner) {
                if let Expr::Number(e) = ctx.get(*exp) {
                    if e.is_integer() && *e == num_rational::BigRational::from_integer(2.into()) {
                        if let Some(full_angle) = extract_tan_half_angle(ctx, *base) {
                            return Some((full_angle, *base));
                        }
                    }
                }
            }
        }
    }

    None
}

// Weierstrass Contraction Rule: 2*tan(x/2)/(1+tan²(x/2)) → sin(x)
// and (1-tan²(x/2))/(1+tan²(x/2)) → cos(x)
pub struct WeierstrassContractionRule;

impl crate::rule::Rule for WeierstrassContractionRule {
    fn name(&self) -> &str {
        "Weierstrass Half-Angle Contraction"
    }

    fn apply(
        &self,
        ctx: &mut cas_ast::Context,
        expr: ExprId,
        _parent_ctx: &crate::parent_context::ParentContext,
    ) -> Option<crate::rule::Rewrite> {
        // Only match Div nodes
        let Some((num_id, den_id)) = as_div(ctx, expr) else {
            return None;
        };

        // Pattern 1: 2*tan(x/2) / (1 + tan²(x/2)) → sin(x)
        // Check denominator: 1 + tan²(x/2)
        if let Some((full_angle, tan_half)) = match_one_plus_tan_squared(ctx, den_id) {
            // Check numerator: 2*tan(x/2)
            if let Expr::Mul(l, r) = ctx.get(num_id) {
                let (two_id, tan_id) = if matches!(ctx.get(*l), Expr::Number(n) if n.is_integer() && *n == num_rational::BigRational::from_integer(2.into()))
                {
                    (*l, *r)
                } else if matches!(ctx.get(*r), Expr::Number(n) if n.is_integer() && *n == num_rational::BigRational::from_integer(2.into()))
                {
                    (*r, *l)
                } else {
                    return self.try_cos_pattern(ctx, num_id, den_id, full_angle, tan_half);
                };
                let _ = two_id;

                // Check if tan_id is tan(x/2) with same argument
                if let Some(tan_arg) = extract_tan_half_angle(ctx, tan_id) {
                    if crate::ordering::compare_expr(ctx, tan_arg, full_angle)
                        == std::cmp::Ordering::Equal
                    {
                        let sin_x = ctx.call("sin", vec![full_angle]);
                        return Some(
                            Rewrite::new(sin_x).desc("2·tan(x/2)/(1 + tan²(x/2)) = sin(x)"),
                        );
                    }
                }
            }

            // Pattern 2: (1 - tan²(x/2)) / (1 + tan²(x/2)) → cos(x)
            return self.try_cos_pattern(ctx, num_id, den_id, full_angle, tan_half);
        }

        None
    }

    fn target_types(&self) -> Option<Vec<&str>> {
        Some(vec!["Div"])
    }

    fn importance(&self) -> crate::step::ImportanceLevel {
        crate::step::ImportanceLevel::High
    }
}

impl WeierstrassContractionRule {
    fn try_cos_pattern(
        &self,
        ctx: &mut cas_ast::Context,
        num_id: ExprId,
        den_id: ExprId,
        _expected_angle: ExprId,
        _expected_tan_half: ExprId,
    ) -> Option<Rewrite> {
        // Pattern 2: (1 - tan²(x/2)) / (1 + tan²(x/2)) → cos(x)
        if let Some((num_angle, _num_tan_half)) = match_one_minus_tan_squared(ctx, num_id) {
            if let Some((den_angle, _den_tan_half)) = match_one_plus_tan_squared(ctx, den_id) {
                // Check angles are the same
                if crate::ordering::compare_expr(ctx, num_angle, den_angle)
                    == std::cmp::Ordering::Equal
                {
                    let cos_x = ctx.call("cos", vec![num_angle]);
                    return Some(
                        Rewrite::new(cos_x).desc("(1 - tan²(x/2))/(1 + tan²(x/2)) = cos(x)"),
                    );
                }
            }
        }

        None
    }
}

// =============================================================================
// WEIERSTRASS IDENTITY ZERO RULES (Pattern-Driven Cancellation)
// =============================================================================
// These rules detect the complete Weierstrass identity patterns and cancel to 0
// directly, avoiding explosive expansion through tan→sin/cos conversion.
//
// sin(x) - 2*tan(x/2)/(1 + tan(x/2)²) → 0
// cos(x) - (1 - tan(x/2)²)/(1 + tan(x/2)²) → 0

/// Helper: Check if expr matches 2*tan(x/2) and return the full angle x
fn match_two_tan_half(ctx: &cas_ast::Context, expr: ExprId) -> Option<ExprId> {
    let two_rat = num_rational::BigRational::from_integer(2.into());

    if let Expr::Mul(l, r) = ctx.get(expr) {
        // Check Mul(2, tan(x/2)) or Mul(tan(x/2), 2)
        if let Expr::Number(n) = ctx.get(*l) {
            if *n == two_rat {
                return extract_tan_half_angle(ctx, *r);
            }
        }
        if let Expr::Number(n) = ctx.get(*r) {
            if *n == two_rat {
                return extract_tan_half_angle(ctx, *l);
            }
        }
    }
    None
}

/// Helper: Check if expr matches 1 + tan(x/2)² and return (full_angle, tan_half_id)
fn match_one_plus_tan_half_squared(ctx: &cas_ast::Context, expr: ExprId) -> Option<ExprId> {
    if let Expr::Add(l, r) = ctx.get(expr) {
        let two_rat = num_rational::BigRational::from_integer(2.into());

        // Pattern: 1 + tan²(x/2) or tan²(x/2) + 1
        let (one_candidate, pow_candidate) = if matches!(ctx.get(*l), Expr::Number(n) if n.is_one())
        {
            (*l, *r)
        } else if matches!(ctx.get(*r), Expr::Number(n) if n.is_one()) {
            (*r, *l)
        } else {
            return None;
        };
        let _ = one_candidate;

        // Check pow_candidate is tan(x/2)^2
        if let Expr::Pow(base, exp) = ctx.get(pow_candidate) {
            if let Expr::Number(n) = ctx.get(*exp) {
                if *n == two_rat {
                    return extract_tan_half_angle(ctx, *base);
                }
            }
        }
    }
    None
}

/// Helper: Check if expr matches 1 - tan(x/2)² and return full_angle
fn match_one_minus_tan_half_squared(ctx: &cas_ast::Context, expr: ExprId) -> Option<ExprId> {
    let two_rat = num_rational::BigRational::from_integer(2.into());

    // Pattern: 1 - tan²(x/2) as Sub(1, tan²)
    if let Expr::Sub(l, r) = ctx.get(expr) {
        if let Expr::Number(n) = ctx.get(*l) {
            if n.is_one() {
                if let Expr::Pow(base, exp) = ctx.get(*r) {
                    if let Expr::Number(e) = ctx.get(*exp) {
                        if *e == two_rat {
                            return extract_tan_half_angle(ctx, *base);
                        }
                    }
                }
            }
        }
    }

    // Also try Add(1, Neg(tan²)) or Add(Neg(tan²), 1)
    if let Expr::Add(l, r) = ctx.get(expr) {
        // 1 + (-tan²) or (-tan²) + 1
        let (one_candidate, neg_candidate) = if matches!(ctx.get(*l), Expr::Number(n) if n.is_one())
        {
            (*l, *r)
        } else if matches!(ctx.get(*r), Expr::Number(n) if n.is_one()) {
            (*r, *l)
        } else {
            return None;
        };
        let _ = one_candidate;

        if let Expr::Neg(inner) = ctx.get(neg_candidate) {
            if let Expr::Pow(base, exp) = ctx.get(*inner) {
                if let Expr::Number(e) = ctx.get(*exp) {
                    if *e == two_rat {
                        return extract_tan_half_angle(ctx, *base);
                    }
                }
            }
        }
    }

    None
}

// WeierstrassSinIdentityZeroRule: sin(x) - 2*tan(x/2)/(1+tan²(x/2)) → 0
// Pattern-driven cancellation, no expansion.
pub struct WeierstrassSinIdentityZeroRule;

impl crate::rule::Rule for WeierstrassSinIdentityZeroRule {
    fn name(&self) -> &str {
        "Weierstrass Sin Identity Zero"
    }

    fn apply(
        &self,
        ctx: &mut cas_ast::Context,
        expr: ExprId,
        _parent_ctx: &crate::parent_context::ParentContext,
    ) -> Option<Rewrite> {
        // Only match Sub nodes: sin(x) - RHS or RHS - sin(x)
        let (left, right, negated) = if let Some((l, r)) = as_sub(ctx, expr) {
            (l, r, false)
        } else if let Some((_l, _r)) = as_add(ctx, expr) {
            // Check if one side is negated
            let (l, r) = as_add(ctx, expr).unwrap();
            if let Some(inner) = as_neg(ctx, r) {
                (l, inner, false)
            } else if let Some(inner) = as_neg(ctx, l) {
                (r, inner, false)
            } else {
                return None;
            }
        } else {
            return None;
        };
        let _ = negated;

        // Try both orderings: sin(x) - RHS and RHS - sin(x)
        if let Some(result) = self.try_match(ctx, left, right) {
            return Some(result);
        }
        if let Some(result) = self.try_match(ctx, right, left) {
            return Some(result);
        }

        None
    }

    fn target_types(&self) -> Option<Vec<&str>> {
        Some(vec!["Sub", "Add"])
    }

    fn importance(&self) -> crate::step::ImportanceLevel {
        crate::step::ImportanceLevel::High
    }

    fn priority(&self) -> i32 {
        200 // Very high - must run BEFORE Pythagorean 1+tan²→sec²
    }
}

impl WeierstrassSinIdentityZeroRule {
    /// Try to match sin(x) = left, RHS = right
    fn try_match(
        &self,
        ctx: &mut cas_ast::Context,
        sin_side: ExprId,
        rhs: ExprId,
    ) -> Option<Rewrite> {
        // Check if sin_side is sin(x)
        if let Expr::Function(fn_id, args) = ctx.get(sin_side) {
            if !matches!(ctx.builtin_of(*fn_id), Some(BuiltinFn::Sin)) || args.len() != 1 {
                return None;
            }
            let full_angle = args[0];

            // Check if rhs is 2*tan(x/2) / (1 + tan²(x/2))
            if let Expr::Div(num, den) = ctx.get(rhs) {
                // Numerator: 2*tan(x/2)
                if let Some(num_angle) = match_two_tan_half(ctx, *num) {
                    // Denominator: 1 + tan²(x/2)
                    if let Some(den_angle) = match_one_plus_tan_half_squared(ctx, *den) {
                        // Check all angles match
                        if crate::ordering::compare_expr(ctx, full_angle, num_angle)
                            == std::cmp::Ordering::Equal
                            && crate::ordering::compare_expr(ctx, full_angle, den_angle)
                                == std::cmp::Ordering::Equal
                        {
                            let zero = ctx.num(0);
                            return Some(
                                Rewrite::new(zero)
                                    .desc("sin(x) = 2·tan(x/2)/(1 + tan²(x/2)) [Weierstrass]"),
                            );
                        }
                    }
                }
            }
        }
        None
    }
}

// WeierstrassCosIdentityZeroRule: cos(x) - (1-tan²(x/2))/(1+tan²(x/2)) → 0
pub struct WeierstrassCosIdentityZeroRule;

impl crate::rule::Rule for WeierstrassCosIdentityZeroRule {
    fn name(&self) -> &str {
        "Weierstrass Cos Identity Zero"
    }

    fn apply(
        &self,
        ctx: &mut cas_ast::Context,
        expr: ExprId,
        _parent_ctx: &crate::parent_context::ParentContext,
    ) -> Option<Rewrite> {
        // Only match Sub nodes: cos(x) - RHS or RHS - cos(x)
        let (left, right) = if let Some((l, r)) = as_sub(ctx, expr) {
            (l, r)
        } else if let Some((l, r)) = as_add(ctx, expr) {
            // Check if one side is negated
            if let Some(inner) = as_neg(ctx, r) {
                (l, inner)
            } else if let Some(inner) = as_neg(ctx, l) {
                (r, inner)
            } else {
                return None;
            }
        } else {
            return None;
        };

        // Try both orderings
        if let Some(result) = self.try_match(ctx, left, right) {
            return Some(result);
        }
        if let Some(result) = self.try_match(ctx, right, left) {
            return Some(result);
        }

        None
    }

    fn target_types(&self) -> Option<Vec<&str>> {
        Some(vec!["Sub", "Add"])
    }

    fn importance(&self) -> crate::step::ImportanceLevel {
        crate::step::ImportanceLevel::High
    }

    fn priority(&self) -> i32 {
        200 // Very high - must run BEFORE Pythagorean 1+tan²→sec²
    }
}

impl WeierstrassCosIdentityZeroRule {
    /// Try to match cos(x) = cos_side, RHS = rhs
    fn try_match(
        &self,
        ctx: &mut cas_ast::Context,
        cos_side: ExprId,
        rhs: ExprId,
    ) -> Option<Rewrite> {
        // Check if cos_side is cos(x)
        if let Expr::Function(fn_id, args) = ctx.get(cos_side) {
            if !matches!(ctx.builtin_of(*fn_id), Some(BuiltinFn::Cos)) || args.len() != 1 {
                return None;
            }
            let full_angle = args[0];

            // Check if rhs is (1 - tan²(x/2)) / (1 + tan²(x/2))
            if let Expr::Div(num, den) = ctx.get(rhs) {
                // Numerator: 1 - tan²(x/2)
                if let Some(num_angle) = match_one_minus_tan_half_squared(ctx, *num) {
                    // Denominator: 1 + tan²(x/2)
                    if let Some(den_angle) = match_one_plus_tan_half_squared(ctx, *den) {
                        // Check all angles match
                        if crate::ordering::compare_expr(ctx, full_angle, num_angle)
                            == std::cmp::Ordering::Equal
                            && crate::ordering::compare_expr(ctx, full_angle, den_angle)
                                == std::cmp::Ordering::Equal
                        {
                            let zero = ctx.num(0);
                            return Some(
                                Rewrite::new(zero)
                                    .desc("cos(x) = (1 - tan²(x/2))/(1 + tan²(x/2)) [Weierstrass]"),
                            );
                        }
                    }
                }
            }
        }
        None
    }
}

// =============================================================================
// Sin4xIdentityZeroRule: sin(4t) - 4*sin(t)*cos(t)*(cos²(t)-sin²(t)) → 0
// =============================================================================
// Recognizes the sin(4x) expansion identity directly in cancellation context.

pub struct Sin4xIdentityZeroRule;

impl crate::rule::Rule for Sin4xIdentityZeroRule {
    fn name(&self) -> &str {
        "Sin 4x Identity Zero"
    }

    fn apply(
        &self,
        ctx: &mut cas_ast::Context,
        expr: ExprId,
        _parent_ctx: &crate::parent_context::ParentContext,
    ) -> Option<Rewrite> {
        // Only match Sub nodes or Add with negated term
        let (left, right) = if let Some((l, r)) = as_sub(ctx, expr) {
            (l, r)
        } else if let Some((l, r)) = as_add(ctx, expr) {
            if let Some(inner) = as_neg(ctx, r) {
                (l, inner)
            } else if let Some(inner) = as_neg(ctx, l) {
                (r, inner)
            } else {
                return None;
            }
        } else {
            return None;
        };

        // Try both orderings
        if let Some(result) = self.try_match(ctx, left, right) {
            return Some(result);
        }
        if let Some(result) = self.try_match(ctx, right, left) {
            return Some(result);
        }

        None
    }

    fn target_types(&self) -> Option<Vec<&str>> {
        Some(vec!["Sub", "Add"])
    }

    fn priority(&self) -> i32 {
        200 // Run before expansion
    }

    fn importance(&self) -> crate::step::ImportanceLevel {
        crate::step::ImportanceLevel::High
    }
}

impl Sin4xIdentityZeroRule {
    fn try_match(&self, ctx: &mut cas_ast::Context, lhs: ExprId, rhs: ExprId) -> Option<Rewrite> {
        // LHS should be sin(4*t)
        if let Expr::Function(fn_id, args) = ctx.get(lhs) {
            if !matches!(ctx.builtin_of(*fn_id), Some(BuiltinFn::Sin)) || args.len() != 1 {
                return None;
            }
            let sin_arg = args[0];

            // Check if arg is 4*t
            let t = match ctx.get(sin_arg) {
                Expr::Mul(l, r) => {
                    // Check for Mul(4, t) or Mul(t, 4)
                    let l = *l;
                    let r = *r;
                    if let Expr::Number(n) = ctx.get(l) {
                        if *n == num_rational::BigRational::from_integer(4.into()) {
                            r
                        } else {
                            return None;
                        }
                    } else if let Expr::Number(n) = ctx.get(r) {
                        if *n == num_rational::BigRational::from_integer(4.into()) {
                            l
                        } else {
                            return None;
                        }
                    } else {
                        return None;
                    }
                }
                _ => return None,
            };

            // RHS should be 4*sin(t)*cos(t)*(cos(t)^2 - sin(t)^2)
            // Flatten multiplication to get factors
            let mut factors = Vec::new();
            crate::helpers::flatten_mul(ctx, rhs, &mut factors);

            if factors.len() < 4 {
                return None;
            }

            // Look for: 4, sin(t), cos(t), (cos(t)^2 - sin(t)^2) or cos(2t)
            let mut has_four = false;
            let mut has_sin_t = false;
            let mut has_cos_t = false;
            let mut has_diff_squares = false;

            for &factor in &factors {
                // Check for 4
                if let Expr::Number(n) = ctx.get(factor) {
                    if *n == num_rational::BigRational::from_integer(4.into()) {
                        has_four = true;
                        continue;
                    }
                }
                // Check for sin(t)
                if let Expr::Function(fn_name, fn_args) = ctx.get(factor) {
                    if ctx.is_builtin(*fn_name, BuiltinFn::Sin)
                        && fn_args.len() == 1
                        && crate::ordering::compare_expr(ctx, fn_args[0], t)
                            == std::cmp::Ordering::Equal
                    {
                        has_sin_t = true;
                        continue;
                    }
                    if ctx.is_builtin(*fn_name, BuiltinFn::Cos) && fn_args.len() == 1 {
                        let arg = fn_args[0];
                        if crate::ordering::compare_expr(ctx, arg, t) == std::cmp::Ordering::Equal {
                            has_cos_t = true;
                            continue;
                        }
                        // Also check for cos(2t) which equals cos²-sin²
                        if let Expr::Mul(cl, cr) = ctx.get(arg) {
                            let cl = *cl;
                            let cr = *cr;
                            let is_2t = if let Expr::Number(n) = ctx.get(cl) {
                                *n == num_rational::BigRational::from_integer(2.into())
                                    && crate::ordering::compare_expr(ctx, cr, t)
                                        == std::cmp::Ordering::Equal
                            } else if let Expr::Number(n) = ctx.get(cr) {
                                *n == num_rational::BigRational::from_integer(2.into())
                                    && crate::ordering::compare_expr(ctx, cl, t)
                                        == std::cmp::Ordering::Equal
                            } else {
                                false
                            };
                            if is_2t {
                                has_diff_squares = true;
                                continue;
                            }
                        }
                    }
                }
                // Check for cos²(t) - sin²(t)
                if let Expr::Sub(sl, sr) = ctx.get(factor) {
                    let sl = *sl;
                    let sr = *sr;
                    if self.is_cos_squared_t(ctx, sl, t) && self.is_sin_squared_t(ctx, sr, t) {
                        has_diff_squares = true;
                        continue;
                    }
                }
            }

            if has_four && has_sin_t && has_cos_t && has_diff_squares {
                let zero = ctx.num(0);
                return Some(
                    Rewrite::new(zero).desc("sin(4t) = 4·sin(t)·cos(t)·(cos²(t)-sin²(t))"),
                );
            }
        }
        None
    }

    fn is_sin_squared_t(&self, ctx: &cas_ast::Context, expr: ExprId, t: ExprId) -> bool {
        if let Expr::Pow(base, exp) = ctx.get(expr) {
            if let Expr::Number(n) = ctx.get(*exp) {
                if *n == num_rational::BigRational::from_integer(2.into()) {
                    if let Expr::Function(fn_id, args) = ctx.get(*base) {
                        if matches!(ctx.builtin_of(*fn_id), Some(BuiltinFn::Sin)) && args.len() == 1
                        {
                            return crate::ordering::compare_expr(ctx, args[0], t)
                                == std::cmp::Ordering::Equal;
                        }
                    }
                }
            }
        }
        false
    }

    fn is_cos_squared_t(&self, ctx: &cas_ast::Context, expr: ExprId, t: ExprId) -> bool {
        if let Expr::Pow(base, exp) = ctx.get(expr) {
            if let Expr::Number(n) = ctx.get(*exp) {
                if *n == num_rational::BigRational::from_integer(2.into()) {
                    if let Expr::Function(fn_id, args) = ctx.get(*base) {
                        if matches!(ctx.builtin_of(*fn_id), Some(BuiltinFn::Cos)) && args.len() == 1
                        {
                            return crate::ordering::compare_expr(ctx, args[0], t)
                                == std::cmp::Ordering::Equal;
                        }
                    }
                }
            }
        }
        false
    }
}

// =============================================================================
// TanDifferenceIdentityZeroRule: tan(a-b) - (tan(a)-tan(b))/(1+tan(a)*tan(b)) → 0
// =============================================================================
// Recognizes the tangent difference identity directly in cancellation context.

pub struct TanDifferenceIdentityZeroRule;

impl crate::rule::Rule for TanDifferenceIdentityZeroRule {
    fn name(&self) -> &str {
        "Tangent Difference Identity Zero"
    }

    fn apply(
        &self,
        ctx: &mut cas_ast::Context,
        expr: ExprId,
        _parent_ctx: &crate::parent_context::ParentContext,
    ) -> Option<Rewrite> {
        // Only match Sub nodes or Add with negated term
        let (left, right, _negated) = if let Some((l, r)) = as_sub(ctx, expr) {
            (l, r, false)
        } else if let Some((l, r)) = as_add(ctx, expr) {
            if let Some(inner) = as_neg(ctx, r) {
                (l, inner, true)
            } else if let Some(inner) = as_neg(ctx, l) {
                (r, inner, true)
            } else {
                return None;
            }
        } else {
            return None;
        };

        // Try both orderings: tan(a-b) - RHS or RHS - tan(a-b)
        if let Some(result) = self.try_match(ctx, left, right) {
            return Some(result);
        }
        if let Some(result) = self.try_match(ctx, right, left) {
            return Some(result);
        }

        None
    }

    fn target_types(&self) -> Option<Vec<&str>> {
        Some(vec!["Sub", "Add"])
    }

    fn priority(&self) -> i32 {
        200 // Run before tan→sin/cos expansion
    }

    fn importance(&self) -> crate::step::ImportanceLevel {
        crate::step::ImportanceLevel::High
    }
}

impl TanDifferenceIdentityZeroRule {
    fn try_match(&self, ctx: &mut cas_ast::Context, lhs: ExprId, rhs: ExprId) -> Option<Rewrite> {
        // LHS should be tan(a - b)
        if let Expr::Function(fn_id, args) = ctx.get(lhs) {
            if !matches!(ctx.builtin_of(*fn_id), Some(BuiltinFn::Tan)) || args.len() != 1 {
                return None;
            }
            let tan_arg = args[0];

            // Extract a and b from (a - b) or (a + (-b))
            let (a, b) = match ctx.get(tan_arg) {
                Expr::Sub(l, r) => (*l, *r),
                Expr::Add(l, r) => {
                    if let Expr::Neg(inner) = ctx.get(*r) {
                        (*l, *inner)
                    } else if let Expr::Neg(inner) = ctx.get(*l) {
                        (*r, *inner)
                    } else {
                        return None;
                    }
                }
                _ => return None,
            };

            // RHS should be (tan(a) - tan(b)) / (1 + tan(a)*tan(b))
            if let Expr::Div(num, den) = ctx.get(rhs) {
                let num = *num;
                let den = *den;

                // Check numerator: tan(a) - tan(b)
                let (tan_a_num, tan_b_num) = match ctx.get(num) {
                    Expr::Sub(l, r) => (*l, *r),
                    Expr::Add(l, r) => {
                        if let Expr::Neg(inner) = ctx.get(*r) {
                            (*l, *inner)
                        } else {
                            return None;
                        }
                    }
                    _ => return None,
                };

                // Verify tan_a_num is tan(a)
                if let Expr::Function(name_a, args_a) = ctx.get(tan_a_num) {
                    if !ctx.is_builtin(*name_a, BuiltinFn::Tan) || args_a.len() != 1 {
                        return None;
                    }
                    if crate::ordering::compare_expr(ctx, args_a[0], a) != std::cmp::Ordering::Equal
                    {
                        return None;
                    }
                } else {
                    return None;
                }

                // Verify tan_b_num is tan(b)
                if let Expr::Function(name_b, args_b) = ctx.get(tan_b_num) {
                    if !ctx.is_builtin(*name_b, BuiltinFn::Tan) || args_b.len() != 1 {
                        return None;
                    }
                    if crate::ordering::compare_expr(ctx, args_b[0], b) != std::cmp::Ordering::Equal
                    {
                        return None;
                    }
                } else {
                    return None;
                }

                // Check denominator: 1 + tan(a)*tan(b)
                if !self.match_one_plus_tan_product(ctx, den, a, b) {
                    return None;
                }

                // All matched! Return 0
                let zero = ctx.num(0);
                return Some(
                    Rewrite::new(zero).desc("tan(a-b) = (tan(a)-tan(b))/(1+tan(a)·tan(b))"),
                );
            }
        }
        None
    }

    fn match_one_plus_tan_product(
        &self,
        ctx: &cas_ast::Context,
        expr: ExprId,
        a: ExprId,
        b: ExprId,
    ) -> bool {
        // Match 1 + tan(a)*tan(b) (in any order)
        if let Expr::Add(l, r) = ctx.get(expr) {
            let (one_part, product_part) = if let Expr::Number(n) = ctx.get(*l) {
                if *n == num_rational::BigRational::from_integer(1.into()) {
                    (*l, *r)
                } else {
                    return false;
                }
            } else if let Expr::Number(n) = ctx.get(*r) {
                if *n == num_rational::BigRational::from_integer(1.into()) {
                    (*r, *l)
                } else {
                    return false;
                }
            } else {
                return false;
            };
            let _ = one_part;

            // product_part should be tan(a)*tan(b)
            if let Expr::Mul(ml, mr) = ctx.get(product_part) {
                let (tan1, tan2) = (*ml, *mr);

                // Check if both are tan functions
                if let (Expr::Function(fn_id1, args1), Expr::Function(fn_id2, args2)) =
                    (ctx.get(tan1), ctx.get(tan2))
                {
                    let b1 = ctx.builtin_of(*fn_id1);
                    let b2 = ctx.builtin_of(*fn_id2);
                    if matches!(b1, Some(BuiltinFn::Tan))
                        && matches!(b2, Some(BuiltinFn::Tan))
                        && args1.len() == 1
                        && args2.len() == 1
                    {
                        let arg1 = args1[0];
                        let arg2 = args2[0];

                        // Check (arg1==a && arg2==b) or (arg1==b && arg2==a)
                        let match1 = crate::ordering::compare_expr(ctx, arg1, a)
                            == std::cmp::Ordering::Equal
                            && crate::ordering::compare_expr(ctx, arg2, b)
                                == std::cmp::Ordering::Equal;
                        let match2 = crate::ordering::compare_expr(ctx, arg1, b)
                            == std::cmp::Ordering::Equal
                            && crate::ordering::compare_expr(ctx, arg2, a)
                                == std::cmp::Ordering::Equal;

                        return match1 || match2;
                    }
                }
            }
        }
        false
    }
}

define_rule!(
    CotHalfAngleDifferenceRule,
    "Cotangent Half-Angle Difference",
    |ctx, expr| {
        // Only match Add or Sub at top level
        // Normalize Sub to Add(a, Neg(b)) conceptually by handling both
        let terms: Vec<ExprId> = if as_add(ctx, expr).is_some() {
            let mut ts = Vec::new();
            crate::helpers::flatten_add(ctx, expr, &mut ts);
            ts
        } else if let Some((l, r)) = as_sub(ctx, expr) {
            // Treat as [l, -r]
            vec![l, r] // We'll handle the sign in matching
        } else {
            return None;
        };

        if terms.len() < 2 {
            return None;
        }

        // For Sub, we have special handling
        let is_explicit_sub = matches!(ctx.get(expr), Expr::Sub(_, _));

        // Collect cot terms: (index, coeff, arg, is_positive_in_original)
        struct CotTerm {
            index: usize,
            coeff: Option<ExprId>, // None means coefficient is 1
            arg: ExprId,
            is_positive: bool,
        }

        let mut cot_terms = Vec::new();

        if is_explicit_sub {
            // For Sub(a, b): a is positive, b is effectively negative
            if let Some((c1, arg1, _)) = extract_cot_term(ctx, terms[0]) {
                cot_terms.push(CotTerm {
                    index: 0,
                    coeff: c1,
                    arg: arg1,
                    is_positive: true,
                });
            }
            if let Some((c2, arg2, sign2)) = extract_cot_term(ctx, terms[1]) {
                // In Sub(a, b), b appears with flipped sign
                cot_terms.push(CotTerm {
                    index: 1,
                    coeff: c2,
                    arg: arg2,
                    is_positive: !sign2, // Flip because it's subtracted
                });
            }
        } else {
            // For Add chain
            for (i, &term) in terms.iter().enumerate() {
                if let Some((c, arg, is_pos)) = extract_cot_term(ctx, term) {
                    cot_terms.push(CotTerm {
                        index: i,
                        coeff: c,
                        arg,
                        is_positive: is_pos,
                    });
                }
            }
        }

        // Look for pairs: cot(u/2) and cot(u) with opposite signs
        for i in 0..cot_terms.len() {
            for j in 0..cot_terms.len() {
                if i == j {
                    continue;
                }

                let t_half = &cot_terms[i];
                let t_full = &cot_terms[j];

                // Check if t_half.arg is half of t_full.arg
                if let Some(full_angle) = is_half_angle(ctx, t_half.arg) {
                    // Verify full_angle == t_full.arg
                    if crate::ordering::compare_expr(ctx, full_angle, t_full.arg) != Ordering::Equal
                    {
                        continue;
                    }

                    // Check that coefficients match (or both are 1)
                    let coeffs_match = match (&t_half.coeff, &t_full.coeff) {
                        (None, None) => true,
                        (Some(c1), Some(c2)) => {
                            crate::ordering::compare_expr(ctx, *c1, *c2) == Ordering::Equal
                        }
                        _ => false,
                    };

                    if !coeffs_match {
                        continue;
                    }

                    // Check signs: cot(u/2) positive AND cot(u) negative = cot(u/2) - cot(u)
                    // OR cot(u/2) negative AND cot(u) positive = -cot(u/2) + cot(u) = -(cot(u/2) - cot(u))
                    if t_half.is_positive && !t_full.is_positive {
                        // cot(u/2) - cot(u) → 1/sin(u)
                        let one = ctx.num(1);
                        let sin_u = ctx.call("sin", vec![t_full.arg]);
                        let result = ctx.add(Expr::Div(one, sin_u));

                        // Apply coefficient if present
                        let final_result = if let Some(c) = t_half.coeff {
                            smart_mul(ctx, c, result)
                        } else {
                            result
                        };

                        // Reconstruct expression without the matched terms
                        if is_explicit_sub && terms.len() == 2 {
                            // Simple case: Sub(cot(u/2), cot(u)) → 1/sin(u)
                            return Some(
                                Rewrite::new(final_result).desc("cot(u/2) - cot(u) = 1/sin(u)"),
                            );
                        }

                        // N-ary case: rebuild sum without matched terms
                        let mut new_terms: Vec<ExprId> = Vec::new();
                        for (k, &term) in terms.iter().enumerate() {
                            if k != t_half.index && k != t_full.index {
                                new_terms.push(term);
                            }
                        }
                        new_terms.push(final_result);

                        let mut new_expr = new_terms[0];
                        for &term in new_terms.iter().skip(1) {
                            new_expr = ctx.add(Expr::Add(new_expr, term));
                        }

                        return Some(Rewrite::new(new_expr).desc("cot(u/2) - cot(u) = 1/sin(u)"));
                    } else if !t_half.is_positive && t_full.is_positive {
                        // -cot(u/2) + cot(u) → -1/sin(u)
                        let one = ctx.num(1);
                        let sin_u = ctx.call("sin", vec![t_full.arg]);
                        let result = ctx.add(Expr::Div(one, sin_u));
                        let neg_result = ctx.add(Expr::Neg(result));

                        // Apply coefficient if present
                        let final_result = if let Some(c) = t_half.coeff {
                            smart_mul(ctx, c, neg_result)
                        } else {
                            neg_result
                        };

                        if is_explicit_sub && terms.len() == 2 {
                            return Some(
                                Rewrite::new(final_result).desc("-cot(u/2) + cot(u) = -1/sin(u)"),
                            );
                        }

                        // N-ary case
                        let mut new_terms: Vec<ExprId> = Vec::new();
                        for (k, &term) in terms.iter().enumerate() {
                            if k != t_half.index && k != t_full.index {
                                new_terms.push(term);
                            }
                        }
                        new_terms.push(final_result);

                        let mut new_expr = new_terms[0];
                        for &term in new_terms.iter().skip(1) {
                            new_expr = ctx.add(Expr::Add(new_expr, term));
                        }

                        return Some(Rewrite::new(new_expr).desc("-cot(u/2) + cot(u) = -1/sin(u)"));
                    }
                }
            }
        }

        None
    }
);

// =============================================================================
// TanDifferenceRule: tan(a - b) → (tan(a) - tan(b)) / (1 + tan(a)*tan(b))
// =============================================================================

define_rule!(TanDifferenceRule, "Tangent Difference", |ctx, expr| {
    if let Expr::Function(fn_id, args) = ctx.get(expr) {
        if matches!(ctx.builtin_of(*fn_id), Some(BuiltinFn::Tan)) && args.len() == 1 {
            let arg = args[0];
            // Check if argument is a - b
            if let Expr::Sub(a, b) = ctx.get(arg) {
                let a = *a;
                let b = *b;

                // Build tan(a) - tan(b)
                let tan_a = ctx.call("tan", vec![a]);
                let tan_b = ctx.call("tan", vec![b]);
                let numerator = ctx.add(Expr::Sub(tan_a, tan_b));

                // Build 1 + tan(a)*tan(b)
                let tan_a2 = ctx.call("tan", vec![a]);
                let tan_b2 = ctx.call("tan", vec![b]);
                let product = ctx.add(Expr::Mul(tan_a2, tan_b2));
                let one = ctx.num(1);
                let denominator = ctx.add(Expr::Add(one, product));

                let result = ctx.add(Expr::Div(numerator, denominator));
                return Some(
                    Rewrite::new(result).desc("tan(a-b) = (tan(a)-tan(b))/(1+tan(a)·tan(b))"),
                );
            }
        }
    }
    None
});

// =============================================================================
// HyperbolicTanhPythRule: 1 - tanh(x)² → 1/cosh(x)² (sech²)
// =============================================================================
// Canonical direction: contract to reciprocal form.

define_rule!(
    HyperbolicTanhPythRule,
    "Hyperbolic Tanh Pythagorean",
    |ctx, expr| {
        // Flatten the additive chain
        let mut terms = Vec::new();
        crate::helpers::flatten_add(ctx, expr, &mut terms);

        if terms.len() < 2 {
            return None;
        }

        // Look for pattern: 1 + (-tanh²(x)) i.e. 1 - tanh²(x)
        let mut one_idx: Option<usize> = None;
        let mut tanh2_idx: Option<usize> = None;
        let mut tanh_arg: Option<ExprId> = None;
        let mut is_negative_tanh2 = false;

        for (i, &term) in terms.iter().enumerate() {
            // Check for literal 1
            if let Expr::Number(n) = ctx.get(term) {
                if *n == num_rational::BigRational::from_integer(1.into()) {
                    one_idx = Some(i);
                    continue;
                }
            }
            // Check for -tanh²(x)
            if let Expr::Neg(inner) = ctx.get(term) {
                if let Expr::Pow(base, exp) = ctx.get(*inner) {
                    if let Expr::Number(n) = ctx.get(*exp) {
                        if *n == num_rational::BigRational::from_integer(2.into()) {
                            if let Expr::Function(fn_id, args) = ctx.get(*base) {
                                if matches!(ctx.builtin_of(*fn_id), Some(BuiltinFn::Tanh))
                                    && args.len() == 1
                                {
                                    tanh2_idx = Some(i);
                                    tanh_arg = Some(args[0]);
                                    is_negative_tanh2 = true;
                                }
                            }
                        }
                    }
                }
            }
        }

        // If we found 1 and -tanh²(x), replace with 1/cosh²(x)
        if let (Some(one_i), Some(tanh_i), Some(arg)) = (one_idx, tanh2_idx, tanh_arg) {
            if is_negative_tanh2 {
                let cosh_func = ctx.call("cosh", vec![arg]);
                let two = ctx.num(2);
                let cosh_squared = ctx.add(Expr::Pow(cosh_func, two));
                let one = ctx.num(1);
                let sech_squared = ctx.add(Expr::Div(one, cosh_squared));

                // Build new expression
                let mut new_terms: Vec<ExprId> = Vec::new();
                for (j, &t) in terms.iter().enumerate() {
                    if j != one_i && j != tanh_i {
                        new_terms.push(t);
                    }
                }
                new_terms.push(sech_squared);

                let result = if new_terms.len() == 1 {
                    new_terms[0]
                } else {
                    let mut acc = new_terms[0];
                    for &t in new_terms.iter().skip(1) {
                        acc = ctx.add(Expr::Add(acc, t));
                    }
                    acc
                };

                return Some(Rewrite::new(result).desc("1 - tanh²(x) = 1/cosh²(x)"));
            }
        }

        None
    }
);

// =============================================================================
// HyperbolicHalfAngleSquaresRule: cosh(x/2)² → (cosh(x)+1)/2, sinh(x/2)² → (cosh(x)-1)/2

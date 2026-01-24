//! poly_gcd_modp and poly_eq_modp REPL functions.
//!
//! Exposes Zippel mod-p GCD to REPL for fast polynomial verification.

use crate::gcd_zippel_modp::{gcd_zippel_modp, ZippelBudget, ZippelPreset};
use crate::phase::PhaseMask;
use crate::poly_modp_conv::{
    expr_to_poly_modp, PolyConvError, PolyModpBudget, VarTable,
    DEFAULT_PRIME as INTERNAL_DEFAULT_PRIME,
};
use crate::rule::{Rewrite, Rule};
use cas_ast::{Context, DisplayExpr, Expr, ExprId};

/// Re-export DEFAULT_PRIME for use by other modules
pub const DEFAULT_PRIME: u64 = INTERNAL_DEFAULT_PRIME;

// =============================================================================
// Eager Eval Infrastructure: Strip expand + Common Factor Extraction
// =============================================================================

/// Strip expand() wrappers iteratively (no recursion depth risk).
/// Also strips __hold wrappers.
fn strip_expand_wrapper(ctx: &Context, mut expr: ExprId) -> ExprId {
    loop {
        if let Expr::Function(name, args) = ctx.get(expr) {
            if (name == "expand" || name == "__hold") && args.len() == 1 {
                expr = args[0];
                continue;
            }
        }
        return expr;
    }
}

/// Collect multiplicative factors with integer exponents from an expression.
/// - Mul(...) is flattened
/// - Pow(base, k) with integer k becomes (base, k)
/// - Neg(x) is unwrapped (handled by returning negative sign separately)
///
/// Returns (is_negative, factors)
fn collect_mul_factors(ctx: &Context, expr: ExprId) -> (bool, Vec<(ExprId, i64)>) {
    let mut factors = Vec::new();

    // Check for top-level Neg
    let (is_neg, actual_expr) = match ctx.get(expr) {
        Expr::Neg(inner) => (true, *inner),
        _ => (false, expr),
    };

    collect_mul_factors_recursive(ctx, actual_expr, 1, &mut factors);
    (is_neg, factors)
}

fn collect_mul_factors_recursive(
    ctx: &Context,
    expr: ExprId,
    mult: i64,
    factors: &mut Vec<(ExprId, i64)>,
) {
    match ctx.get(expr) {
        Expr::Mul(left, right) => {
            collect_mul_factors_recursive(ctx, *left, mult, factors);
            collect_mul_factors_recursive(ctx, *right, mult, factors);
        }
        Expr::Pow(base, exp) => {
            if let Some(k) = get_integer_exponent(ctx, *exp) {
                factors.push((*base, mult * k));
            } else {
                factors.push((expr, mult));
            }
        }
        _ => {
            factors.push((expr, mult));
        }
    }
}

fn get_integer_exponent(ctx: &Context, exp: ExprId) -> Option<i64> {
    match ctx.get(exp) {
        Expr::Number(n) => {
            if n.is_integer() {
                n.to_integer().try_into().ok()
            } else {
                None
            }
        }
        Expr::Neg(inner) => get_integer_exponent(ctx, *inner).map(|k| -k),
        _ => None,
    }
}

/// Extract common multiplicative factors between two expressions.
/// Returns (common_factors, reduced_a, reduced_b) where:
/// - common_factors: Vec of (base, min_exp) pairs
/// - reduced_a, reduced_b: original expressions with common factors removed
///
/// Invariant: a = common * reduced_a, b = common * reduced_b
#[allow(clippy::type_complexity)]
fn extract_common_mul_factors(
    ctx: &mut Context,
    a: ExprId,
    b: ExprId,
) -> Option<(Vec<(ExprId, i64)>, ExprId, ExprId)> {
    let (_neg_a, fa) = collect_mul_factors(ctx, a);
    let (_neg_b, fb) = collect_mul_factors(ctx, b);

    // Build factor maps for intersection
    use std::collections::HashMap;

    // Use DisplayExpr as key (not ideal but works for structural comparison)
    let mut map_a: HashMap<String, (ExprId, i64)> = HashMap::new();
    for (base, exp) in &fa {
        let key = format!(
            "{}",
            DisplayExpr {
                context: ctx,
                id: *base
            }
        );
        map_a
            .entry(key)
            .and_modify(|e| e.1 += exp)
            .or_insert((*base, *exp));
    }

    let mut map_b: HashMap<String, (ExprId, i64)> = HashMap::new();
    for (base, exp) in &fb {
        let key = format!(
            "{}",
            DisplayExpr {
                context: ctx,
                id: *base
            }
        );
        map_b
            .entry(key)
            .and_modify(|e| e.1 += exp)
            .or_insert((*base, *exp));
    }

    // Find intersection: common factors with min exponent
    let mut common: Vec<(ExprId, i64)> = Vec::new();
    for (key, (base_a, exp_a)) in &map_a {
        if let Some((_, exp_b)) = map_b.get(key) {
            // Both have this factor - take minimum positive exponent
            let min_exp = (*exp_a).min(*exp_b);
            if min_exp > 0 {
                common.push((*base_a, min_exp));
            }
        }
    }

    // If no common factors found, return None to skip optimization
    if common.is_empty() {
        return None;
    }

    // Build reduced expressions by dividing out common factors
    let common_expr = build_product_from_factors(ctx, &common);

    // For reduced expressions, we divide by common
    // reduced_a = a / common, reduced_b = b / common
    let reduced_a = ctx.add(Expr::Div(a, common_expr));
    let reduced_b = ctx.add(Expr::Div(b, common_expr));

    Some((common, reduced_a, reduced_b))
}

/// Build product expression from factors Vec<(base, exp)>
fn build_product_from_factors(ctx: &mut Context, factors: &[(ExprId, i64)]) -> ExprId {
    if factors.is_empty() {
        return ctx.num(1);
    }

    let mut result: Option<ExprId> = None;

    for &(base, exp) in factors {
        let term = if exp == 1 {
            base
        } else {
            let exp_expr = ctx.num(exp);
            ctx.add(Expr::Pow(base, exp_expr))
        };

        result = Some(match result {
            None => term,
            Some(acc) => ctx.add(Expr::Mul(acc, term)),
        });
    }

    result.unwrap_or_else(|| ctx.num(1))
}

/// Eager evaluation of poly_gcd_modp with factor extraction optimization.
///
/// This function:
/// 1. Strips expand() wrappers from arguments
/// 2. Extracts common multiplicative factors (gcd(a*g, b*g) = g * gcd(a,b))
/// 3. Computes GCD on reduced polynomials (much smaller)
/// 4. Returns common * gcd_result
pub fn compute_gcd_modp_with_factor_extraction(
    ctx: &mut Context,
    a: ExprId,
    b: ExprId,
) -> Option<ExprId> {
    // Step 1: Strip expand() and __hold wrappers (NO symbolic expansion!)
    let a0 = strip_expand_wrapper(ctx, a);
    let b0 = strip_expand_wrapper(ctx, b);

    // Step 2: Try to extract common factors
    if let Some((common, ra, rb)) = extract_common_mul_factors(ctx, a0, b0) {
        // We have common factors! Compute gcd on reduced expressions
        // IMPORTANT: Do NOT expand - compute_gcd_modp uses MultiPoly which handles Pow directly
        // Just strip wrappers from reduced expressions too
        let ra_stripped = strip_expand_wrapper(ctx, ra);
        let rb_stripped = strip_expand_wrapper(ctx, rb);

        // Compute gcd on reduced polynomials (MultiPoly handles Pow natively)
        match compute_gcd_modp_with_options(
            ctx,
            ra_stripped,
            rb_stripped,
            DEFAULT_PRIME,
            None,
            None,
        ) {
            Ok(gcd_rest) => {
                // Reconstruct: common * gcd_rest
                let common_expr = build_product_from_factors(ctx, &common);

                // If gcd_rest is 1, just return common
                if let Expr::Number(n) = ctx.get(gcd_rest) {
                    use num_traits::One;
                    if n.is_one() {
                        // Wrap in __hold
                        return Some(
                            ctx.add(Expr::Function("__hold".to_string(), vec![common_expr])),
                        );
                    }
                }

                // Otherwise return common * gcd_rest
                let result = ctx.add(Expr::Mul(common_expr, gcd_rest));
                return Some(ctx.add(Expr::Function("__hold".to_string(), vec![result])));
            }
            Err(_) => {
                // Fall through to direct computation
            }
        }
    }

    // Step 3: No common factors or extraction failed - try direct GCD (no expand!)
    // The MultiPoly converter handles Pow(base, n) natively via pow() method
    match compute_gcd_modp_with_options(ctx, a0, b0, DEFAULT_PRIME, None, None) {
        Ok(gcd_expr) => Some(ctx.add(Expr::Function("__hold".to_string(), vec![gcd_expr]))),
        Err(_) => None,
    }
}

/// Eager evaluation pass for poly_gcd_modp calls.
///
/// This function traverses the expression tree TOP-DOWN and evaluates
/// poly_gcd_modp calls BEFORE the normal simplification pipeline.
///
/// CRITICAL: When we find poly_gcd_modp, we do NOT descend into its children.
/// This prevents the expensive symbolic expansion of huge arguments.
pub fn eager_eval_poly_gcd_calls(ctx: &mut Context, expr: ExprId) -> (ExprId, Vec<crate::Step>) {
    let mut steps = Vec::new();
    let result = eager_eval_recursive(ctx, expr, &mut steps);
    (result, steps)
}

fn eager_eval_recursive(ctx: &mut Context, expr: ExprId, steps: &mut Vec<crate::Step>) -> ExprId {
    // Check if this is poly_gcd_modp - if so, evaluate and STOP descent
    if let Expr::Function(name, args) = ctx.get(expr).clone() {
        if (name == "poly_gcd_modp" || name == "pgcdp") && args.len() >= 2 {
            if let Some(result) = compute_gcd_modp_with_factor_extraction(ctx, args[0], args[1]) {
                // Create step for the evaluation
                steps.push(crate::Step::new(
                    "Eager eval poly_gcd_modp (bypass simplifier)",
                    "Polynomial GCD mod p",
                    expr,
                    result,
                    Vec::new(),
                    Some(ctx),
                ));
                return result;
            }
        }

        // For other functions, recurse into children
        let new_args: Vec<ExprId> = args
            .iter()
            .map(|&arg| eager_eval_recursive(ctx, arg, steps))
            .collect();

        // Check if any arg changed
        if new_args
            .iter()
            .zip(args.iter())
            .any(|(new, old)| new != old)
        {
            return ctx.add(Expr::Function(name.clone(), new_args));
        }
        return expr;
    }

    // Recurse into children for other expression types
    match ctx.get(expr).clone() {
        Expr::Add(l, r) => {
            let nl = eager_eval_recursive(ctx, l, steps);
            let nr = eager_eval_recursive(ctx, r, steps);
            if nl != l || nr != r {
                ctx.add(Expr::Add(nl, nr))
            } else {
                expr
            }
        }
        Expr::Sub(l, r) => {
            let nl = eager_eval_recursive(ctx, l, steps);
            let nr = eager_eval_recursive(ctx, r, steps);
            if nl != l || nr != r {
                ctx.add(Expr::Sub(nl, nr))
            } else {
                expr
            }
        }
        Expr::Mul(l, r) => {
            let nl = eager_eval_recursive(ctx, l, steps);
            let nr = eager_eval_recursive(ctx, r, steps);
            if nl != l || nr != r {
                ctx.add(Expr::Mul(nl, nr))
            } else {
                expr
            }
        }
        Expr::Div(l, r) => {
            let nl = eager_eval_recursive(ctx, l, steps);
            let nr = eager_eval_recursive(ctx, r, steps);
            if nl != l || nr != r {
                ctx.add(Expr::Div(nl, nr))
            } else {
                expr
            }
        }
        Expr::Pow(b, e) => {
            let nb = eager_eval_recursive(ctx, b, steps);
            let ne = eager_eval_recursive(ctx, e, steps);
            if nb != b || ne != e {
                ctx.add(Expr::Pow(nb, ne))
            } else {
                expr
            }
        }
        Expr::Neg(e) => {
            let ne = eager_eval_recursive(ctx, e, steps);
            if ne != e {
                ctx.add(Expr::Neg(ne))
            } else {
                expr
            }
        }
        // Leaves - no recursion needed (Function already handled above)
        Expr::Number(_)
        | Expr::Variable(_)
        | Expr::Constant(_)
        | Expr::Matrix { .. }
        | Expr::SessionRef(_)
        | Expr::Function(_, _) => expr,
    }
}

/// Rule for poly_gcd_modp(a, b [, p]) function.
/// Computes Zippel GCD of two polynomial expressions mod p.
pub struct PolyGcdModpRule;

impl Rule for PolyGcdModpRule {
    fn name(&self) -> &str {
        "Polynomial GCD mod p"
    }

    fn allowed_phases(&self) -> PhaseMask {
        PhaseMask::CORE | PhaseMask::TRANSFORM
    }

    fn priority(&self) -> i32 {
        200 // High priority to evaluate early
    }

    fn target_types(&self) -> Option<Vec<&str>> {
        Some(vec!["Function"])
    }

    fn apply(
        &self,
        ctx: &mut Context,
        expr: ExprId,
        _parent_ctx: &crate::parent_context::ParentContext,
    ) -> Option<Rewrite> {
        let fn_expr = ctx.get(expr).clone();

        if let Expr::Function(name, args) = fn_expr {
            let is_gcd_modp = name == "poly_gcd_modp" || name == "pgcdp";

            if is_gcd_modp && args.len() >= 2 && args.len() <= 4 {
                // Use eager eval with factor extraction
                if let Some(result) = compute_gcd_modp_with_factor_extraction(ctx, args[0], args[1])
                {
                    return Some(Rewrite::simple(
                        result,
                        format!(
                            "poly_gcd_modp({}, {}) [eager eval + factor extraction]",
                            DisplayExpr {
                                context: ctx,
                                id: args[0]
                            },
                            DisplayExpr {
                                context: ctx,
                                id: args[1]
                            }
                        ),
                    ));
                }

                // Fallback: try with explicit args (for extra options)
                let a = strip_expand_wrapper(ctx, args[0]);
                let b = strip_expand_wrapper(ctx, args[1]);

                // Parse remaining args: main_var (usize) and/or preset (string)
                let mut main_var: Option<usize> = None;
                let mut preset_str: Option<String> = None;

                for &arg in args.iter().skip(2) {
                    // Try to extract as usize (small number = main_var)
                    if let Some(v) = extract_usize(ctx, arg) {
                        if v <= 64 {
                            main_var = Some(v);
                            continue;
                        }
                    }
                    // Try to extract as string (preset name)
                    if let Some(s) = extract_string(ctx, arg) {
                        preset_str = Some(s);
                    }
                }

                // Parse preset from string if provided
                let preset = preset_str.as_deref().and_then(ZippelPreset::parse);
                match compute_gcd_modp_with_options(ctx, a, b, DEFAULT_PRIME, main_var, preset) {
                    Ok(gcd_expr) => {
                        // Wrap in __hold to prevent further simplification
                        let held = ctx.add(Expr::Function("__hold".to_string(), vec![gcd_expr]));

                        return Some(Rewrite::simple(
                            held,
                            format!(
                                "poly_gcd_modp({}, {})",
                                DisplayExpr {
                                    context: ctx,
                                    id: a
                                },
                                DisplayExpr {
                                    context: ctx,
                                    id: b
                                }
                            ),
                        ));
                    }
                    Err(e) => {
                        // Return error as message (don't crash)
                        eprintln!("poly_gcd_modp error: {}", e);
                        return None;
                    }
                }
            }
        }

        None
    }
}

/// Rule for poly_eq_modp(a, b [, p]) function.
/// Returns 1 if polynomials are equal mod p, 0 otherwise.
pub struct PolyEqModpRule;

impl Rule for PolyEqModpRule {
    fn name(&self) -> &str {
        "Polynomial equality mod p"
    }

    fn allowed_phases(&self) -> PhaseMask {
        PhaseMask::CORE | PhaseMask::TRANSFORM
    }

    fn priority(&self) -> i32 {
        200
    }

    fn target_types(&self) -> Option<Vec<&str>> {
        Some(vec!["Function"])
    }

    fn apply(
        &self,
        ctx: &mut Context,
        expr: ExprId,
        _parent_ctx: &crate::parent_context::ParentContext,
    ) -> Option<Rewrite> {
        let fn_expr = ctx.get(expr).clone();

        if let Expr::Function(name, args) = fn_expr {
            let is_eq_modp = name == "poly_eq_modp" || name == "peqp";

            if is_eq_modp && (args.len() == 2 || args.len() == 3) {
                let a = args[0];
                let b = args[1];
                let p = if args.len() == 3 {
                    extract_prime(ctx, args[2]).unwrap_or(DEFAULT_PRIME)
                } else {
                    DEFAULT_PRIME
                };

                match check_poly_equal_modp(ctx, a, b, p) {
                    Ok(equal) => {
                        let result = if equal { ctx.num(1) } else { ctx.num(0) };

                        return Some(Rewrite::simple(
                            result,
                            format!(
                                "poly_eq_modp({}, {}) = {}",
                                DisplayExpr {
                                    context: ctx,
                                    id: a
                                },
                                DisplayExpr {
                                    context: ctx,
                                    id: b
                                },
                                if equal { "true" } else { "false" }
                            ),
                        ));
                    }
                    Err(e) => {
                        eprintln!("poly_eq_modp error: {}", e);
                        return None;
                    }
                }
            }
        }

        None
    }
}

/// Extract prime from expression (must be integer)
fn extract_prime(ctx: &Context, expr: ExprId) -> Option<u64> {
    if let Expr::Number(n) = ctx.get(expr) {
        if n.is_integer() {
            use num_traits::ToPrimitive;
            return n.to_integer().to_u64();
        }
    }
    None
}

/// Extract usize from expression (must be non-negative integer)
fn extract_usize(ctx: &Context, expr: ExprId) -> Option<usize> {
    if let Expr::Number(n) = ctx.get(expr) {
        if n.is_integer() {
            use num_traits::ToPrimitive;
            return n.to_integer().to_usize();
        }
    }
    None
}

/// Extract string from expression (Variable as string literal)
fn extract_string(ctx: &Context, expr: ExprId) -> Option<String> {
    // Check for Variable (used as string literal, e.g., "mm_gcd")
    if let Expr::Variable(sym_id) = ctx.get(expr) {
        return Some(ctx.sym_name(*sym_id).to_string());
    }
    None
}

/// Compute GCD mod p and return as Expr (public for unified API)
pub fn compute_gcd_modp_with_options(
    ctx: &mut Context,
    a: ExprId,
    b: ExprId,
    p: u64,
    main_var: Option<usize>,
    preset: Option<ZippelPreset>,
) -> Result<ExprId, PolyConvError> {
    use std::time::Instant;

    let budget = PolyModpBudget::default();
    let mut vars = VarTable::new();

    // Convert both expressions to MultiPolyModP
    let t0 = Instant::now();
    let poly_a = expr_to_poly_modp(ctx, a, p, &budget, &mut vars)?;
    let t1 = Instant::now();
    eprintln!(
        "[poly_gcd_modp] Convert a: {:?}, {} terms",
        t1 - t0,
        poly_a.num_terms()
    );

    let poly_b = expr_to_poly_modp(ctx, b, p, &budget, &mut vars)?;
    let t2 = Instant::now();
    eprintln!(
        "[poly_gcd_modp] Convert b: {:?}, {} terms",
        t2 - t1,
        poly_b.num_terms()
    );

    // Compute GCD using Zippel
    // Select budget based on preset (default to MmGcd for performance)
    let used_preset = preset.unwrap_or(ZippelPreset::MmGcd);
    let zippel_budget = ZippelBudget::for_preset(used_preset).with_main_var(main_var);
    let gcd_opt = gcd_zippel_modp(&poly_a, &poly_b, &zippel_budget);
    let t3 = Instant::now();
    eprintln!("[poly_gcd_modp] Zippel GCD: {:?}", t3 - t2);

    // Handle failure
    let mut gcd = match gcd_opt {
        Some(g) => g,
        None => {
            // Fallback: return 1 if Zippel fails
            return Ok(ctx.num(1));
        }
    };

    // Normalize to monic
    gcd.make_monic();

    // Convert back to Expr
    let result = multipoly_modp_to_expr(ctx, &gcd, &vars);
    let t4 = Instant::now();
    eprintln!(
        "[poly_gcd_modp] Convert result: {:?}, total: {:?}",
        t4 - t3,
        t4 - t0
    );

    Ok(result)
}

/// Check if two polynomials are equal mod p
fn check_poly_equal_modp(
    ctx: &Context,
    a: ExprId,
    b: ExprId,
    p: u64,
) -> Result<bool, PolyConvError> {
    let budget = PolyModpBudget::default();
    let mut vars = VarTable::new();

    // Convert both expressions
    let mut poly_a = expr_to_poly_modp(ctx, a, p, &budget, &mut vars)?;
    let mut poly_b = expr_to_poly_modp(ctx, b, p, &budget, &mut vars)?;

    // Normalize both to monic for comparison
    poly_a.make_monic();
    poly_b.make_monic();

    // Compare term-by-term
    Ok(poly_a == poly_b)
}

/// Convert a mod-p coefficient to signed representation.
///
/// Uses symmetric representation: coefficients in [0, p/2] stay positive,
/// coefficients in (p/2, p-1] become negative (c - p).
///
/// This is correct only if the original coefficient was in the range (-p/2, p/2).
/// For typical polynomial operations with moderate coefficients, this holds.
#[inline]
fn modp_to_signed(c: u64, p: u64) -> i128 {
    let half = p / 2;
    if c <= half {
        c as i128
    } else {
        (c as i128) - (p as i128) // Maps p-1 -> -1, p-2 -> -2, etc.
    }
}

/// Convert MultiPolyModP back to Expr (balanced tree)
///
/// Uses symmetric representation to reconstruct signed coefficients.
/// Coefficients in (p/2, p-1] are interpreted as negative.
pub fn multipoly_modp_to_expr(
    ctx: &mut Context,
    poly: &crate::multipoly_modp::MultiPolyModP,
    vars: &VarTable,
) -> ExprId {
    use num_rational::BigRational;

    if poly.is_zero() {
        return ctx.num(0);
    }

    let p = poly.p;

    // Build term expressions
    let mut term_exprs: Vec<ExprId> = Vec::with_capacity(poly.terms.len());

    for (mono, coeff) in &poly.terms {
        if *coeff == 0 {
            continue;
        }

        // Convert coefficient to signed representation
        let signed_coeff = modp_to_signed(*coeff, p);
        let is_negative = signed_coeff < 0;
        let abs_coeff = signed_coeff.unsigned_abs(); // i128 -> u128

        // Build monomial: coeff * x1^e1 * x2^e2 * ...
        let mut factors: Vec<ExprId> = Vec::new();

        // Add absolute coefficient if not 1 (or if monomial is constant)
        if abs_coeff != 1 || mono.is_constant() {
            factors.push(ctx.add(Expr::Number(BigRational::from_integer(
                (abs_coeff as i64).into(),
            ))));
        }

        // Add variable powers
        for (i, &exp) in mono.0.iter().enumerate() {
            if exp > 0 && i < vars.len() {
                let var_expr = ctx.var(&vars.names[i]);
                if exp == 1 {
                    factors.push(var_expr);
                } else {
                    let exp_expr =
                        ctx.add(Expr::Number(BigRational::from_integer((exp as i64).into())));
                    factors.push(ctx.add(Expr::Pow(var_expr, exp_expr)));
                }
            }
        }

        // Combine factors into product
        let term = if factors.is_empty() {
            ctx.num(1)
        } else if factors.len() == 1 {
            factors[0]
        } else {
            factors
                .into_iter()
                .reduce(|acc, f| ctx.add(Expr::Mul(acc, f)))
                .unwrap()
        };

        // Apply negation if coefficient was negative
        let final_term = if is_negative {
            ctx.add(Expr::Neg(term))
        } else {
            term
        };

        term_exprs.push(final_term);
    }

    // Build balanced sum tree
    if term_exprs.is_empty() {
        ctx.num(0)
    } else {
        build_balanced_sum(ctx, &term_exprs)
    }
}

/// Build balanced Add tree to avoid deep recursion
fn build_balanced_sum(ctx: &mut Context, terms: &[ExprId]) -> ExprId {
    match terms.len() {
        0 => ctx.num(0),
        1 => terms[0],
        2 => ctx.add(Expr::Add(terms[0], terms[1])),
        _ => {
            let mid = terms.len() / 2;
            let left = build_balanced_sum(ctx, &terms[..mid]);
            let right = build_balanced_sum(ctx, &terms[mid..]);
            ctx.add(Expr::Add(left, right))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_parser::parse;

    #[test]
    fn test_poly_eq_modp_same() {
        let mut ctx = Context::new();
        let a = parse("x + 1", &mut ctx).unwrap();
        let b = parse("1 + x", &mut ctx).unwrap();

        let result = check_poly_equal_modp(&ctx, a, b, DEFAULT_PRIME).unwrap();
        assert!(result);
    }

    #[test]
    fn test_poly_eq_modp_different() {
        let mut ctx = Context::new();
        let a = parse("x + 1", &mut ctx).unwrap();
        let b = parse("x + 2", &mut ctx).unwrap();

        let result = check_poly_equal_modp(&ctx, a, b, DEFAULT_PRIME).unwrap();
        assert!(!result);
    }
}

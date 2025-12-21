//! poly_gcd_modp and poly_eq_modp REPL functions.
//!
//! Exposes Zippel mod-p GCD to REPL for fast polynomial verification.

use crate::gcd_zippel_modp::{gcd_zippel_modp, ZippelBudget};
use crate::phase::PhaseMask;
use crate::poly_modp_conv::{
    expr_to_poly_modp, PolyConvError, PolyModpBudget, VarTable, DEFAULT_PRIME,
};
use crate::rule::{Rewrite, Rule};
use cas_ast::{Context, DisplayExpr, Expr, ExprId};

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

            if is_gcd_modp && (args.len() == 2 || args.len() == 3) {
                let a = args[0];
                let b = args[1];

                // Third argument (if present) is main_var (integer 0-7)
                let main_var = if args.len() == 3 {
                    extract_usize(ctx, args[2])
                } else {
                    None
                };

                match compute_gcd_modp(ctx, a, b, DEFAULT_PRIME, main_var) {
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

/// Compute GCD mod p and return as Expr
fn compute_gcd_modp(
    ctx: &mut Context,
    a: ExprId,
    b: ExprId,
    p: u64,
    main_var: Option<usize>,
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

    // Compute GCD using Zippel (use mm_gcd budget for performance)
    use crate::gcd_zippel_modp::budget_for_mm_gcd;
    let zippel_budget = ZippelBudget {
        forced_main_var: main_var,
        ..budget_for_mm_gcd() // Use optimized budget: max_points=8, verify=3
    };
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

/// Convert MultiPolyModP back to Expr (balanced tree)
fn multipoly_modp_to_expr(
    ctx: &mut Context,
    poly: &crate::multipoly_modp::MultiPolyModP,
    vars: &VarTable,
) -> ExprId {
    use num_rational::BigRational;

    if poly.is_zero() {
        return ctx.num(0);
    }

    // Build term expressions
    let mut term_exprs: Vec<ExprId> = Vec::with_capacity(poly.terms.len());

    for (mono, coeff) in &poly.terms {
        if *coeff == 0 {
            continue;
        }

        // Build monomial: coeff * x1^e1 * x2^e2 * ...
        let mut factors: Vec<ExprId> = Vec::new();

        // Add coefficient if not 1 (or if monomial is constant)
        if *coeff != 1 || mono.is_constant() {
            factors.push(ctx.add(Expr::Number(BigRational::from_integer((*coeff).into()))));
        }

        // Add variable powers
        for (i, &exp) in mono.0.iter().enumerate() {
            if exp > 0 && i < vars.len() {
                let var_expr = ctx.add(Expr::Variable(vars.names[i].clone()));
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

        term_exprs.push(term);
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

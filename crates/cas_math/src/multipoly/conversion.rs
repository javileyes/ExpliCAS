//! Conversion between AST expressions and MultiPoly representation.

use num_rational::BigRational;
use num_traits::{One, Zero};
use std::collections::BTreeSet;

use cas_ast::{Context, Expr, ExprId};

use super::{Monomial, MultiPoly, PolyBudget, PolyError};

// =============================================================================
// AST → MultiPoly
// =============================================================================

/// Collect all variable names from expression.
///
/// Delegates to the canonical `cas_ast::collect_variables` traversal,
/// converting to `BTreeSet` for deterministic variable ordering.
pub fn collect_poly_vars(ctx: &Context, expr: ExprId) -> BTreeSet<String> {
    cas_ast::collect_variables(ctx, expr).into_iter().collect()
}

/// Convert expression to MultiPoly
pub fn multipoly_from_expr(
    ctx: &Context,
    expr: ExprId,
    budget: &PolyBudget,
) -> Result<MultiPoly, PolyError> {
    // Collect and sort variables
    let vars_set = collect_poly_vars(ctx, expr);
    let vars: Vec<String> = vars_set.into_iter().collect();

    // Convert
    from_expr_recursive(ctx, expr, &vars, budget)
}

fn from_expr_recursive(
    ctx: &Context,
    expr: ExprId,
    vars: &[String],
    budget: &PolyBudget,
) -> Result<MultiPoly, PolyError> {
    match ctx.get(expr) {
        Expr::Number(n) => Ok(MultiPoly {
            vars: vars.to_vec(),
            terms: if n.is_zero() {
                vec![]
            } else {
                vec![(n.clone(), vec![0; vars.len()])]
            },
        }),

        Expr::Variable(sym_id) => {
            let name = ctx.sym_name(*sym_id);
            let idx = vars
                .iter()
                .position(|v| v == name)
                .ok_or(PolyError::NonPolynomial)?;
            let mut mono = vec![0; vars.len()];
            mono[idx] = 1;
            Ok(MultiPoly {
                vars: vars.to_vec(),
                terms: vec![(BigRational::one(), mono)],
            })
        }

        Expr::Neg(a) => {
            let p = from_expr_recursive(ctx, *a, vars, budget)?;
            Ok(p.neg())
        }

        Expr::Add(a, b) => {
            let pa = from_expr_recursive(ctx, *a, vars, budget)?;
            let pb = from_expr_recursive(ctx, *b, vars, budget)?;
            let result = pa.add(&pb)?;
            check_budget(&result, budget)?;
            Ok(result)
        }

        Expr::Sub(a, b) => {
            let pa = from_expr_recursive(ctx, *a, vars, budget)?;
            let pb = from_expr_recursive(ctx, *b, vars, budget)?;
            let result = pa.sub(&pb)?;
            check_budget(&result, budget)?;
            Ok(result)
        }

        Expr::Mul(a, b) => {
            let pa = from_expr_recursive(ctx, *a, vars, budget)?;
            let pb = from_expr_recursive(ctx, *b, vars, budget)?;
            pa.mul(&pb, budget)
        }

        Expr::Div(a, b) => {
            let pa = from_expr_recursive(ctx, *a, vars, budget)?;
            let pb = from_expr_recursive(ctx, *b, vars, budget)?;
            // Only allow division by constants
            if let Some(c) = pb.constant_value() {
                if c.is_zero() {
                    return Err(PolyError::NonPolynomial);
                }
                Ok(pa.mul_scalar(&(BigRational::one() / c)))
            } else {
                Err(PolyError::NonConstantDivision)
            }
        }

        Expr::Pow(base, exp) => {
            // Exponent must be non-negative integer constant
            if let Expr::Number(n) = ctx.get(*exp) {
                if n.is_integer() && *n >= BigRational::zero() {
                    let e: u32 = n
                        .to_integer()
                        .try_into()
                        .map_err(|_| PolyError::BadExponent)?;

                    // Check if exponent exceeds budget for Pow(sum, n) expansion
                    // Only apply budget check if base is a sum (Add) - constants/vars are cheap
                    if e > budget.max_pow_exp && matches!(ctx.get(*base), Expr::Add(_, _)) {
                        return Err(PolyError::BudgetExceeded);
                    }

                    let pb = from_expr_recursive(ctx, *base, vars, budget)?;
                    return pow_poly(&pb, e, budget);
                }
            }
            Err(PolyError::BadExponent)
        }

        Expr::Constant(_) => {
            // Constants like pi, e - treat as non-polynomial
            Err(PolyError::NonPolynomial)
        }

        _ => Err(PolyError::NonPolynomial),
    }
}

fn pow_poly(p: &MultiPoly, exp: u32, budget: &PolyBudget) -> Result<MultiPoly, PolyError> {
    if exp == 0 {
        return Ok(MultiPoly::one(p.vars.clone()));
    }
    if exp == 1 {
        return Ok(p.clone());
    }

    // Binary exponentiation
    let mut result = MultiPoly::one(p.vars.clone());
    let mut base = p.clone();
    let mut e = exp;

    while e > 0 {
        if e & 1 == 1 {
            result = result.mul(&base, budget)?;
        }
        e >>= 1;
        if e > 0 {
            base = base.mul(&base, budget)?;
        }
    }

    Ok(result)
}

fn check_budget(p: &MultiPoly, budget: &PolyBudget) -> Result<(), PolyError> {
    if p.num_terms() > budget.max_terms {
        return Err(PolyError::BudgetExceeded);
    }
    if p.total_degree() > budget.max_total_degree {
        return Err(PolyError::BudgetExceeded);
    }
    Ok(())
}

// =============================================================================
// MultiPoly → AST
// =============================================================================

/// Convert MultiPoly back to expression
pub fn multipoly_to_expr(p: &MultiPoly, ctx: &mut Context) -> ExprId {
    if p.is_zero() {
        return ctx.num(0);
    }

    // Collect terms with their total degree for sorting
    let mut terms_with_degree: Vec<(usize, ExprId)> = Vec::new();

    for (coeff, mono) in &p.terms {
        let term = build_term_expr(ctx, coeff, mono, &p.vars);
        let total_deg: usize = mono.iter().map(|&e| e as usize).sum();
        terms_with_degree.push((total_deg, term));
    }

    // Sort by descending total degree for canonical polynomial form (x² - 4x + 8)
    terms_with_degree.sort_by(|a, b| b.0.cmp(&a.0));

    // Sum all terms
    let mut result = terms_with_degree[0].1;
    for (_, t) in &terms_with_degree[1..] {
        result = ctx.add_raw(Expr::Add(result, *t));
    }

    result
}

fn build_term_expr(
    ctx: &mut Context,
    coeff: &BigRational,
    mono: &Monomial,
    vars: &[String],
) -> ExprId {
    // Build monomial part: x^a * y^b * ...
    let mut factors: Vec<ExprId> = Vec::new();

    for (i, &exp) in mono.iter().enumerate() {
        if exp > 0 {
            let var_expr = ctx.var(&vars[i]);
            if exp == 1 {
                factors.push(var_expr);
            } else {
                let exp_expr = ctx.num(exp as i64);
                factors.push(ctx.add_raw(Expr::Pow(var_expr, exp_expr)));
            }
        }
    }

    // Combine monomial factors
    let mono_expr = if factors.is_empty() {
        None
    } else {
        let mut m = factors[0];
        for &f in &factors[1..] {
            m = ctx.add_raw(Expr::Mul(m, f));
        }
        Some(m)
    };

    // Combine with coefficient
    if coeff.is_one() {
        mono_expr.unwrap_or_else(|| ctx.num(1))
    } else if *coeff == -BigRational::one() {
        let m = mono_expr.unwrap_or_else(|| ctx.num(1));
        ctx.add_raw(Expr::Neg(m))
    } else {
        let c_expr = ctx.add_raw(Expr::Number(coeff.clone()));
        if let Some(m) = mono_expr {
            ctx.add_raw(Expr::Mul(c_expr, m))
        } else {
            c_expr
        }
    }
}

//! Symbolic 2×2 linear systems: the unknowns list drives linearity and every
//! OTHER variable is a parameter folded into exact `MultiPoly` coefficients.
//!
//! The rational path (`solve2.rs`) owns fully numeric systems; this sibling
//! covers `a·x + y = 1`-style systems by exact Cramer over multipoly
//! coefficients. A symbolic determinant is never guessed nonzero: the emitted
//! solution carries `det ≠ 0` as a structured condition (result-as-contract).

use std::collections::BTreeMap;

use cas_ast::{Context, Expr, ExprId};
use cas_math::multipoly::{
    multipoly_from_expr, multipoly_to_expr, Monomial, MultiPoly, PolyBudget,
};
use num_rational::BigRational;
use num_traits::{Signed, Zero};

use super::{with_equation_index, LinearSystemError};

/// Wider than the rational extractor's budget on purpose: parameters may carry
/// higher powers (`a^2·x`) without the SYSTEM being nonlinear in the unknowns.
fn symbolic_budget() -> PolyBudget {
    PolyBudget {
        max_terms: 200,
        max_total_degree: 16,
        max_pow_exp: 16,
    }
}

pub(crate) enum Symbolic2x2Outcome {
    /// Unique solution. `det_condition` names the nonzero requirement when the
    /// determinant stays symbolic (`None` when it folded to a nonzero rational).
    Unique {
        values: Vec<ExprId>,
        det_condition: Option<ExprId>,
    },
    /// `det ≡ 0` as a polynomial in the parameters: degenerate for EVERY
    /// parameter value, and rank classification with symbolic entries is a
    /// future rung — decline honestly instead of branching blind.
    DegenerateSymbolic,
}

/// Linear coefficients `a·x + b·y + c` where each entry is a polynomial in the
/// PARAMETER variables (everything except the two unknowns).
struct SymCoeffs {
    a: MultiPoly,
    b: MultiPoly,
    c: MultiPoly,
}

fn extract_symbolic_coeffs(
    ctx: &Context,
    expr: ExprId,
    var_x: &str,
    var_y: &str,
) -> Result<SymCoeffs, LinearSystemError> {
    let poly = multipoly_from_expr(ctx, expr, &symbolic_budget())
        .map_err(LinearSystemError::PolyConversion)?;
    let idx_x = poly.vars.iter().position(|v| v == var_x);
    let idx_y = poly.vars.iter().position(|v| v == var_y);

    let param_vars: Vec<String> = poly
        .vars
        .iter()
        .enumerate()
        .filter(|(i, _)| Some(*i) != idx_x && Some(*i) != idx_y)
        .map(|(_, v)| v.clone())
        .collect();

    let mut a_map: BTreeMap<Monomial, BigRational> = BTreeMap::new();
    let mut b_map: BTreeMap<Monomial, BigRational> = BTreeMap::new();
    let mut c_map: BTreeMap<Monomial, BigRational> = BTreeMap::new();
    for (coef, mono) in &poly.terms {
        let ex = idx_x.map_or(0, |i| mono[i]);
        let ey = idx_y.map_or(0, |i| mono[i]);
        let param_mono: Monomial = mono
            .iter()
            .enumerate()
            .filter(|(i, _)| Some(*i) != idx_x && Some(*i) != idx_y)
            .map(|(_, &e)| e)
            .collect();
        let bucket = match (ex, ey) {
            (0, 0) => &mut c_map,
            (1, 0) => &mut a_map,
            (0, 1) => &mut b_map,
            _ => {
                return Err(LinearSystemError::NotLinear(
                    "degree > 1 in the system".to_string(),
                ))
            }
        };
        let entry = bucket.entry(param_mono).or_insert_with(BigRational::zero);
        *entry = &*entry + coef;
    }
    let mk = |map: BTreeMap<Monomial, BigRational>| MultiPoly::from_map(param_vars.clone(), map);
    Ok(SymCoeffs {
        a: mk(a_map),
        b: mk(b_map),
        c: mk(c_map),
    })
}

/// Union of two parameter-variable lists, deterministic (sorted).
fn union_vars(lhs: &[String], rhs: &[String]) -> Vec<String> {
    let mut union: Vec<String> = lhs.to_vec();
    for v in rhs {
        if !union.contains(v) {
            union.push(v.clone());
        }
    }
    union.sort();
    union
}

/// Render `num/den` as an expression: exact polynomial quotient when it
/// divides, otherwise a `Div` node over sign/content-normalized parts.
fn quotient_expr(ctx: &mut Context, num: &MultiPoly, den: &MultiPoly) -> ExprId {
    if num.is_zero() {
        return ctx.num(0);
    }
    if let Some(k) = den.constant_value() {
        let scaled = num.mul_scalar(&(BigRational::from_integer(1.into()) / k));
        return multipoly_to_expr(&scaled, ctx);
    }
    if let Some(q) = num.div_exact(den) {
        return multipoly_to_expr(&q, ctx);
    }
    let num_expr = multipoly_to_expr(num, ctx);
    let den_expr = multipoly_to_expr(den, ctx);
    ctx.add(Expr::Div(num_expr, den_expr))
}

/// Normalize the determinant for display/conditions: primitive part with a
/// positive leading coefficient. Returns the scalar `s` with `det = s·det'`
/// so numerators can absorb it and quotients stay identical.
fn normalize_det(det: &MultiPoly) -> (BigRational, MultiPoly) {
    let (content, mut prim) = det.primitive_part();
    let mut scale = content;
    let leading_negative = prim
        .leading_term_lex()
        .map(|(coef, _)| coef.is_negative())
        .unwrap_or(false);
    if leading_negative {
        prim = prim.neg();
        scale = -scale;
    }
    (scale, prim)
}

/// Exact Cramer for `expr1 = 0`, `expr2 = 0` linear in `var_x`/`var_y` with
/// polynomial parameter coefficients. Errors keep the rational path's
/// `NotLinear` vocabulary so callers/messages stay uniform.
pub(crate) fn solve_2x2_symbolic(
    ctx: &mut Context,
    expr1: ExprId,
    expr2: ExprId,
    var_x: &str,
    var_y: &str,
) -> Result<Symbolic2x2Outcome, LinearSystemError> {
    let c1 =
        extract_symbolic_coeffs(ctx, expr1, var_x, var_y).map_err(|e| with_equation_index(e, 1))?;
    let c2 =
        extract_symbolic_coeffs(ctx, expr2, var_x, var_y).map_err(|e| with_equation_index(e, 2))?;

    let vars = union_vars(&c1.a.vars, &c2.a.vars);
    let a1 = c1.a.align_vars(&vars);
    let b1 = c1.b.align_vars(&vars);
    let d1 = c1.c.align_vars(&vars).neg();
    let a2 = c2.a.align_vars(&vars);
    let b2 = c2.b.align_vars(&vars);
    let d2 = c2.c.align_vars(&vars).neg();

    let budget = symbolic_budget();
    let mul =
        |p: &MultiPoly, q: &MultiPoly| p.mul(q, &budget).map_err(LinearSystemError::PolyConversion);
    let sub = |p: MultiPoly, q: MultiPoly| p.sub(&q).map_err(LinearSystemError::PolyConversion);

    let det = sub(mul(&a1, &b2)?, mul(&a2, &b1)?)?;
    if det.is_zero() {
        return Ok(Symbolic2x2Outcome::DegenerateSymbolic);
    }
    let x_num = sub(mul(&d1, &b2)?, mul(&b1, &d2)?)?;
    let y_num = sub(mul(&a1, &d2)?, mul(&d1, &a2)?)?;

    if let Some(k) = det.constant_value() {
        let inv = BigRational::from_integer(1.into()) / k;
        let x = multipoly_to_expr(&x_num.mul_scalar(&inv), ctx);
        let y = multipoly_to_expr(&y_num.mul_scalar(&inv), ctx);
        return Ok(Symbolic2x2Outcome::Unique {
            values: vec![x, y],
            det_condition: None,
        });
    }

    // Symbolic determinant: absorb its content/sign into the numerators so the
    // displayed quotients and the `det ≠ 0` condition use the tidy primitive
    // form (`a + 1`, never `-2·a - 2`).
    let (scale, det_prim) = normalize_det(&det);
    let inv = BigRational::from_integer(1.into()) / scale;
    let x = quotient_expr(ctx, &x_num.mul_scalar(&inv), &det_prim);
    let y = quotient_expr(ctx, &y_num.mul_scalar(&inv), &det_prim);
    let det_expr = multipoly_to_expr(&det_prim, ctx);
    Ok(Symbolic2x2Outcome::Unique {
        values: vec![x, y],
        det_condition: Some(det_expr),
    })
}

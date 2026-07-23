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

/// Generalized partition: `expr` as `Σ coeff_i·u_i + c` where the `u_i` are
/// the DECLARED unknowns and every coefficient is a polynomial in the
/// remaining (parameter) variables. Any term with total unknown-degree > 1
/// is NotLinear. Single implementation shared by the 2×2 and n×n paths.
fn extract_symbolic_row(
    ctx: &Context,
    expr: ExprId,
    unknowns: &[String],
) -> Result<(Vec<MultiPoly>, MultiPoly), LinearSystemError> {
    let poly = multipoly_from_expr(ctx, expr, &symbolic_budget())
        .map_err(LinearSystemError::PolyConversion)?;
    let unknown_idx: Vec<Option<usize>> = unknowns
        .iter()
        .map(|u| poly.vars.iter().position(|v| v == u))
        .collect();
    let is_unknown_pos = |i: usize| -> bool { unknown_idx.contains(&Some(i)) };

    let param_vars: Vec<String> = poly
        .vars
        .iter()
        .enumerate()
        .filter(|(i, _)| !is_unknown_pos(*i))
        .map(|(_, v)| v.clone())
        .collect();

    let mut coeff_maps: Vec<BTreeMap<Monomial, BigRational>> =
        vec![BTreeMap::new(); unknowns.len()];
    let mut c_map: BTreeMap<Monomial, BigRational> = BTreeMap::new();
    for (coef, mono) in &poly.terms {
        let degs: Vec<u32> = unknown_idx
            .iter()
            .map(|idx| idx.map_or(0, |i| mono[i]))
            .collect();
        let total: u32 = degs.iter().sum();
        let param_mono: Monomial = mono
            .iter()
            .enumerate()
            .filter(|(i, _)| !is_unknown_pos(*i))
            .map(|(_, &e)| e)
            .collect();
        let bucket = match total {
            0 => &mut c_map,
            1 => {
                let which = degs.iter().position(|&d| d == 1).expect("total == 1");
                &mut coeff_maps[which]
            }
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
    let coeffs = coeff_maps.into_iter().map(mk).collect();
    Ok((coeffs, mk_const(&param_vars, c_map)))
}

fn mk_const(param_vars: &[String], map: BTreeMap<Monomial, BigRational>) -> MultiPoly {
    MultiPoly::from_map(param_vars.to_vec(), map)
}

fn extract_symbolic_coeffs(
    ctx: &Context,
    expr: ExprId,
    var_x: &str,
    var_y: &str,
) -> Result<SymCoeffs, LinearSystemError> {
    let unknowns = [var_x.to_string(), var_y.to_string()];
    let (mut coeffs, c) = extract_symbolic_row(ctx, expr, &unknowns)?;
    let b = coeffs.pop().expect("two unknowns");
    let a = coeffs.pop().expect("two unknowns");
    Ok(SymCoeffs { a, b, c })
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

/// Determinant of a square matrix of `MultiPoly` entries by cofactor
/// expansion, budget-checked. Shared by the n×n symbolic Cramer and the
/// Sylvester resultant (S5).
pub(crate) fn poly_determinant(
    matrix: &[Vec<MultiPoly>],
    budget: &PolyBudget,
) -> Option<MultiPoly> {
    let n = matrix.len();
    if n == 0 {
        return None;
    }
    if n == 1 {
        return Some(matrix[0][0].clone());
    }
    let vars = matrix[0][0].vars.clone();
    let mut det = MultiPoly::zero(vars);
    for (col, entry) in matrix[0].iter().enumerate() {
        if entry.is_zero() {
            continue;
        }
        let minor: Vec<Vec<MultiPoly>> = matrix[1..]
            .iter()
            .map(|row| {
                row.iter()
                    .enumerate()
                    .filter(|(j, _)| *j != col)
                    .map(|(_, e)| e.clone())
                    .collect()
            })
            .collect();
        let sub = poly_determinant(&minor, budget)?;
        let term = entry.mul(&sub, budget).ok()?;
        det = if col % 2 == 0 {
            det.add(&term).ok()?
        } else {
            det.sub(&term).ok()?
        };
    }
    Some(det)
}

/// Exact symbolic Cramer for an n×n linear system over parameter-polynomial
/// coefficients (frente S · S6 — wired for n = 3; the cofactor determinant
/// makes larger n a deliberate future step, not an accident). Same contract
/// as the 2×2 path: `det ≠ 0` rides as a structured condition; `det ≡ 0`
/// declines honestly (symbolic rank classification is a future rung).
pub(crate) fn solve_nxn_symbolic(
    ctx: &mut Context,
    exprs: &[ExprId],
    vars: &[String],
) -> Result<Symbolic2x2Outcome, LinearSystemError> {
    let n = vars.len();
    let mut rows = Vec::with_capacity(n);
    for (i, &expr) in exprs.iter().enumerate() {
        let row =
            extract_symbolic_row(ctx, expr, vars).map_err(|e| with_equation_index(e, i + 1))?;
        rows.push(row);
    }

    // Union parameter set + align every entry.
    let mut union: Vec<String> = Vec::new();
    for (coeffs, c) in &rows {
        for poly in coeffs.iter().chain(std::iter::once(c)) {
            for v in &poly.vars {
                if !union.contains(v) {
                    union.push(v.clone());
                }
            }
        }
    }
    union.sort();
    let matrix: Vec<Vec<MultiPoly>> = rows
        .iter()
        .map(|(coeffs, _)| coeffs.iter().map(|p| p.align_vars(&union)).collect())
        .collect();
    let d: Vec<MultiPoly> = rows
        .iter()
        .map(|(_, c)| c.align_vars(&union).neg())
        .collect();

    let budget = symbolic_budget();
    let Some(det) = poly_determinant(&matrix, &budget) else {
        return Err(LinearSystemError::NotLinear(
            "symbolic determinant exceeded the polynomial budget".to_string(),
        ));
    };
    if det.is_zero() {
        return Ok(Symbolic2x2Outcome::DegenerateSymbolic);
    }

    let mut numerators = Vec::with_capacity(n);
    for col in 0..n {
        let replaced: Vec<Vec<MultiPoly>> = matrix
            .iter()
            .enumerate()
            .map(|(r, row)| {
                row.iter()
                    .enumerate()
                    .map(|(c, e)| if c == col { d[r].clone() } else { e.clone() })
                    .collect()
            })
            .collect();
        let Some(num) = poly_determinant(&replaced, &budget) else {
            return Err(LinearSystemError::NotLinear(
                "symbolic determinant exceeded the polynomial budget".to_string(),
            ));
        };
        numerators.push(num);
    }

    if let Some(k) = det.constant_value() {
        let inv = BigRational::from_integer(1.into()) / k;
        let values = numerators
            .iter()
            .map(|num| multipoly_to_expr(&num.mul_scalar(&inv), ctx))
            .collect();
        return Ok(Symbolic2x2Outcome::Unique {
            values,
            det_condition: None,
        });
    }

    let (scale, det_prim) = normalize_det(&det);
    let inv = BigRational::from_integer(1.into()) / scale;
    let values = numerators
        .iter()
        .map(|num| quotient_expr(ctx, &num.mul_scalar(&inv), &det_prim))
        .collect();
    let det_expr = multipoly_to_expr(&det_prim, ctx);
    Ok(Symbolic2x2Outcome::Unique {
        values,
        det_condition: Some(det_expr),
    })
}

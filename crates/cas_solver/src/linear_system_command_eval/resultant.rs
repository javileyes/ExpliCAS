//! Sylvester-resultant elimination for bivariate polynomial 2×2 systems
//! (frente S · S5) — the rung AFTER substitution: when NO equation is linear
//! in an unknown (two conics, `x·y = 6 ∧ x² + y² = 13`), eliminate one
//! variable as the determinant of the Sylvester matrix over `MultiPoly`
//! coefficients, solve the univariate resultant with the mature `solve`
//! machinery, back-substitute per root, and emit ONLY pairs that verify
//! exactly against BOTH original residuals (the same D5 gate as S2).
//!
//! Soundness split:
//! - EMISSION is sound unconditionally (per-pair verification gate).
//! - The "no solution" claim needs completeness of the x-candidate set:
//!   `Res(x0) = 0` at every common point holds whenever some leading
//!   y-coefficient survives at `x0`; where BOTH leading coefficients vanish
//!   we add their real roots as EXTRA candidates, so no common point can
//!   hide. If any candidate could not be enumerated in `y`, we decline
//!   instead of claiming emptiness.
//! - `Res ≡ 0` (common polynomial component) stays an honest decline: over
//!   the reals the shared factor may or may not carry points.

use std::collections::BTreeMap;

use cas_ast::{Equation, ExprId, RelOp, SolutionSet};
use cas_math::multipoly::{
    multipoly_from_expr, multipoly_to_expr, Monomial, MultiPoly, PolyBudget,
};
use num_rational::BigRational;
use num_traits::Zero;

fn resultant_budget() -> PolyBudget {
    PolyBudget {
        max_terms: 400,
        max_total_degree: 24,
        max_pow_exp: 24,
    }
}

/// Narration payload for the resultant path (S3 mould).
pub(crate) struct ResultantNarration {
    pub(crate) eliminated_var: String,
    pub(crate) root_var: String,
    pub(crate) resultant: ExprId,
    pub(crate) sylvester_dim: usize,
}

/// `poly` (over exactly `[x, y]`-style vars) viewed as a polynomial in
/// `var_idx`: coefficient polys over the OTHER variable, indexed by degree.
fn coefficients_in(poly: &MultiPoly, var_idx: usize) -> Vec<MultiPoly> {
    let rest_vars: Vec<String> = poly
        .vars
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != var_idx)
        .map(|(_, v)| v.clone())
        .collect();
    let max_deg = poly
        .terms
        .iter()
        .map(|(_, mono)| mono[var_idx])
        .max()
        .unwrap_or(0) as usize;
    let mut maps: Vec<BTreeMap<Monomial, BigRational>> = vec![BTreeMap::new(); max_deg + 1];
    for (coef, mono) in &poly.terms {
        let d = mono[var_idx] as usize;
        let rest_mono: Monomial = mono
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != var_idx)
            .map(|(_, &e)| e)
            .collect();
        let entry = maps[d].entry(rest_mono).or_insert_with(BigRational::zero);
        *entry = &*entry + coef;
    }
    maps.into_iter()
        .map(|m| MultiPoly::from_map(rest_vars.clone(), m))
        .collect()
}

/// Determinant of a square matrix of `MultiPoly` entries by cofactor
/// expansion (Sylvester dims for curricular conics are ≤ 6). Budget-checked
/// multiplication; `None` on budget blowup.
fn poly_determinant(matrix: &[Vec<MultiPoly>], budget: &PolyBudget) -> Option<MultiPoly> {
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

/// Sylvester matrix of `f` (degree m) and `g` (degree n) as polynomials in
/// the eliminated variable, coefficients over the other variable. Rows are
/// the classic shifted coefficient vectors: n rows of `f`, m rows of `g`.
fn sylvester_matrix(f_coeffs: &[MultiPoly], g_coeffs: &[MultiPoly]) -> Vec<Vec<MultiPoly>> {
    let m = f_coeffs.len() - 1;
    let n = g_coeffs.len() - 1;
    let dim = m + n;
    let vars = f_coeffs[0].vars.clone();
    let zero = MultiPoly::zero(vars);
    let mut matrix = vec![vec![zero; dim]; dim];
    // Coefficient vectors highest-degree first.
    for shift in 0..n {
        for (k, c) in f_coeffs.iter().rev().enumerate() {
            matrix[shift][shift + k] = c.clone();
        }
    }
    for shift in 0..m {
        for (k, c) in g_coeffs.iter().rev().enumerate() {
            matrix[n + shift][shift + k] = c.clone();
        }
    }
    matrix
}

fn zero_test(simplifier: &mut crate::Simplifier, expr: ExprId) -> bool {
    let (folded, _) = simplifier.simplify(expr);
    matches!(
        simplifier.context.get(folded),
        cas_ast::Expr::Number(n) if n.is_zero()
    )
}

/// Univariate `Discrete` roots of `expr = 0` in `var`; `None` when the solve
/// outcome is not a finite enumerable set.
fn discrete_roots(
    simplifier: &mut crate::Simplifier,
    expr: ExprId,
    var: &str,
) -> Option<Vec<ExprId>> {
    let zero = simplifier.context.num(0);
    let eq = Equation {
        lhs: expr,
        rhs: zero,
        op: RelOp::Eq,
    };
    match crate::api::solve(&eq, var, simplifier) {
        Ok((SolutionSet::Discrete(roots), _)) => Some(roots),
        Ok((SolutionSet::Empty, _)) => Some(Vec::new()),
        _ => None,
    }
}

/// Try the resultant path. Caller guarantees: 2 equations, 2 unknowns,
/// parameter-free, and the substitution path already declined.
pub(crate) fn try_solve_resultant_2x2(
    simplifier: &mut crate::Simplifier,
    exprs: &[ExprId],
    vars: &[String],
) -> Option<(crate::LinSolveResult, ResultantNarration)> {
    let budget = resultant_budget();
    // Deterministic: eliminate vars[1], roots in vars[0]; if that shape is
    // unavailable (an equation lacks the eliminated variable), swap.
    for (elim_idx, root_idx) in [(1usize, 0usize), (0, 1)] {
        let elim_var = &vars[elim_idx];
        let root_var = &vars[root_idx];

        let mut coeff_views = Vec::with_capacity(2);
        let mut ok = true;
        for &expr in exprs {
            let Ok(poly) = multipoly_from_expr(&simplifier.context, expr, &budget) else {
                return None;
            };
            // Align over BOTH unknowns so var indices are stable.
            let mut sorted_vars = vars.to_vec();
            sorted_vars.sort();
            let aligned = poly.align_vars(&sorted_vars);
            let Some(idx) = aligned.vars.iter().position(|v| v == elim_var) else {
                ok = false;
                break;
            };
            let coeffs = coefficients_in(&aligned, idx);
            if coeffs.len() < 2 {
                // Degree 0 in the eliminated variable: this orientation does
                // not eliminate anything — try the swapped one.
                ok = false;
                break;
            }
            coeff_views.push(coeffs);
        }
        if !ok {
            continue;
        }

        let f_coeffs = &coeff_views[0];
        let g_coeffs = &coeff_views[1];
        let matrix = sylvester_matrix(f_coeffs, g_coeffs);
        let dim = matrix.len();
        let res_poly = poly_determinant(&matrix, &budget)?;
        if res_poly.is_zero() {
            // Common polynomial component: real-point structure undecided
            // here — honest decline (future rung).
            return None;
        }

        // x-candidates: resultant roots + real roots of BOTH leading
        // coefficients where they could vanish together (completeness guard
        // for the emptiness claim). A NONZERO CONSTANT resultant has no
        // roots at all (concentric circles): skip the solver, candidates = ∅.
        let res_expr = multipoly_to_expr(&res_poly, &mut simplifier.context);
        let mut candidates = if res_poly.is_constant() {
            Vec::new()
        } else {
            discrete_roots(simplifier, res_expr, root_var)?
        };
        let mut enumeration_complete = true;
        let f_lead = f_coeffs.last().expect("nonempty");
        let g_lead = g_coeffs.last().expect("nonempty");
        if !f_lead.is_constant() && !g_lead.is_constant() {
            for lead in [f_lead.clone(), g_lead.clone()] {
                let lead_expr = multipoly_to_expr(&lead, &mut simplifier.context);
                match discrete_roots(simplifier, lead_expr, root_var) {
                    Some(extra) => candidates.extend(extra),
                    None => enumeration_complete = false,
                }
            }
        }

        let narration = ResultantNarration {
            eliminated_var: elim_var.clone(),
            root_var: root_var.clone(),
            resultant: res_expr,
            sylvester_dim: dim,
        };

        let root_sym = simplifier.context.var(root_var);
        let elim_sym = simplifier.context.var(elim_var);
        let mut pairs: Vec<Vec<ExprId>> = Vec::new();
        let mut seen: Vec<(ExprId, ExprId)> = Vec::new();
        for &x0_raw in &candidates {
            let (x0, _) = simplifier.simplify(x0_raw);
            // Back-substitute: specialize each equation at x0; the first one
            // that stays enumerable in the eliminated variable provides the
            // y-candidates (the verification gate below makes the choice of
            // source irrelevant for soundness).
            let mut y_candidates: Option<Vec<ExprId>> = None;
            for &expr in exprs {
                let specialized = crate::substitute::substitute_power_aware(
                    &mut simplifier.context,
                    expr,
                    root_sym,
                    x0,
                    crate::substitute::SubstituteOptions::exact(),
                );
                if let Some(roots) = discrete_roots(simplifier, specialized, elim_var) {
                    y_candidates = Some(roots);
                    break;
                }
            }
            let Some(y_candidates) = y_candidates else {
                enumeration_complete = false;
                continue;
            };
            for &y0_raw in &y_candidates {
                let (y0, _) = simplifier.simplify(y0_raw);
                let mut verified = true;
                for &residual in exprs {
                    let sub_x = crate::substitute::substitute_power_aware(
                        &mut simplifier.context,
                        residual,
                        root_sym,
                        x0,
                        crate::substitute::SubstituteOptions::exact(),
                    );
                    let sub_xy = crate::substitute::substitute_power_aware(
                        &mut simplifier.context,
                        sub_x,
                        elim_sym,
                        y0,
                        crate::substitute::SubstituteOptions::exact(),
                    );
                    if !zero_test(simplifier, sub_xy) {
                        verified = false;
                        break;
                    }
                }
                if !verified {
                    continue;
                }
                if seen.contains(&(x0, y0)) {
                    continue;
                }
                seen.push((x0, y0));
                let mut pair = vec![x0; 2];
                pair[root_idx] = x0;
                pair[elim_idx] = y0;
                pairs.push(pair);
            }
        }

        if pairs.is_empty() {
            if enumeration_complete {
                // Every x-candidate enumerated and verified empty: the system
                // has NO real solutions (resultant completeness + gate).
                return Some((crate::LinSolveResult::Inconsistent, narration));
            }
            return None;
        }
        return Some((crate::LinSolveResult::SolutionPairs(pairs), narration));
    }
    None
}

//! Nonlinear 2×2 systems by VERIFIED substitution (frente S · S2).
//!
//! Composition over invention: isolate an unknown from an equation that is
//! linear in it (constant coefficient), substitute into the sibling equation,
//! delegate the univariate polynomial to the mature `solve` machinery, then
//! back-substitute — and emit ONLY the pairs that verify exactly against BOTH
//! original residuals (the dsolve D5 doctrine transferred: verification by
//! substitution subsumes the domain conditions).
//!
//! Scope guard (declared intent, protects the footprint): parameter-free
//! systems only — every variable must be one of the two unknowns. Parametric
//! nonlinear systems stay honest residuals for a future rung.

use std::collections::BTreeMap;

use cas_ast::{Context, Equation, Expr, ExprId, RelOp, SolutionSet};
use cas_math::multipoly::{
    multipoly_from_expr, multipoly_to_expr, Monomial, MultiPoly, PolyBudget,
};
use num_rational::BigRational;
use num_traits::Zero;

fn nonlinear_budget() -> PolyBudget {
    PolyBudget {
        max_terms: 200,
        max_total_degree: 16,
        max_pow_exp: 16,
    }
}

/// `expr` viewed as `A·u + C` where `A` is a nonzero RATIONAL constant and `C`
/// is a polynomial in the remaining variables. Returns the isolation
/// `u = -C/A` as an expression. Constant-only coefficients keep the division
/// total (a variable coefficient could vanish on the solution set).
fn isolate_linear_unknown(ctx: &mut Context, expr: ExprId, unknown: &str) -> Option<ExprId> {
    let poly = multipoly_from_expr(ctx, expr, &nonlinear_budget()).ok()?;
    let idx_u = poly.vars.iter().position(|v| v == unknown)?;

    let rest_vars: Vec<String> = poly
        .vars
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != idx_u)
        .map(|(_, v)| v.clone())
        .collect();

    let mut a_const = BigRational::zero();
    let mut c_map: BTreeMap<Monomial, BigRational> = BTreeMap::new();
    for (coef, mono) in &poly.terms {
        let eu = mono[idx_u];
        let rest_mono: Monomial = mono
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != idx_u)
            .map(|(_, &e)| e)
            .collect();
        match eu {
            0 => {
                let entry = c_map.entry(rest_mono).or_insert_with(BigRational::zero);
                *entry = &*entry + coef;
            }
            1 => {
                if rest_mono.iter().any(|&e| e != 0) {
                    // Variable coefficient on the unknown: not a safe isolation.
                    return None;
                }
                a_const = &a_const + coef;
            }
            _ => return None,
        }
    }
    if a_const.is_zero() {
        return None;
    }
    let c_poly = MultiPoly::from_map(rest_vars, c_map);
    let scaled = c_poly.mul_scalar(&(-BigRational::from_integer(1.into()) / a_const));
    Some(multipoly_to_expr(&scaled, ctx))
}

/// Exact zero test after a full simplify pass — the same gate shape dsolve
/// uses: emission happens only on a literal `0`.
fn reduces_to_zero(simplifier: &mut crate::Simplifier, expr: ExprId) -> bool {
    let (folded, _) = simplifier.simplify(expr);
    matches!(
        simplifier.context.get(folded),
        Expr::Number(n) if n.is_zero()
    )
}

/// Narration payload for the S3/S5 educational half — which route solved the
/// nonlinear system.
pub(crate) enum SystemNarration {
    Substitution(NonlinearNarration),
    Resultant(super::resultant::ResultantNarration),
}

/// Narration payload for the S3 educational half: what was isolated, what
/// univariate equation resulted, and how many pairs survived verification.
pub(crate) struct NonlinearNarration {
    pub(crate) isolated_var: String,
    pub(crate) isolation: ExprId,
    pub(crate) source_index: usize,
    pub(crate) free_var: String,
    pub(crate) univariate: ExprId,
}

/// Try to solve a parameter-free nonlinear 2×2 system by verified
/// substitution. `exprs` are the normalized residuals (`lhs - rhs`).
/// Returns `None` when out of scope — the caller keeps its honest decline.
///
/// Steps ritual: the internal machinery (univariate solve, verification
/// simplifies) must NOT leak micro-steps to a session step listener — the
/// user-facing narration is built separately (S3). Save/off/restore both the
/// listener and the steps mode around the whole body, on every return path.
pub(crate) fn try_solve_nonlinear_2x2(
    simplifier: &mut crate::Simplifier,
    exprs: &[ExprId],
    vars: &[String],
) -> Option<(crate::LinSolveResult, Option<SystemNarration>)> {
    let previous_listener = simplifier.replace_step_listener(None);
    let previous_mode = simplifier.steps_mode;
    simplifier.steps_mode = cas_engine::StepsMode::Off;
    let result = try_solve_nonlinear_2x2_inner(simplifier, exprs, vars);
    simplifier.steps_mode = previous_mode;
    simplifier.set_step_listener(previous_listener);
    result
}

fn try_solve_nonlinear_2x2_inner(
    simplifier: &mut crate::Simplifier,
    exprs: &[ExprId],
    vars: &[String],
) -> Option<(crate::LinSolveResult, Option<SystemNarration>)> {
    if exprs.len() != 2 || vars.len() != 2 {
        return None;
    }
    // Parameter-free guard: every variable in the system is an unknown.
    for &expr in exprs {
        for name in cas_ast::collect_variables(&simplifier.context, expr) {
            if !vars.contains(&name) {
                return None;
            }
        }
    }

    // Deterministic isolation search: each equation × each unknown.
    for (src, other) in [(0usize, 1usize), (1, 0)] {
        for (u_idx, w_idx) in [(1usize, 0usize), (0, 1)] {
            let unknown = &vars[u_idx];
            let free_var = &vars[w_idx];
            let Some(iso) = isolate_linear_unknown(&mut simplifier.context, exprs[src], unknown)
            else {
                continue;
            };
            let u_var = simplifier.context.var(unknown);
            let substituted = crate::substitute::substitute_power_aware(
                &mut simplifier.context,
                exprs[other],
                u_var,
                iso,
                crate::substitute::SubstituteOptions::exact(),
            );
            let zero = simplifier.context.num(0);
            let univariate = Equation {
                lhs: substituted,
                rhs: zero,
                op: RelOp::Eq,
            };
            let Ok((solution_set, _steps)) = crate::api::solve(&univariate, free_var, simplifier)
            else {
                continue;
            };
            let narration = NonlinearNarration {
                isolated_var: unknown.clone(),
                isolation: iso,
                source_index: src,
                free_var: free_var.clone(),
                univariate: substituted,
            };
            let roots = match solution_set {
                SolutionSet::Discrete(roots) => roots,
                // A PROVEN empty set over the reals is an honest "no solution"
                // (circle and a line that misses it).
                SolutionSet::Empty => {
                    return Some((
                        crate::LinSolveResult::Inconsistent,
                        Some(SystemNarration::Substitution(narration)),
                    ))
                }
                _ => continue,
            };
            if roots.is_empty() {
                return Some((
                    crate::LinSolveResult::Inconsistent,
                    Some(SystemNarration::Substitution(narration)),
                ));
            }

            let w_var = simplifier.context.var(free_var);
            let mut pairs: Vec<Vec<ExprId>> = Vec::new();
            for &root in &roots {
                let u_val_raw = crate::substitute::substitute_power_aware(
                    &mut simplifier.context,
                    iso,
                    w_var,
                    root,
                    crate::substitute::SubstituteOptions::exact(),
                );
                let (u_val, _) = simplifier.simplify(u_val_raw);
                let (w_val, _) = simplifier.simplify(root);

                // D5 transferred: the pair must annihilate BOTH original
                // residuals exactly, or it is not emitted.
                let mut verified = true;
                for &residual in exprs {
                    let subbed_u = crate::substitute::substitute_power_aware(
                        &mut simplifier.context,
                        residual,
                        u_var,
                        u_val,
                        crate::substitute::SubstituteOptions::exact(),
                    );
                    let subbed = crate::substitute::substitute_power_aware(
                        &mut simplifier.context,
                        subbed_u,
                        w_var,
                        w_val,
                        crate::substitute::SubstituteOptions::exact(),
                    );
                    if !reduces_to_zero(simplifier, subbed) {
                        verified = false;
                        break;
                    }
                }
                if !verified {
                    continue;
                }
                let mut pair = vec![zero; 2];
                pair[u_idx] = u_val;
                pair[w_idx] = w_val;
                pairs.push(pair);
            }
            if pairs.is_empty() {
                // Roots existed but none verified: do not assert emptiness —
                // decline honestly (verification is the emission gate, not an
                // emptiness proof).
                return None;
            }
            return Some((
                crate::LinSolveResult::SolutionPairs(pairs),
                Some(SystemNarration::Substitution(narration)),
            ));
        }
    }
    // S5: no isolatable equation — the Sylvester-resultant rung (two conics,
    // x·y = 6 ∧ x² + y² = 13). Same verification gate; own narration.
    if let Some((result, narration)) =
        super::resultant::try_solve_resultant_2x2(simplifier, exprs, vars)
    {
        return Some((result, Some(SystemNarration::Resultant(narration))));
    }
    None
}

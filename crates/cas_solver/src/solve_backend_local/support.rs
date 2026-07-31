//! `solve_backend_local`: familia `support`.
//!
//! Ver la cabecera de `solve_backend_local.rs` para el contexto.

use super::*;

/// True if `expr` contains the variable named `var`.
pub(super) fn expr_contains_named_var(ctx: &Context, expr: ExprId, var: &str) -> bool {
    cas_ast::collect_variables(ctx, expr)
        .iter()
        .any(|s| s == var)
}

/// Flip a strict/non-strict inequality operator (for multiplying by −1).
pub(super) fn flip_inequality(op: cas_ast::RelOp) -> cas_ast::RelOp {
    use cas_ast::RelOp;
    match op {
        RelOp::Lt => RelOp::Gt,
        RelOp::Gt => RelOp::Lt,
        RelOp::Leq => RelOp::Geq,
        RelOp::Geq => RelOp::Leq,
        other => other,
    }
}

/// Solve `lhs {op} rhs` recursively and return just the solution set.
pub(super) fn solve_relation_set(
    simplifier: &mut Simplifier,
    var: &str,
    lhs: ExprId,
    rhs: ExprId,
    op: cas_ast::RelOp,
) -> Option<SolutionSet> {
    let eq = Equation { lhs, rhs, op };
    crate::solver_entrypoints_solve::solve(&eq, var, simplifier)
        .ok()
        .map(|(s, _)| s)
}

pub(super) fn solve_polynomial_in_atom(
    simplifier: &mut Simplifier,
    u_expr: ExprId,
    u_var: &str,
    var: &str,
    back_sub_atom: ExprId,
    steps_out: &mut Vec<crate::SolveStep>,
) -> Option<SolutionSet> {
    use cas_math::polynomial::Polynomial;
    let u_poly = Polynomial::from_expr(&simplifier.context, u_expr, u_var).ok()?;
    if u_poly.degree() < 2 {
        return None;
    }
    let zero = simplifier.context.num(0);
    let u_eq = Equation {
        lhs: u_expr,
        rhs: zero,
        op: cas_ast::RelOp::Eq,
    };
    let (u_solution, u_steps) =
        crate::solver_entrypoints_solve::solve(&u_eq, u_var, simplifier).ok()?;
    let u_roots = match u_solution {
        SolutionSet::Discrete(roots) => roots,
        SolutionSet::Empty => return Some(SolutionSet::Empty),
        _ => return None, // non-discrete / unsolved u-polynomial: leave to the existing path
    };
    // Narration: detection, then HOW the u-roots were obtained, then one
    // back-substitution per root. The internal `u_var` is a collision-safe
    // synthetic name (`__trig_u`, `__rps_u`, …) the student never wrote, so
    // every republished line is rewritten to the display `u` first.
    //
    // Until 2026-07-28 the u-polynomial's own steps were dropped on the floor
    // (`let (u_solution, _) = solve(...)`) with the reason «the exp route
    // shows neither». The exp route now DOES show them, and dropping them left
    // the reader watching `u = 1` and `u = 2` appear from nowhere — the very
    // gap the user reported.
    let atom_display = format!(
        "{}",
        cas_formatter::DisplayExpr {
            context: &simplifier.context,
            id: back_sub_atom
        }
    );
    let display_u = simplifier.context.var("u");
    let internal_u = simplifier.context.var(u_var);
    let display_u_expr =
        substitute_expr_by_id(&mut simplifier.context, u_expr, internal_u, display_u);
    // Strip identity-zero noise (`u^2 - 3u + 2 - 0`) without a general simplify
    // (which would factor/reorder the polynomial the student should recognize).
    let display_u_expr = cas_solver_core::eval_step_pipeline::normalize_expr_for_display(
        &mut simplifier.context,
        display_u_expr,
    );
    let display_zero = simplifier.context.num(0);
    steps_out.push(crate::SolveStep::new(
        format!("Detected substitution: u = {}", atom_display),
        Equation {
            lhs: display_u_expr,
            rhs: display_zero,
            op: cas_ast::RelOp::Eq,
        },
        crate::ImportanceLevel::Medium,
    ));
    steps_out.extend(rewrite_substitution_steps_for_display(
        simplifier, u_steps, u_var,
    ));
    // Back-substitute each root `atom = u_root`, then union the branch solutions (periodic-aware).
    let mut branch_sets = Vec::with_capacity(u_roots.len());
    for u_root in u_roots {
        let back_eq = Equation {
            lhs: back_sub_atom,
            rhs: u_root,
            op: cas_ast::RelOp::Eq,
        };
        let root_display = format!(
            "{}",
            cas_formatter::DisplayExpr {
                context: &simplifier.context,
                id: u_root
            }
        );
        steps_out.push(crate::SolveStep::new(
            format!("Back-substitute: {} = {}", atom_display, root_display),
            back_eq.clone(),
            crate::ImportanceLevel::Medium,
        ));
        let (xs, branch_steps) =
            crate::solver_entrypoints_solve::solve(&back_eq, var, simplifier).ok()?;
        steps_out.extend(branch_steps);
        branch_sets.push(xs);
    }
    union_branch_solutions(simplifier, branch_sets)
}

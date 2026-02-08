//! Solver utility functions: verification, substitution, domain guards.

use cas_ast::{Context, ExprId, SolutionSet};

use crate::build::mul2_raw;
use crate::engine::Simplifier;
use crate::error::CasError;
use crate::solver::isolation::contains_var;
use crate::solver::SolveStep;

/// Verify a candidate solution by substitution into the original equation.
pub(crate) fn verify_solution(
    eq: &cas_ast::Equation,
    var: &str,
    sol: ExprId,
    simplifier: &mut Simplifier,
) -> bool {
    // 1. Substitute
    let lhs_sub = substitute(&mut simplifier.context, eq.lhs, var, sol);
    let rhs_sub = substitute(&mut simplifier.context, eq.rhs, var, sol);

    // 2. Simplify
    let (lhs_sim, _) = simplifier.simplify(lhs_sub);
    let (rhs_sim, _) = simplifier.simplify(rhs_sub);

    // 3. Check equality
    simplifier.are_equivalent(lhs_sim, rhs_sim)
}

/// Check if an expression is "symbolic" (contains functions or variables).
/// Symbolic expressions cannot be verified by substitution because they don't
/// simplify to pure numbers. Examples: ln(c/d)/ln(a/b), x + a, sqrt(y)
pub(crate) fn is_symbolic_expr(ctx: &Context, expr: ExprId) -> bool {
    use cas_ast::Expr;
    match ctx.get(expr) {
        Expr::Number(_) => false,
        Expr::Constant(_) => true, // Pi, E, etc are symbolic
        Expr::Variable(_) => true,
        Expr::Function(_, _) => true,
        Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) | Expr::Div(l, r) | Expr::Pow(l, r) => {
            is_symbolic_expr(ctx, *l) || is_symbolic_expr(ctx, *r)
        }
        Expr::Neg(e) | Expr::Hold(e) => is_symbolic_expr(ctx, *e),
        Expr::Matrix { data, .. } => data.iter().any(|d| is_symbolic_expr(ctx, *d)),
        Expr::SessionRef(_) => true, // Unresolved reference — treat as symbolic
    }
}

/// Substitute a variable with a value expression throughout the AST.
pub(crate) fn substitute(ctx: &mut Context, expr: ExprId, var: &str, val: ExprId) -> ExprId {
    use cas_ast::Expr;
    let expr_data = ctx.get(expr).clone();
    match expr_data {
        Expr::Variable(sym_id) if ctx.sym_name(sym_id) == var => val,
        Expr::Add(l, r) => {
            let nl = substitute(ctx, l, var, val);
            let nr = substitute(ctx, r, var, val);
            if nl != l || nr != r {
                ctx.add(Expr::Add(nl, nr))
            } else {
                expr
            }
        }
        Expr::Sub(l, r) => {
            let nl = substitute(ctx, l, var, val);
            let nr = substitute(ctx, r, var, val);
            if nl != l || nr != r {
                ctx.add(Expr::Sub(nl, nr))
            } else {
                expr
            }
        }
        Expr::Mul(l, r) => {
            let nl = substitute(ctx, l, var, val);
            let nr = substitute(ctx, r, var, val);
            if nl != l || nr != r {
                mul2_raw(ctx, nl, nr)
            } else {
                expr
            }
        }
        Expr::Div(l, r) => {
            let nl = substitute(ctx, l, var, val);
            let nr = substitute(ctx, r, var, val);
            if nl != l || nr != r {
                ctx.add(Expr::Div(nl, nr))
            } else {
                expr
            }
        }
        Expr::Pow(b, e) => {
            let nb = substitute(ctx, b, var, val);
            let ne = substitute(ctx, e, var, val);
            if nb != b || ne != e {
                ctx.add(Expr::Pow(nb, ne))
            } else {
                expr
            }
        }
        Expr::Neg(e) => {
            let ne = substitute(ctx, e, var, val);
            if ne != e {
                ctx.add(Expr::Neg(ne))
            } else {
                expr
            }
        }
        Expr::Function(name, args) => {
            let mut new_args = Vec::new();
            let mut changed = false;
            for arg in args {
                let new_arg = substitute(ctx, arg, var, val);
                if new_arg != arg {
                    changed = true;
                }
                new_args.push(new_arg);
            }
            if changed {
                ctx.add(Expr::Function(name, new_args))
            } else {
                expr
            }
        }
        Expr::Hold(inner) => {
            let new_inner = substitute(ctx, inner, var, val);
            if new_inner != inner {
                ctx.add(Expr::Hold(new_inner))
            } else {
                expr
            }
        }
        Expr::Matrix { rows, cols, data } => {
            let mut new_data = Vec::new();
            let mut changed = false;
            for elem in data {
                let new_elem = substitute(ctx, elem, var, val);
                if new_elem != elem {
                    changed = true;
                }
                new_data.push(new_elem);
            }
            if changed {
                ctx.add(Expr::Matrix {
                    rows,
                    cols,
                    data: new_data,
                })
            } else {
                expr
            }
        }
        // Leaves — no children to substitute into
        Expr::Number(_) | Expr::Constant(_) | Expr::Variable(_) | Expr::SessionRef(_) => expr,
    }
}

/// V2.1 Issue #10: Extract all denominators from an expression that contain the given variable.
///
/// Used to detect domain restrictions when solving equations with fractions.
/// Returns a list of ExprIds that appear as denominators and contain the variable.
///
/// Example: `(x*y)/x` returns `[x]` (the denominator x contains var "x")
pub(crate) fn extract_denominators_with_var(ctx: &Context, expr: ExprId, var: &str) -> Vec<ExprId> {
    use std::collections::HashSet;
    let mut denoms_set: HashSet<ExprId> = HashSet::new();
    collect_denominators_into_set(ctx, expr, var, &mut denoms_set);
    denoms_set.into_iter().collect()
}

/// Helper to recursively collect denominators into a HashSet
fn collect_denominators_into_set(
    ctx: &Context,
    expr: ExprId,
    var: &str,
    denoms: &mut std::collections::HashSet<ExprId>,
) {
    use cas_ast::Expr;
    match ctx.get(expr) {
        Expr::Div(num, denom) => {
            // Check if denominator contains the variable
            if contains_var(ctx, *denom, var) {
                denoms.insert(*denom);
            }
            // Also check for nested divisions in numerator and denominator
            collect_denominators_into_set(ctx, *num, var, denoms);
            collect_denominators_into_set(ctx, *denom, var, denoms);
        }
        Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) | Expr::Pow(l, r) => {
            collect_denominators_into_set(ctx, *l, var, denoms);
            collect_denominators_into_set(ctx, *r, var, denoms);
        }
        Expr::Neg(e) | Expr::Hold(e) => {
            collect_denominators_into_set(ctx, *e, var, denoms);
        }
        Expr::Function(_, args) => {
            for arg in args {
                collect_denominators_into_set(ctx, *arg, var, denoms);
            }
        }
        Expr::Matrix { data, .. } => {
            for elem in data {
                collect_denominators_into_set(ctx, *elem, var, denoms);
            }
        }
        // Leaves — no children to collect from
        Expr::Number(_) | Expr::Constant(_) | Expr::Variable(_) | Expr::SessionRef(_) => {}
    }
}

/// V2.1 Issue #10: Wrap a solve result with domain guards for denominators.
///
/// If there are domain exclusions (denominators that must be non-zero),
/// this wraps the result in a Conditional with NonZero guards.
pub(crate) fn wrap_with_domain_guards(
    result: Result<(SolutionSet, Vec<SolveStep>), CasError>,
    exclusions: &[ExprId],
    _simplifier: &mut Simplifier,
) -> Result<(SolutionSet, Vec<SolveStep>), CasError> {
    // If no exclusions, return as-is
    if exclusions.is_empty() {
        return result;
    }

    let (solution_set, steps) = result?;

    // Build the NonZero guard condition set
    let mut guard = cas_ast::ConditionSet::empty();
    for &denom in exclusions {
        guard.push(cas_ast::ConditionPredicate::NonZero(denom));
    }

    // Wrap in Conditional: [guard -> solution, otherwise -> Empty (undefined)]
    let cases = vec![
        cas_ast::Case::new(guard, solution_set),
        cas_ast::Case::new(cas_ast::ConditionSet::empty(), SolutionSet::Empty),
    ];

    Ok((SolutionSet::Conditional(cases).simplify(), steps))
}

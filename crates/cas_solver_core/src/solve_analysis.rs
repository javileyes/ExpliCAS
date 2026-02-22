use cas_ast::{Case, ConditionPredicate, ConditionSet, Context, Expr, ExprId, SolutionSet};
use std::collections::HashSet;

/// Check if an expression is symbolic (contains variables/functions/constants).
pub fn is_symbolic_expr(ctx: &Context, expr: ExprId) -> bool {
    match ctx.get(expr) {
        Expr::Number(_) => false,
        Expr::Constant(_) => true,
        Expr::Variable(_) => true,
        Expr::Function(_, _) => true,
        Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) | Expr::Div(l, r) | Expr::Pow(l, r) => {
            is_symbolic_expr(ctx, *l) || is_symbolic_expr(ctx, *r)
        }
        Expr::Neg(e) | Expr::Hold(e) => is_symbolic_expr(ctx, *e),
        Expr::Matrix { data, .. } => data.iter().any(|d| is_symbolic_expr(ctx, *d)),
        Expr::SessionRef(_) => true,
    }
}

/// Split discrete solutions into `(symbolic, non_symbolic)` buckets.
pub fn partition_discrete_symbolic(ctx: &Context, sols: &[ExprId]) -> (Vec<ExprId>, Vec<ExprId>) {
    let mut symbolic = Vec::new();
    let mut non_symbolic = Vec::new();
    for &sol in sols {
        if is_symbolic_expr(ctx, sol) {
            symbolic.push(sol);
        } else {
            non_symbolic.push(sol);
        }
    }
    (symbolic, non_symbolic)
}

/// Keep only solutions accepted by a verifier callback.
pub fn retain_verified_discrete<F>(sols: Vec<ExprId>, mut verify: F) -> Vec<ExprId>
where
    F: FnMut(ExprId) -> bool,
{
    let mut out = Vec::new();
    for sol in sols {
        if verify(sol) {
            out.push(sol);
        }
    }
    out
}

/// Merge symbolic roots with verified numeric roots.
///
/// Returns `symbolic ++ verified_numeric` preserving solver behavior.
pub fn merge_symbolic_with_verified_numeric<F>(
    mut symbolic_solutions: Vec<ExprId>,
    numeric_solutions: Vec<ExprId>,
    verify_numeric: F,
) -> Vec<ExprId>
where
    F: FnMut(ExprId) -> bool,
{
    let verified_numeric = retain_verified_discrete(numeric_solutions, verify_numeric);
    symbolic_solutions.extend(verified_numeric);
    symbolic_solutions
}

/// Decide whether a rewritten residual should replace the current one.
///
/// Accept when:
/// - The target variable was eliminated, or
/// - Tree size was reduced by more than 25% (avoids cosmetic rewrites).
pub fn should_accept_rewritten_residual(
    var_eliminated: bool,
    old_nodes: usize,
    new_nodes: usize,
) -> bool {
    var_eliminated || (old_nodes > 4 && new_nodes * 4 < old_nodes * 3)
}

/// Extract all denominators that contain the target variable.
pub fn extract_denominators_with_var(ctx: &Context, expr: ExprId, var: &str) -> Vec<ExprId> {
    let mut denoms_set: HashSet<ExprId> = HashSet::new();
    collect_denominators_into_set(ctx, expr, var, &mut denoms_set);
    denoms_set.into_iter().collect()
}

/// Collect unique denominator expressions containing `var` across equation sides.
pub fn collect_unique_denominators_with_var(
    ctx: &Context,
    lhs: ExprId,
    rhs: ExprId,
    var: &str,
) -> Vec<ExprId> {
    let mut denoms_set: HashSet<ExprId> = HashSet::new();
    denoms_set.extend(extract_denominators_with_var(ctx, lhs, var));
    denoms_set.extend(extract_denominators_with_var(ctx, rhs, var));
    denoms_set.into_iter().collect()
}

fn collect_denominators_into_set(
    ctx: &Context,
    expr: ExprId,
    var: &str,
    denoms: &mut HashSet<ExprId>,
) {
    match ctx.get(expr) {
        Expr::Div(num, denom) => {
            if super::isolation_utils::contains_var(ctx, *denom, var) {
                denoms.insert(*denom);
            }
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
        Expr::Number(_) | Expr::Constant(_) | Expr::Variable(_) | Expr::SessionRef(_) => {}
    }
}

/// Apply non-zero exclusion guards to a solution set.
pub fn apply_nonzero_exclusion_guards(
    solution_set: SolutionSet,
    exclusions: &[ExprId],
) -> SolutionSet {
    if exclusions.is_empty() {
        return solution_set;
    }

    let mut guard = ConditionSet::empty();
    for &denom in exclusions {
        guard.push(ConditionPredicate::NonZero(denom));
    }

    let cases = vec![
        Case::new(guard, solution_set),
        Case::new(ConditionSet::empty(), SolutionSet::Empty),
    ];
    SolutionSet::Conditional(cases).simplify()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn symbolic_number_vs_variable() {
        let mut ctx = Context::new();
        let two = ctx.num(2);
        let x = ctx.var("x");
        assert!(!is_symbolic_expr(&ctx, two));
        assert!(is_symbolic_expr(&ctx, x));
    }

    #[test]
    fn extract_denominators_basic() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let div = ctx.add(Expr::Div(y, x));
        let denoms = extract_denominators_with_var(&ctx, div, "x");
        assert_eq!(denoms.len(), 1);
        assert_eq!(denoms[0], x);
    }

    #[test]
    fn apply_guards_builds_conditional() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let sol = SolutionSet::Discrete(vec![ctx.num(1)]);
        let guarded = apply_nonzero_exclusion_guards(sol, &[x]);
        assert!(matches!(guarded, SolutionSet::Conditional(_)));
    }

    #[test]
    fn collect_unique_denominators_deduplicates_between_sides() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let one = ctx.num(1);
        let lhs = ctx.add(Expr::Div(one, x));
        let one2 = ctx.num(1);
        let rhs = ctx.add(Expr::Div(one2, x));
        let denoms = collect_unique_denominators_with_var(&ctx, lhs, rhs, "x");
        assert_eq!(denoms.len(), 1);
    }

    #[test]
    fn partition_discrete_symbolic_splits_expected() {
        let mut ctx = Context::new();
        let two = ctx.num(2);
        let x = ctx.var("x");
        let (symbolic, non_symbolic) = partition_discrete_symbolic(&ctx, &[two, x]);
        assert_eq!(symbolic, vec![x]);
        assert_eq!(non_symbolic, vec![two]);
    }

    #[test]
    fn retain_verified_discrete_keeps_only_verified() {
        let sols = vec![
            cas_ast::ExprId::from_raw(1),
            cas_ast::ExprId::from_raw(2),
            cas_ast::ExprId::from_raw(3),
        ];
        let kept = retain_verified_discrete(sols, |id| id.index() % 2 == 1);
        assert_eq!(
            kept,
            vec![cas_ast::ExprId::from_raw(1), cas_ast::ExprId::from_raw(3)]
        );
    }

    #[test]
    fn accept_rewritten_residual_when_variable_eliminated() {
        assert!(should_accept_rewritten_residual(true, 100, 99));
    }

    #[test]
    fn accept_rewritten_residual_on_significant_reduction() {
        assert!(should_accept_rewritten_residual(false, 20, 14));
    }

    #[test]
    fn reject_rewritten_residual_on_cosmetic_change() {
        assert!(!should_accept_rewritten_residual(false, 20, 19));
    }

    #[test]
    fn merge_symbolic_with_verified_numeric_preserves_order_and_filters_numeric() {
        let x = cas_ast::ExprId::from_raw(11);
        let y = cas_ast::ExprId::from_raw(12);
        let two = cas_ast::ExprId::from_raw(2);
        let three = cas_ast::ExprId::from_raw(3);

        let out =
            merge_symbolic_with_verified_numeric(vec![x, y], vec![two, three], |id| id == three);
        assert_eq!(out, vec![x, y, three]);
    }
}
